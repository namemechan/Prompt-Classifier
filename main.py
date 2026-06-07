import sys
import os
import shutil
import time
import concurrent.futures
from PIL import Image

from PyQt5.QtWidgets import (
    QComboBox, QInputDialog, QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QTextEdit, QSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QDialog, QDoubleSpinBox,
    QSizePolicy, QFrame,
)
from PyQt5.QtCore import QThread, pyqtSignal
from settings_manager import SettingsManager
from image_utils import read_info_from_image, extract_prompt_blocks_from_image
from resolution import (
    make_res_folder_name,
    find_res_group,
    passes_res_filters,
    ResListDialog,
    SPECIAL_FOLDER_MAP,
)


# ──────────────────────────────────────────────────────────────────
#  유틸
# ──────────────────────────────────────────────────────────────────
def sanitize_for_path(name: str) -> str:
    for ch in r'<>:"/\\|?*':
        name = name.replace(ch, '_')
    return name[:100]




# ──────────────────────────────────────────────────────────────────
#  와일드카드 처리
# ──────────────────────────────────────────────────────────────────
def get_wildcard_base_dir() -> str:
    """main.py 또는 패키징된 exe 기준의 wildcard 폴더 경로를 반환."""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "wildcard")


def expand_wildcards(raw_text, enabled, _visited=None):
    """
    키워드 문자열을 파싱하여 최종 키워드 목록을 반환.
    - enabled=False : | 로만 분리, __태그__를 일반 키워드로 취급
    - enabled=True  : __태그__ 발견 시 wildcard/<태그>.txt 읽어 재귀 확장
    """
    if _visited is None:
        _visited = set()

    result = []
    parts = [p.strip() for p in raw_text.split('|') if p.strip()]

    if not enabled:
        return parts

    wildcard_dir = get_wildcard_base_dir()

    for part in parts:
        if part.startswith('__') and part.endswith('__') and len(part) > 4:
            tag = part[2:-2]
            if tag in _visited:
                result.append(part)
                continue
            txt_path = os.path.join(wildcard_dir, tag + '.txt')
            if not os.path.isfile(txt_path):
                continue
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                merged = '|'.join(line.strip() for line in lines if line.strip())
                result.extend(expand_wildcards(merged, True, _visited | {tag}))
            except (IOError, OSError):
                continue
        else:
            result.append(part)

    return result


def get_image_size(image_path: str):
    """(width, height) 반환. 실패 시 None."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
#  프롬프트 분류 – 멀티코어용 독립 함수
# ──────────────────────────────────────────────────────────────────
def process_single_image_task(image_path, keywords, exclude_keywords=None):
    img_file = os.path.basename(image_path)
    try:
        file_size = os.path.getsize(image_path)
        prompt_blocks = extract_prompt_blocks_from_image(image_path)
        if not prompt_blocks:
            return {"status": "no_prompt", "path": image_path,
                    "log": f"{img_file}: 프롬프트 데이터 없음"}

        exclude_keywords = [kw.replace('\\\\', '\\').lower() for kw in (exclude_keywords or []) if kw]
        keywords = [kw for kw in (keywords or []) if kw]

        for block in reversed(prompt_blocks):
            block_lower = block.replace('\\\\', '\\').lower()
            if exclude_keywords:
                for ex_kw in exclude_keywords:
                    if ex_kw in block_lower:
                        return {"status": "excluded", "path": image_path, "size": file_size,
                                "log": f"{img_file}: 제외 키워드 [{ex_kw}] 포함 — 건너뜀"}
            for keyword in keywords:
                if keyword.replace('\\\\', '\\').lower() in block_lower:
                    return {"status": "success", "path": image_path,
                            "keyword": keyword, "size": file_size}

        if not keywords:
            return {"status": "no_keyword_match", "path": image_path, "size": file_size}
        return {"status": "no_keyword_match", "path": image_path, "size": file_size,
                "log": f"{img_file}: 일치하는 키워드 없음"}
    except Exception as e:
        return {"status": "error", "path": image_path,
                "log": f"{img_file} 처리 중 오류: {str(e)}"}


# ──────────────────────────────────────────────────────────────────
#  워커 스레드
# ──────────────────────────────────────────────────────────────────
class ImageClassifierWorker(QThread):
    progress_updated          = pyqtSignal(int)
    log_updated               = pyqtSignal(str)
    completed                 = pyqtSignal(int)
    safe_mode_dialog_required = pyqtSignal(int, float)

    def __init__(self, source_dir, prompt_levels, rename_mode="keep", rename_custom_name="",
                 handle_others="off",
                 resolve_conflicts=False, multicore_enabled=False, multicore_core_count=4,
                 full_tracking_enabled=False, full_tracking_prompt="", full_tracking_exclude_prompt="",
                 custom_dest_enabled=False, custom_dest_path="",
                 safe_mode_enabled=False, clone_mode_enabled=False,
                 wildcard_enabled=False,
                 res_cfg: dict = None):
        super().__init__()
        self.source_dir                   = source_dir
        self.prompt_levels                = prompt_levels
        self.rename_mode                  = rename_mode          # "keep" | "prompt" | "custom"
        self.rename_custom_name           = rename_custom_name
        self.rename_images                = (rename_mode == "prompt")  # 내부 호환
        self.handle_others                = handle_others
        self.resolve_conflicts            = resolve_conflicts
        self.multicore_enabled            = multicore_enabled
        self.multicore_core_count         = multicore_core_count
        self.full_tracking_enabled        = full_tracking_enabled
        self.full_tracking_prompt         = full_tracking_prompt
        self.full_tracking_exclude_prompt = full_tracking_exclude_prompt
        self.custom_dest_enabled          = custom_dest_enabled
        self.custom_dest_path             = custom_dest_path
        self.safe_mode_enabled            = safe_mode_enabled
        self.clone_mode_enabled           = clone_mode_enabled
        self.wildcard_enabled             = wildcard_enabled
        self.res_cfg                      = res_cfg or {}
        self.canceled                     = False
        self.undo_info                    = []
        self.created_dirs                 = []
        self.processed_files_info         = []

    # ── 메인 실행 ──────────────────────────────────────────────────
    def run(self):
        op             = 'copy' if (self.safe_mode_enabled or self.clone_mode_enabled) else 'move'
        res_enabled    = self.res_cfg.get("res_enabled", False)
        res_standalone = self.res_cfg.get("res_standalone", False)
        res_timing     = self.res_cfg.get("res_timing", "after")

        # 단독 모드: 해상도만 실행
        if res_enabled and res_standalone:
            self.log_updated.emit("▶ [해상도 단독 분류] 실행합니다.")
            self._run_resolution_pass(op)
            if self.canceled:
                self.log_updated.emit("작업 취소됨.")
                self.completed.emit(0); return
        else:
            # 1) 해상도 분류 먼저
            if res_enabled and res_timing == "before":
                self.log_updated.emit("▶ [해상도 분류] 먼저 실행합니다.")
                self._run_resolution_pass(op)
                if self.canceled:
                    self.log_updated.emit("작업 취소됨.")
                    self.completed.emit(0); return
                self.log_updated.emit("▶ [프롬프트 분류] 시작합니다.")

            # 2) 프롬프트 분류
            self._run_prompt_pass(op)
            if self.canceled:
                self.log_updated.emit("작업 취소됨.")
                self.completed.emit(0); return

            # 3) 해상도 분류 나중
            if res_enabled and res_timing == "after":
                self.log_updated.emit("▶ [해상도 분류] 프롬프트 분류 완료 후 실행합니다.")
                self._run_resolution_pass(op)
                if self.canceled:
                    self.log_updated.emit("작업 취소됨.")
                    self.completed.emit(0); return

        if self.safe_mode_enabled:
            total_mb = sum(i['size'] for i in self.processed_files_info) / (1024 * 1024)
            self.safe_mode_dialog_required.emit(len(self.processed_files_info), total_mb)
        else:
            self.completed.emit(len(self.processed_files_info))

    # ── 프롬프트 분류 패스 ─────────────────────────────────────────
    def _run_prompt_pass(self, op):
        if self.full_tracking_enabled:
            self.log_updated.emit("전체추적 모드: 모든 하위 폴더 검색")
            images = self._find_all_image_files_recursive(self.source_dir)
            if not images:
                return
            keywords = expand_wildcards(self.full_tracking_prompt, self.wildcard_enabled)
            exc_kws  = expand_wildcards(self.full_tracking_exclude_prompt, self.wildcard_enabled)
            self._process_images_by_keywords(images, keywords, op,
                                             handle_others_flag=(self.handle_others != "off"),
                                             exclude_keywords=exc_kws)
        else:
            current_dirs = [self.source_dir]
            active_levels = []
            for idx, lv in enumerate(self.prompt_levels):
                en = lv[0]; pr = lv[1] if len(lv) > 1 else ""; ex = lv[2] if len(lv) > 2 else ""
                if en and pr.strip():
                    active_levels.append((idx, pr, ex))

            for level_idx, prompt_string, exclude_string in active_levels:
                if self.canceled: return
                self.log_updated.emit(f"레벨 {level_idx+1} 처리 중...")
                level_images = self._collect_level_images(current_dirs)
                if not level_images: break
                keywords = expand_wildcards(prompt_string, self.wildcard_enabled)
                exc_kws  = expand_wildcards(exclude_string, self.wildcard_enabled)
                next_dirs = self._process_images_by_keywords(level_images, keywords, op,
                                                              handle_others_flag=False,
                                                              exclude_keywords=exc_kws)

                # 각 레벨별 그 외 처리: 이 레벨에서 매칭 안 된 파일을 즉시 other로
                if self.handle_others == "per_level" and not self.canceled:
                    remaining = self._collect_level_images(current_dirs)
                    if remaining:
                        self.log_updated.emit(
                            f"레벨 {level_idx+1} 미분류 {len(remaining)}개 → 'other' 이동")
                        other_counters = {'other': 0}
                        for img_dir, img_file in remaining:
                            if self.canceled: break
                            path = os.path.join(img_dir, img_file)
                            self._process_image_file(img_dir, img_file, path,
                                                     os.path.getsize(path), 'other',
                                                     other_counters, op, no_rename=False)
                        # other 폴더도 다음 레벨 탐색 대상에 포함
                        other_dir = (self.custom_dest_path
                                     if self.custom_dest_enabled and self.custom_dest_path
                                     else os.path.join(self.source_dir, 'other'))
                        if next_dirs is None:
                            next_dirs = []
                        if os.path.isdir(other_dir) and other_dir not in next_dirs:
                            next_dirs.append(other_dir)

                if next_dirs:
                    current_dirs = next_dirs
                else:
                    break

            # 후행 그 외 처리: 모든 레벨 완료 후 소스 폴더에 남은 파일을 other로
            if self.handle_others == "after" and not self.canceled:
                remaining = self._collect_level_images([self.source_dir])
                if remaining:
                    self.log_updated.emit(f"{len(remaining)}개 미분류 파일 → 'other' 이동")
                    counters = {'other': 0}
                    for img_dir, img_file in remaining:
                        if self.canceled: break
                        path = os.path.join(img_dir, img_file)
                        self._process_image_file(img_dir, img_file, path,
                                                 os.path.getsize(path), 'other', counters, op,
                                                 no_rename=False)

    # ── 해상도 분류 패스 ──────────────────────────────────────────
    def _run_resolution_pass(self, op):
        """
        source_dir 전체를 재귀 탐색하여 해상도별로 분류.
        '후' 타이밍이면 이미 이동된 파일들이 하위 폴더에 있으므로 재귀 탐색이 맞음.
        """
        cfg        = self.res_cfg
        fmt        = cfg.get("res_folder_format", "wh")
        ignore_ori = cfg.get("res_ignore_orientation", False)
        min_cnt_en = cfg.get("res_min_count_enabled", False)
        min_cnt    = cfg.get("res_min_count", 3)

        images = self._find_all_image_files_recursive(self.source_dir)
        if not images:
            self.log_updated.emit("[해상도 분류] 처리할 이미지 없음")
            return

        # 해상도 읽기 + 필터링
        # { (rep_w, rep_h): [(img_dir, img_file, img_path, w, h), ...] }
        groups  = {}
        special = {}  # "too_small" | "too_large" | "blacklisted" | "no_size" → list
        total   = len(images)
        self.log_updated.emit(f"[해상도 분류] {total}개 이미지 해상도 분석 중...")

        for i, (img_dir, img_file) in enumerate(images):
            if self.canceled: return
            img_path = os.path.join(img_dir, img_file)
            size = get_image_size(img_path)
            if size is None:
                special.setdefault("no_size", []).append((img_dir, img_file, img_path, 0, 0))
                continue
            w, h = size

            ok, reason = passes_res_filters(w, h, cfg)
            if not ok:
                special.setdefault(reason, []).append((img_dir, img_file, img_path, w, h))
                continue

            rep = find_res_group(w, h, cfg)
            groups.setdefault(rep, []).append((img_dir, img_file, img_path, w, h))

            self.progress_updated.emit(int((i + 1) / total * 50))  # 0~50%

        # 최소 파일 수 미달 그룹 → special["below_min"]
        if min_cnt_en and min_cnt > 1:
            to_remove = []
            for rep, items in groups.items():
                if len(items) < min_cnt:
                    special.setdefault("below_min", []).extend(items)
                    to_remove.append(rep)
            for rep in to_remove:
                del groups[rep]

        # 실제 파일 이동
        counters   = {}
        total_move = sum(len(v) for v in groups.values()) + sum(len(v) for v in special.values())
        done       = 0

        for rep, items in groups.items():
            if self.canceled: return
            folder_name = make_res_folder_name(rep[0], rep[1], fmt, ignore_ori)
            for img_dir, img_file, img_path, w, h in items:
                if self.canceled: return
                if not os.path.exists(img_path):
                    continue
                file_size = os.path.getsize(img_path)
                self._process_image_file(img_dir, img_file, img_path, file_size,
                                         folder_name, counters, op, no_rename=True,
                                         extra_log_tag="해상도")
                done += 1
                self.progress_updated.emit(50 + int(done / max(total_move, 1) * 50))

        # special 폴더들 (resolution.py의 SPECIAL_FOLDER_MAP 사용)
        for reason, items in special.items():
            folder_name = SPECIAL_FOLDER_MAP.get(reason, f"res_{reason}")
            for img_dir, img_file, img_path, w, h in items:
                if self.canceled: return
                if not os.path.exists(img_path):
                    continue
                file_size = os.path.getsize(img_path)
                self._process_image_file(img_dir, img_file, img_path, file_size,
                                         folder_name, counters, op, no_rename=True,
                                         extra_log_tag="해상도-특수")
                done += 1
                self.progress_updated.emit(50 + int(done / max(total_move, 1) * 50))

        self.log_updated.emit(f"[해상도 분류] 완료. {done}개 파일 처리됨.")

    # ── 이미지 수집 ────────────────────────────────────────────────
    def _collect_level_images(self, directories):
        images = []
        for d in directories:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) \
                        and os.path.isfile(os.path.join(d, f)):
                    images.append((d, f))
        return images

    def _find_all_image_files_recursive(self, directory):
        images = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    images.append((root, f))
        return images

    # ── 프롬프트 기반 분류 ─────────────────────────────────────────
    def _process_images_by_keywords(self, images, keywords, operation_type,
                                    handle_others_flag, exclude_keywords=None):
        exclude_keywords = exclude_keywords or []
        total     = len(images)
        processed = 0
        next_dirs = []
        unmatched = []
        counters  = {k: 0 for k in keywords}
        paths     = [os.path.join(d, f) for d, f in images]

        if self.multicore_enabled and self.multicore_core_count > 1:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.multicore_core_count) as executor:
                future_map = {
                    executor.submit(process_single_image_task, p, keywords, exclude_keywords): p
                    for p in paths
                }
                for future in concurrent.futures.as_completed(future_map):
                    if self.canceled:
                        if sys.version_info >= (3, 9):
                            executor.shutdown(wait=False, cancel_futures=True)
                        else:
                            executor.shutdown(wait=False)
                        return []
                    try:
                        result    = future.result()
                        if result.get('log'): self.log_updated.emit(result['log'])
                        img_path  = result["path"]
                        file_size = result.get("size", 0)
                        img_dir   = os.path.dirname(img_path)
                        img_file  = os.path.basename(img_path)
                        if result["status"] == "success":
                            td = self._process_image_file(
                                img_dir, img_file, img_path, file_size,
                                result["keyword"], counters, operation_type)
                            if td and td not in next_dirs:
                                next_dirs.append(td)
                        else:
                            unmatched.append((img_dir, img_file, img_path, file_size))
                    except Exception as e:
                        self.log_updated.emit(f"처리 중 예외: {e}")
                    processed += 1
                    if total > 0:
                        self.progress_updated.emit(int(processed / total * 100))
        else:
            for path in paths:
                if self.canceled: return []
                result    = process_single_image_task(path, keywords, exclude_keywords)
                if result.get('log'): self.log_updated.emit(result['log'])
                img_path  = result["path"]
                file_size = result.get("size", 0)
                img_dir   = os.path.dirname(img_path)
                img_file  = os.path.basename(img_path)
                if result["status"] == "success":
                    td = self._process_image_file(
                        img_dir, img_file, img_path, file_size,
                        result["keyword"], counters, operation_type)
                    if td and td not in next_dirs:
                        next_dirs.append(td)
                else:
                    unmatched.append((img_dir, img_file, img_path, file_size))
                processed += 1
                if total > 0:
                    self.progress_updated.emit(int(processed / total * 100))

        if handle_others_flag and unmatched:
            other_counters = {'other': 0}
            for img_dir, img_file, img_path, file_size in unmatched:
                if self.canceled: break
                self._process_image_file(img_dir, img_file, img_path, file_size,
                                         'other', other_counters, operation_type)
        return next_dirs

    # ── 단일 파일 처리 ────────────────────────────────────────────
    def _process_image_file(self, img_dir, img_file, img_path, file_size,
                            keyword, counters, operation_type,
                            no_rename=False, extra_log_tag=None):
        """
        no_rename=True 이면 rename_images 설정과 무관하게 파일명 유지.
        해상도 분류는 항상 no_rename=True.
        """
        sanitized = sanitize_for_path(keyword)

        # 대상 폴더: custom_dest 가 켜져 있어도 해상도 분류(no_rename)는
        # 원본 파일이 있는 폴더 기준으로 해상도 하위폴더를 만든다.
        if no_rename:
            target_dir = os.path.join(img_dir, sanitized)
        else:
            target_dir = (self.custom_dest_path
                          if self.custom_dest_enabled and self.custom_dest_path
                          else os.path.join(img_dir, sanitized))

        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
            except OSError as e:
                self.log_updated.emit(f"폴더 생성 실패: {e}")
                return None

        counters.setdefault(keyword, 0)
        do_rename = (self.rename_mode != "keep") and not no_rename
        if do_rename:
            counters[keyword] += 1
            idx_str = str(counters[keyword]).zfill(6)
            ext = os.path.splitext(img_file)[1]
            if self.rename_mode == "custom" and self.rename_custom_name.strip():
                base_name = sanitize_for_path(self.rename_custom_name.strip())
            else:  # "prompt"
                base_name = sanitized
            dest_filename = f"{base_name}_{idx_str}{ext}"
        else:
            dest_filename = img_file

        dest_path = os.path.join(target_dir, dest_filename)

        if os.path.exists(dest_path):
            if not self.resolve_conflicts:
                self.log_updated.emit(f"⏭️ [{keyword}] {img_file} → 동일명 파일 존재, 건너뜀")
                return None
            base, ext = os.path.splitext(dest_path)
            c = 1
            while os.path.exists(f"{base} ({str(c).zfill(2)}){ext}"):
                c += 1
            dest_path = f"{base} ({str(c).zfill(2)}){ext}"

        try:
            if operation_type == 'copy':
                shutil.copy2(img_path, dest_path)
            else:
                shutil.move(img_path, dest_path)

            self.processed_files_info.append(
                {'src': img_path, 'dest': dest_path, 'size': file_size})
            if not self.safe_mode_enabled:
                self.undo_info.append(
                    {'src': img_path, 'dest': dest_path, 'op': operation_type})

            tag    = f"[{extra_log_tag}] " if extra_log_tag else ""
            action = "복사" if operation_type == 'copy' else "이동"
            self.log_updated.emit(f"✅ {tag}[{keyword}] {img_file} {action} 완료")
            return target_dir
        except Exception as e:
            self.log_updated.emit(f"파일 처리 실패: {e}")
            return None

    # ── 안전 모드 finalize ─────────────────────────────────────────
    def finalize_safe_mode(self, choice):
        logs = []
        if choice == "delete":
            logs.append("파일 무결성 검증 및 원본 삭제 진행...")
            for info in self.processed_files_info:
                try:
                    src, dest, size = info['src'], info['dest'], info['size']
                    if os.path.exists(dest) and os.path.getsize(dest) == size:
                        if os.path.exists(src): os.remove(src)
                        self.undo_info.append({'src': src, 'dest': dest, 'op': 'move'})
                    else:
                        logs.append(f"경고: 복사본 무결성 실패, 원본 보존: {src}")
                except Exception as e:
                    logs.append(f"오류: {info['src']} 삭제 실패: {e}")
        elif choice == "keep":
            for info in self.processed_files_info:
                self.undo_info.append(
                    {'src': info['src'], 'dest': info['dest'], 'op': 'copy'})
        elif choice == "undo":
            logs.append("복사본 삭제(실행 취소) 중...")
            for info in self.processed_files_info:
                try:
                    if os.path.exists(info['dest']): os.remove(info['dest'])
                except Exception:
                    pass
            self.undo_info = []

        count = len(self.processed_files_info) if choice != "undo" else 0
        return logs, count

    # ── 되돌리기 ──────────────────────────────────────────────────
    def undo_last_operation(self):
        if not self.undo_info: return
        success = 0
        for info in reversed(self.undo_info):
            try:
                dest, src, op_type = info['dest'], info['src'], info['op']
                if op_type == 'move' and os.path.exists(dest):
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dest, src)
                    success += 1
                elif op_type == 'copy' and os.path.exists(dest):
                    os.remove(dest)
                    success += 1
            except Exception:
                pass
        for d in reversed(self.created_dirs):
            try:
                if os.path.exists(d) and not os.listdir(d):
                    os.rmdir(d)
            except Exception:
                pass
        self.log_updated.emit(f"작업 취소 완료: {success}개 파일 복원됨.")
        self.undo_info            = []
        self.created_dirs         = []
        self.processed_files_info = []

    def cancel(self): self.canceled = True


# ──────────────────────────────────────────────────────────────────
#  메인 윈도우
# ──────────────────────────────────────────────────────────────────
class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Classifier")
        self.setGeometry(100, 100, 860, 700)
        self.settings_manager = SettingsManager()
        self.source_dir  = ""
        self.worker      = None
        self.start_time  = 0.0
        self._res_whitelist = []
        self._res_blacklist = []
        self.init_ui()

    # ── UI 구성 ────────────────────────────────────────────────────
    def init_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QVBoxLayout(); ml.setSpacing(6)

        # 소스 디렉토리
        dl = QHBoxLayout()
        dl.addWidget(QLabel("소스 디렉토리:"))
        self.dir_path_label = QLabel("디렉토리가 선택되지 않았습니다")
        self.dir_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        dl.addWidget(self.dir_path_label, 1)
        bb = QPushButton("찾아보기..."); bb.clicked.connect(self.browse_directory)
        dl.addWidget(bb); ml.addLayout(dl)

        # 프리셋
        pl = QHBoxLayout(); pl.addWidget(QLabel("프리셋:"))
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        pl.addWidget(self.preset_combo, 1)
        self.save_preset_btn   = QPushButton("저장"); self.save_preset_btn.clicked.connect(self.show_save_preset_dialog)
        self.delete_preset_btn = QPushButton("삭제"); self.delete_preset_btn.clicked.connect(self.delete_preset)
        pl.addWidget(self.save_preset_btn); pl.addWidget(self.delete_preset_btn)
        ml.addLayout(pl)

        # ── 옵션 행 0: 이름 처리 ───────────────────────────────────
        nl = QHBoxLayout()
        nl.addWidget(QLabel("이름 처리:"))
        self.rename_keep_radio   = QRadioButton("유지")
        self.rename_prompt_radio = QRadioButton("변경 (분류 프롬프트)")
        self.rename_custom_radio = QRadioButton("지정")
        self.rename_keep_radio.setChecked(True)
        self._rename_group = QButtonGroup(self)
        self._rename_group.addButton(self.rename_keep_radio,   0)
        self._rename_group.addButton(self.rename_prompt_radio, 1)
        self._rename_group.addButton(self.rename_custom_radio, 2)
        self._rename_group.buttonClicked.connect(self._toggle_rename_custom_input)
        nl.addWidget(self.rename_keep_radio)
        nl.addWidget(self.rename_prompt_radio)
        nl.addWidget(self.rename_custom_radio)
        self.rename_custom_input = QLineEdit()
        self.rename_custom_input.setPlaceholderText("사용자 지정 이름")
        self.rename_custom_input.setEnabled(False)
        self.rename_custom_input.setMaximumWidth(180)
        nl.addWidget(self.rename_custom_input)
        sep_wc = QFrame(); sep_wc.setFrameShape(QFrame.VLine); sep_wc.setFrameShadow(QFrame.Sunken)
        nl.addWidget(sep_wc)
        self.wildcard_check = QCheckBox("와일드카드 처리")
        self.wildcard_check.setToolTip(
            "키워드 입력란에서 __파일명__ 형식으로 외부 txt 파일의 키워드를 불러옵니다.\n"
            "wildcard 폴더는 exe(또는 main.py)와 같은 위치에 있어야 합니다."
        )
        nl.addWidget(self.wildcard_check)
        nl.addStretch()
        ml.addLayout(nl)

        # ── 옵션 행 1: 기본 이동 옵션 ──────────────────────────────
        ol = QHBoxLayout()
        ol.addWidget(QLabel("그 외 처리:"))
        self._handle_others_off_radio   = QRadioButton("안 함")
        self._handle_others_after_radio = QRadioButton("후행")
        self._handle_others_per_radio   = QRadioButton("각 레벨별")
        self._handle_others_off_radio.setChecked(True)
        self._handle_others_group = QButtonGroup(self)
        self._handle_others_group.addButton(self._handle_others_off_radio,   0)
        self._handle_others_group.addButton(self._handle_others_after_radio, 1)
        self._handle_others_group.addButton(self._handle_others_per_radio,   2)
        ol.addWidget(self._handle_others_off_radio)
        ol.addWidget(self._handle_others_after_radio)
        ol.addWidget(self._handle_others_per_radio)
        sep0 = QFrame(); sep0.setFrameShape(QFrame.VLine); sep0.setFrameShadow(QFrame.Sunken)
        ol.addWidget(sep0)
        self.resolve_conflicts_check = QCheckBox("동일명 숫자 추가");  ol.addWidget(self.resolve_conflicts_check)
        self.safe_mode_check         = QCheckBox("안전 모드")
        self.safe_mode_check.toggled.connect(self._toggle_safety_modes); ol.addWidget(self.safe_mode_check)
        self.clone_mode_check        = QCheckBox("복제 모드")
        self.clone_mode_check.toggled.connect(self._toggle_safety_modes); ol.addWidget(self.clone_mode_check)
        # 멀티코어 – 같은 행 오른쪽
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setFrameShadow(QFrame.Sunken)
        ol.addWidget(sep)
        self.multicore_check = QCheckBox("멀티코어")
        self.multicore_check.toggled.connect(self._toggle_multicore_input)
        ol.addWidget(self.multicore_check)
        self.core_count_spinbox = QSpinBox()
        self.core_count_spinbox.setRange(1, os.cpu_count() or 1)
        self.core_count_spinbox.setSuffix(" 코어")
        self.core_count_spinbox.setEnabled(False)
        ol.addWidget(self.core_count_spinbox)
        ol.addStretch(); ml.addLayout(ol)

        # ── 전체추적 ───────────────────────────────────────────────
        fl = QHBoxLayout()
        self.full_tracking_check = QCheckBox("전체추적")
        self.full_tracking_check.toggled.connect(self._toggle_full_tracking_input)
        fl.addWidget(self.full_tracking_check)
        self.full_tracking_prompt_input = QLineEdit()
        self.full_tracking_prompt_input.setPlaceholderText("포함 키워드 (| 구분)")
        fl.addWidget(self.full_tracking_prompt_input, 1)
        fl.addWidget(QLabel("제외:"))
        self.full_tracking_exclude_input = QLineEdit()
        self.full_tracking_exclude_input.setPlaceholderText("제외 키워드 (| 구분)")
        fl.addWidget(self.full_tracking_exclude_input, 1)
        ml.addLayout(fl)

        # ── 사용자 폴더 ────────────────────────────────────────────
        cdl = QHBoxLayout()
        self.custom_dest_check = QCheckBox("사용자 폴더 사용")
        self.custom_dest_check.toggled.connect(self._toggle_custom_dest_input)
        cdl.addWidget(self.custom_dest_check)
        self.custom_dest_path_input = QLineEdit()
        self.custom_dest_path_input.setPlaceholderText("이동할 폴더 경로")
        cdl.addWidget(self.custom_dest_path_input, 1)
        cb = QPushButton("찾아보기..."); cb.clicked.connect(self._browse_custom_dest_directory)
        cdl.addWidget(cb); ml.addLayout(cdl)

        # ── 레벨 1~5 ──────────────────────────────────────────────
        self.prompt_inputs = []
        for i in range(5):
            ll = QHBoxLayout()
            lc = QCheckBox(f"레벨 {i+1}:"); lc.setChecked(i == 0); ll.addWidget(lc)
            pi = QLineEdit(); pi.setPlaceholderText("포함 키워드 (| 구분)"); ll.addWidget(pi, 1)
            ll.addWidget(QLabel("제외:"))
            ei = QLineEdit(); ei.setPlaceholderText("제외 키워드 (| 구분)"); ll.addWidget(ei, 1)
            ml.addLayout(ll); self.prompt_inputs.append((lc, pi, ei))

        # ── 해상도 분류 그룹박스 ───────────────────────────────────
        ml.addWidget(self._build_res_group())

        # 진행 표시줄 / 로그
        self.progress_bar = QProgressBar(); ml.addWidget(self.progress_bar)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); ml.addWidget(self.log_text)

        # 버튼 행
        bl = QHBoxLayout()
        self.start_btn  = QPushButton("분류 시작"); self.start_btn.clicked.connect(self.start_classification)
        self.cancel_btn = QPushButton("취소"); self.cancel_btn.clicked.connect(self.cancel_classification)
        self.cancel_btn.setEnabled(False)
        self.undo_btn   = QPushButton("이전 작업 취소"); self.undo_btn.clicked.connect(self.undo_last_operation)
        bl.addWidget(self.start_btn); bl.addWidget(self.cancel_btn); bl.addWidget(self.undo_btn)
        ml.addLayout(bl)

        cw.setLayout(ml)

    def _build_res_group(self):
        gb = QGroupBox("해상도 분류")
        gb.setCheckable(True); gb.setChecked(False)
        gb.toggled.connect(self._toggle_res_group)
        self._res_groupbox = gb
        vl = QVBoxLayout(gb)
        vl.setSpacing(4)

        # 행 1: 단독 분류 + 분류 타이밍 + 폴더명 형식
        row1 = QHBoxLayout()
        self.res_standalone_check = QCheckBox("단독 분류 (프롬프트 분류 없이 해상도만)")
        self.res_standalone_check.toggled.connect(self._toggle_res_standalone)
        row1.addWidget(self.res_standalone_check)
        row1.addStretch()
        vl.addLayout(row1)

        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("분류 타이밍:"))
        self.res_timing_before = QRadioButton("선(먼저)")
        self.res_timing_after  = QRadioButton("후(나중에)")
        self.res_timing_after.setChecked(True)
        self._res_timing_group = QButtonGroup(self)
        self._res_timing_group.addButton(self.res_timing_before, 0)
        self._res_timing_group.addButton(self.res_timing_after,  1)
        row1b.addWidget(self.res_timing_before); row1b.addWidget(self.res_timing_after)

        row1b.addWidget(QLabel("  폴더명 형식:"))
        self.res_fmt_wh     = QRadioButton("1216x832")
        self.res_fmt_wh_dir = QRadioButton("1216x832_(가로/세로)")
        self.res_fmt_ratio  = QRadioButton("16:9 비율")
        self.res_fmt_wh.setChecked(True)
        self._res_fmt_group = QButtonGroup(self)
        self._res_fmt_group.addButton(self.res_fmt_wh,     0)
        self._res_fmt_group.addButton(self.res_fmt_wh_dir, 1)
        self._res_fmt_group.addButton(self.res_fmt_ratio,  2)
        row1b.addWidget(self.res_fmt_wh); row1b.addWidget(self.res_fmt_wh_dir)
        row1b.addWidget(self.res_fmt_ratio); row1b.addStretch()
        vl.addLayout(row1b)

        # 행 2: 비율 분류 + 가로세로 통일
        row2 = QHBoxLayout()
        self.res_group_ratio_check = QCheckBox("비율 분류 (동일 비율 묶기)")
        self.res_ignore_ori_check  = QCheckBox("가로세로 통일 (방향 무시)")
        row2.addWidget(self.res_group_ratio_check)
        row2.addWidget(self.res_ignore_ori_check)
        row2.addStretch(); vl.addLayout(row2)

        # 행 3: 허용 오차
        row3 = QHBoxLayout()
        self.res_tol_check = QCheckBox("허용 오차")
        self.res_tol_check.toggled.connect(self._toggle_res_tolerance)
        row3.addWidget(self.res_tol_check)
        self.res_tol_pct_radio = QRadioButton("% 설정")
        self.res_tol_px_radio  = QRadioButton("픽셀 설정")
        self.res_tol_pct_radio.setChecked(True)
        self._res_tol_mode_group = QButtonGroup(self)
        self._res_tol_mode_group.addButton(self.res_tol_pct_radio, 0)
        self._res_tol_mode_group.addButton(self.res_tol_px_radio,  1)
        self.res_tol_pct_radio.setEnabled(False); self.res_tol_px_radio.setEnabled(False)
        row3.addWidget(self.res_tol_pct_radio); row3.addWidget(self.res_tol_px_radio)
        self.res_tol_spinbox = QDoubleSpinBox()
        self.res_tol_spinbox.setRange(0, 9999); self.res_tol_spinbox.setValue(5)
        self.res_tol_spinbox.setDecimals(1); self.res_tol_spinbox.setEnabled(False)
        row3.addWidget(self.res_tol_spinbox); row3.addStretch(); vl.addLayout(row3)

        # 행 4: 최소 파일 수 + 컷오프
        row4 = QHBoxLayout()
        self.res_min_cnt_check = QCheckBox("최소 파일 수")
        self.res_min_cnt_check.toggled.connect(lambda c: self.res_min_cnt_spin.setEnabled(c))
        self.res_min_cnt_spin  = QSpinBox()
        self.res_min_cnt_spin.setRange(1, 9999); self.res_min_cnt_spin.setValue(3)
        self.res_min_cnt_spin.setSuffix("개 미만")
        self.res_min_cnt_spin.setToolTip("이 수 미만의 파일이 있는 해상도 그룹은 'res_below_min' 폴더로 이동합니다.")
        self.res_min_cnt_spin.setEnabled(False)
        row4.addWidget(self.res_min_cnt_check); row4.addWidget(self.res_min_cnt_spin)

        row4.addSpacing(16)
        self.res_cutoff_check = QCheckBox("해상도 컷오프")
        self.res_cutoff_check.toggled.connect(self._toggle_res_cutoff)
        row4.addWidget(self.res_cutoff_check)
        row4.addWidget(QLabel("최소 픽셀:"))
        self.res_cutoff_min_spin = QSpinBox()
        self.res_cutoff_min_spin.setRange(0, 999_999_999); self.res_cutoff_min_spin.setValue(0)
        self.res_cutoff_min_spin.setSuffix(" px²"); self.res_cutoff_min_spin.setEnabled(False)
        row4.addWidget(self.res_cutoff_min_spin)
        row4.addWidget(QLabel("최대 픽셀:"))
        self.res_cutoff_max_spin = QSpinBox()
        self.res_cutoff_max_spin.setRange(0, 999_999_999); self.res_cutoff_max_spin.setValue(0)
        self.res_cutoff_max_spin.setSuffix(" px²"); self.res_cutoff_max_spin.setEnabled(False)
        row4.addWidget(self.res_cutoff_max_spin)
        row4.addStretch(); vl.addLayout(row4)

        # 행 5: 화이트/블랙리스트
        row5 = QHBoxLayout()
        self.res_wl_check = QCheckBox("화이트리스트 사용")
        self.res_bl_check = QCheckBox("블랙리스트 사용")
        wl_btn = QPushButton("화이트리스트 편집...")
        bl_btn = QPushButton("블랙리스트 편집...")
        wl_btn.clicked.connect(self._edit_whitelist)
        bl_btn.clicked.connect(self._edit_blacklist)
        row5.addWidget(self.res_wl_check); row5.addWidget(wl_btn)
        row5.addSpacing(8)
        row5.addWidget(self.res_bl_check); row5.addWidget(bl_btn)
        row5.addStretch(); vl.addLayout(row5)

        return gb

    # ── 해상도 그룹 토글 ──────────────────────────────────────────
    def _toggle_res_group(self, checked):
        pass  # QGroupBox 자체가 체크박스이므로 자동 처리

    def _toggle_res_standalone(self, checked):
        """단독 분류 ON → 타이밍 라디오버튼 비활성(의미 없음)."""
        self.res_timing_before.setEnabled(not checked)
        self.res_timing_after.setEnabled(not checked)

    def _toggle_rename_custom_input(self, btn):
        is_custom = (self._rename_group.checkedId() == 2)
        self.rename_custom_input.setEnabled(is_custom)

    def _toggle_res_tolerance(self, checked):
        self.res_tol_pct_radio.setEnabled(checked)
        self.res_tol_px_radio.setEnabled(checked)
        self.res_tol_spinbox.setEnabled(checked)

    def _toggle_res_cutoff(self, checked):
        self.res_cutoff_min_spin.setEnabled(checked)
        self.res_cutoff_max_spin.setEnabled(checked)

    def _edit_whitelist(self):
        dlg = ResListDialog("화이트리스트 편집", self._res_whitelist, self)
        if dlg.exec_() == QDialog.Accepted:
            self._res_whitelist = dlg.get_items()

    def _edit_blacklist(self):
        dlg = ResListDialog("블랙리스트 편집", self._res_blacklist, self)
        if dlg.exec_() == QDialog.Accepted:
            self._res_blacklist = dlg.get_items()

    # ── 해상도 cfg 딕셔너리 수집 ──────────────────────────────────
    def _collect_res_cfg(self):
        fmt_map  = {0: "wh", 1: "wh_dir", 2: "ratio"}
        fmt_id   = self._res_fmt_group.checkedId()
        tol_mode = "percent" if self._res_tol_mode_group.checkedId() == 0 else "pixel"
        return {
            "res_enabled":            self._res_groupbox.isChecked(),
            "res_standalone":         self.res_standalone_check.isChecked(),
            "res_timing":             "before" if self._res_timing_group.checkedId() == 0 else "after",
            "res_folder_format":      fmt_map.get(fmt_id, "wh"),
            "res_group_by_ratio":     self.res_group_ratio_check.isChecked(),
            "res_ignore_orientation": self.res_ignore_ori_check.isChecked(),
            "res_tolerance_enabled":  self.res_tol_check.isChecked(),
            "res_tolerance_mode":     tol_mode,
            "res_tolerance_value":    int(self.res_tol_spinbox.value()),
            "res_min_count_enabled":  self.res_min_cnt_check.isChecked(),
            "res_min_count":          self.res_min_cnt_spin.value(),
            "res_cutoff_enabled":     self.res_cutoff_check.isChecked(),
            "res_cutoff_min":         self.res_cutoff_min_spin.value(),
            "res_cutoff_max":         self.res_cutoff_max_spin.value(),
            "res_whitelist_enabled":  self.res_wl_check.isChecked(),
            "res_blacklist_enabled":  self.res_bl_check.isChecked(),
            "res_whitelist":          list(self._res_whitelist),
            "res_blacklist":          list(self._res_blacklist),
        }

    def _apply_res_cfg(self, cfg):
        self._res_groupbox.setChecked(cfg.get("res_enabled", False))

        standalone = cfg.get("res_standalone", False)
        self.res_standalone_check.setChecked(standalone)
        self._toggle_res_standalone(standalone)

        timing = cfg.get("res_timing", "after")
        if timing == "before": self.res_timing_before.setChecked(True)
        else:                  self.res_timing_after.setChecked(True)

        fmt = cfg.get("res_folder_format", "wh")
        if fmt == "wh_dir":  self.res_fmt_wh_dir.setChecked(True)
        elif fmt == "ratio": self.res_fmt_ratio.setChecked(True)
        else:                self.res_fmt_wh.setChecked(True)

        self.res_group_ratio_check.setChecked(cfg.get("res_group_by_ratio", False))
        self.res_ignore_ori_check.setChecked(cfg.get("res_ignore_orientation", False))

        tol_en = cfg.get("res_tolerance_enabled", False)
        self.res_tol_check.setChecked(tol_en)
        self._toggle_res_tolerance(tol_en)
        if cfg.get("res_tolerance_mode", "percent") == "pixel":
            self.res_tol_px_radio.setChecked(True)
        else:
            self.res_tol_pct_radio.setChecked(True)
        self.res_tol_spinbox.setValue(cfg.get("res_tolerance_value", 5))

        min_en = cfg.get("res_min_count_enabled", False)
        self.res_min_cnt_check.setChecked(min_en)
        self.res_min_cnt_spin.setEnabled(min_en)
        self.res_min_cnt_spin.setValue(cfg.get("res_min_count", 3))

        cut_en = cfg.get("res_cutoff_enabled", False)
        self.res_cutoff_check.setChecked(cut_en)
        self._toggle_res_cutoff(cut_en)
        self.res_cutoff_min_spin.setValue(cfg.get("res_cutoff_min", 0))
        self.res_cutoff_max_spin.setValue(cfg.get("res_cutoff_max", 0))

        self.res_wl_check.setChecked(cfg.get("res_whitelist_enabled", False))
        self.res_bl_check.setChecked(cfg.get("res_blacklist_enabled", False))
        self._res_whitelist = list(cfg.get("res_whitelist", []))
        self._res_blacklist = list(cfg.get("res_blacklist", []))

    # ── 토글 헬퍼 ─────────────────────────────────────────────────
    def _toggle_safety_modes(self, checked):
        s = self.sender()
        if checked:
            if s == self.safe_mode_check:    self.clone_mode_check.setEnabled(False)
            elif s == self.clone_mode_check: self.safe_mode_check.setEnabled(False)
        else:
            self.safe_mode_check.setEnabled(True); self.clone_mode_check.setEnabled(True)

    def _toggle_multicore_input(self, c):
        self.core_count_spinbox.setEnabled(c)

    def _toggle_full_tracking_input(self, c):
        self.full_tracking_prompt_input.setEnabled(c)
        self.full_tracking_exclude_input.setEnabled(c)
        for lc, pi, ei in self.prompt_inputs:
            lc.setEnabled(not c); pi.setEnabled(not c); ei.setEnabled(not c)

    def _toggle_custom_dest_input(self, c):
        self.custom_dest_path_input.setEnabled(c)

    def _browse_custom_dest_directory(self):
        dp = QFileDialog.getExistingDirectory(self, "대상 폴더 선택")
        if dp: self.custom_dest_path_input.setText(dp)

    def browse_directory(self):
        dp = QFileDialog.getExistingDirectory(self, "소스 폴더 선택")
        if dp: self.source_dir = dp; self.dir_path_label.setText(dp)

    # ── 설정 저장/불러오기 ────────────────────────────────────────
    def save_current_settings(self):
        pl         = [(c.isChecked(), i.text(), e.text()) for c, i, e in self.prompt_inputs]
        res        = self._collect_res_cfg()
        rename_mode   = ["keep", "prompt", "custom"][self._rename_group.checkedId()]
        rename_custom = self.rename_custom_input.text()
        s = self.settings_manager.create_settings_from_ui(
            self.source_dir,
            rename_mode, rename_custom,
            ["off", "after", "per_level"][self._handle_others_group.checkedId()],
            self.resolve_conflicts_check.isChecked(),
            self.multicore_check.isChecked(), self.core_count_spinbox.value(),
            pl,
            self.full_tracking_check.isChecked(), self.full_tracking_prompt_input.text(),
            self.full_tracking_exclude_input.text(),
            self.custom_dest_check.isChecked(), self.custom_dest_path_input.text(),
            self.safe_mode_check.isChecked(), self.clone_mode_check.isChecked(),
            self.wildcard_check.isChecked(),
            res["res_enabled"], res["res_standalone"], res["res_timing"], res["res_folder_format"],
            res["res_group_by_ratio"], res["res_ignore_orientation"],
            res["res_tolerance_enabled"], res["res_tolerance_mode"], res["res_tolerance_value"],
            res["res_min_count_enabled"], res["res_min_count"],
            res["res_cutoff_enabled"], res["res_cutoff_min"], res["res_cutoff_max"],
            res["res_whitelist_enabled"], res["res_blacklist_enabled"],
            res["res_whitelist"], res["res_blacklist"],
        )
        self.settings_manager.save_settings(s)

    def load_settings(self):
        s = self.settings_manager.get_settings_for_ui()
        self.source_dir = s[0]; self.dir_path_label.setText(s[0] or "디렉토리가 선택되지 않았습니다")
        # 이름 처리 라디오버튼 (s[1]=rename_mode, s[2]=rename_custom_name)
        mode_map = {"keep": 0, "prompt": 1, "custom": 2}
        mode_id  = mode_map.get(s[1], 0)
        btn = self._rename_group.button(mode_id)
        if btn: btn.setChecked(True)
        self.rename_custom_input.setText(s[2])
        self.rename_custom_input.setEnabled(mode_id == 2)

        ho_map = {"off": 0, "after": 1, "per_level": 2}
        ho_btn = self._handle_others_group.button(ho_map.get(s[3], 0))
        if ho_btn: ho_btn.setChecked(True)
        self.resolve_conflicts_check.setChecked(s[4])
        self.multicore_check.setChecked(s[5]); self.core_count_spinbox.setValue(s[6])
        self._toggle_multicore_input(s[5])
        self.full_tracking_check.setChecked(s[8])
        self.full_tracking_prompt_input.setText(s[9])
        self.full_tracking_exclude_input.setText(s[10])
        self._toggle_full_tracking_input(s[8])
        self.custom_dest_check.setChecked(s[11]); self.custom_dest_path_input.setText(s[12])
        self._toggle_custom_dest_input(s[11])
        self.safe_mode_check.setChecked(s[13]); self.clone_mode_check.setChecked(s[14])
        self.wildcard_check.setChecked(s[15])
        for i, (chk, inp, exc) in enumerate(self.prompt_inputs):
            if i < len(s[7]):
                chk.setChecked(s[7][i][0]); inp.setText(s[7][i][1]); exc.setText(s[7][i][2])
        # 해상도 설정 (s[16]~)
        cfg_keys = [
            "res_enabled", "res_standalone", "res_timing", "res_folder_format",
            "res_group_by_ratio", "res_ignore_orientation",
            "res_tolerance_enabled", "res_tolerance_mode", "res_tolerance_value",
            "res_min_count_enabled", "res_min_count",
            "res_cutoff_enabled", "res_cutoff_min", "res_cutoff_max",
            "res_whitelist_enabled", "res_blacklist_enabled", "res_whitelist", "res_blacklist",
        ]
        res_cfg = {k: s[16 + i] for i, k in enumerate(cfg_keys)}
        self._apply_res_cfg(res_cfg)
        self.update_preset_list()

    def update_preset_list(self):
        self.preset_combo.clear(); self.preset_combo.addItem("기본 설정")
        for p in self.settings_manager.get_preset_list():
            self.preset_combo.addItem(p)

    def load_preset(self, index):
        if index <= 0: return
        p = self.settings_manager.load_preset(self.preset_combo.currentText())
        self.source_dir = p.get("source_directory", "")
        self.dir_path_label.setText(self.source_dir or "없음")
        # 이름 처리
        mode_map = {"keep": 0, "prompt": 1, "custom": 2}
        # 하위 호환: rename_images(bool) 도 처리
        if "rename_mode" in p:
            mode_id = mode_map.get(p.get("rename_mode", "keep"), 0)
        else:
            mode_id = 1 if p.get("rename_images", False) else 0
        btn = self._rename_group.button(mode_id)
        if btn: btn.setChecked(True)
        self.rename_custom_input.setText(p.get("rename_custom_name", ""))
        self.rename_custom_input.setEnabled(mode_id == 2)
        ho_map = {"off": 0, "after": 1, "per_level": 2}
        ho_val = p.get("handle_others", "off")
        # 하위 호환: bool이 저장된 경우
        if isinstance(ho_val, bool):
            ho_val = "after" if ho_val else "off"
        ho_btn = self._handle_others_group.button(ho_map.get(ho_val, 0))
        if ho_btn: ho_btn.setChecked(True)
        self.resolve_conflicts_check.setChecked(p.get("resolve_conflicts", False))

        self.safe_mode_check.blockSignals(True); self.clone_mode_check.blockSignals(True)
        self.safe_mode_check.setChecked(p.get("safe_mode_enabled", False))
        self.clone_mode_check.setChecked(p.get("clone_mode_enabled", False))
        self.safe_mode_check.blockSignals(False); self.clone_mode_check.blockSignals(False)
        self.safe_mode_check.setEnabled(not self.clone_mode_check.isChecked())
        self.clone_mode_check.setEnabled(not self.safe_mode_check.isChecked())

        self.wildcard_check.setChecked(p.get("wildcard_enabled", False))

        self.multicore_check.setChecked(p.get("multicore_enabled", False))
        self.core_count_spinbox.setValue(p.get("multicore_core_count", os.cpu_count() or 4))
        self._toggle_multicore_input(p.get("multicore_enabled", False))

        self.full_tracking_check.setChecked(p.get("full_tracking_enabled", False))
        self.full_tracking_prompt_input.setText(p.get("full_tracking_prompt", ""))
        self.full_tracking_exclude_input.setText(p.get("full_tracking_exclude_prompt", ""))
        self._toggle_full_tracking_input(p.get("full_tracking_enabled", False))

        self.custom_dest_check.setChecked(p.get("custom_dest_enabled", False))
        self.custom_dest_path_input.setText(p.get("custom_dest_path", ""))
        self._toggle_custom_dest_input(p.get("custom_dest_enabled", False))

        pl = p.get("prompt_levels", [])
        for i, (c, inp, exc) in enumerate(self.prompt_inputs):
            if i < len(pl):
                c.setChecked(pl[i].get("enabled", False))
                inp.setText(pl[i].get("prompt", ""))
                exc.setText(pl[i].get("exclude_prompt", ""))
            else:
                c.setChecked(False); inp.setText(""); exc.setText("")

        self._apply_res_cfg(p)

    def show_save_preset_dialog(self):
        name, ok = QInputDialog.getText(self, "저장", "프리셋 이름:")
        if ok and name:
            self.save_current_settings()
            if self.settings_manager.save_preset(name):
                self.update_preset_list()
                idx = self.preset_combo.findText(name)
                if idx >= 0: self.preset_combo.setCurrentIndex(idx)

    def delete_preset(self):
        if self.preset_combo.currentIndex() <= 0: return
        name = self.preset_combo.currentText()
        if QMessageBox.question(self, '삭제', f"'{name}' 삭제?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            if self.settings_manager.delete_preset(name):
                self.preset_combo.removeItem(self.preset_combo.currentIndex())

    # ── 분류 시작 ─────────────────────────────────────────────────
    def start_classification(self):
        if not self.source_dir:
            QMessageBox.warning(self, "경고", "소스 디렉토리를 선택하세요."); return

        self.save_current_settings()
        self.start_time = time.time()
        pl      = [(c.isChecked(), i.text(), e.text()) for c, i, e in self.prompt_inputs]
        res_cfg = self._collect_res_cfg()

        handle_others_val = ["off", "after", "per_level"][self._handle_others_group.checkedId()]

        has_prompt_task = (
            self.full_tracking_check.isChecked()
            or any(en for en, _, __ in pl)
            or handle_others_val != "off"
        )
        res_standalone = res_cfg.get("res_standalone", False)
        if not has_prompt_task and not res_cfg["res_enabled"]:
            QMessageBox.information(self, "알림", "활성화된 분류 작업이 없습니다."); return
        if res_cfg["res_enabled"] and res_standalone and not has_prompt_task:
            pass  # 해상도 단독 모드는 프롬프트 작업 없어도 OK
        elif not has_prompt_task and not res_cfg["res_enabled"]:
            QMessageBox.information(self, "알림", "활성화된 분류 작업이 없습니다."); return

        rename_mode   = ["keep", "prompt", "custom"][self._rename_group.checkedId()]
        rename_custom = self.rename_custom_input.text()

        self.start_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0); self.log_text.clear()

        self.worker = ImageClassifierWorker(
            self.source_dir, pl,
            rename_mode, rename_custom,
            handle_others_val,
            self.resolve_conflicts_check.isChecked(),
            self.multicore_check.isChecked(), self.core_count_spinbox.value(),
            self.full_tracking_check.isChecked(), self.full_tracking_prompt_input.text(),
            self.full_tracking_exclude_input.text(),
            self.custom_dest_check.isChecked(), self.custom_dest_path_input.text(),
            self.safe_mode_check.isChecked(), self.clone_mode_check.isChecked(),
            self.wildcard_check.isChecked(),
            res_cfg=res_cfg,
        )
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(lambda m: self.log_text.append(m))
        self.worker.completed.connect(self.classification_completed)
        self.worker.safe_mode_dialog_required.connect(self.show_safe_mode_popup)
        self.worker.start()

    def show_safe_mode_popup(self, count, total_size_mb):
        if count == 0:
            logs, count = self.worker.finalize_safe_mode("undo")
            for m in logs: self.log_text.append(m)
            self.classification_completed(count); return
        msg = QMessageBox(self); msg.setWindowTitle("안전 모드")
        msg.setText(f"복사 완료: {count}개, {total_size_mb:.2f} MB\n원본 파일을 어떻게 처리할까요?")
        d_btn = msg.addButton("원본 삭제",            QMessageBox.YesRole)
        k_btn = msg.addButton("보존",                 QMessageBox.NoRole)
        u_btn = msg.addButton("실행 취소(복사본 삭제)", QMessageBox.RejectRole)
        msg.exec_()
        cb     = msg.clickedButton()
        choice = "delete" if cb == d_btn else ("keep" if cb == k_btn else "undo")
        logs, count = self.worker.finalize_safe_mode(choice)
        for m in logs: self.log_text.append(m)
        self.classification_completed(count)

    def cancel_classification(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel(); self.log_text.append("취소 중..."); self.cancel_btn.setEnabled(False)

    def classification_completed(self, count):
        elapsed = time.time() - self.start_time
        self.log_text.append(f"✔ 완료! ({elapsed:.2f}초, {count}개 파일 처리)")
        self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.undo_btn.setEnabled(True); self.progress_bar.setValue(100)

    def undo_last_operation(self):
        if not self.worker or not self.worker.undo_info:
            QMessageBox.warning(self, "경고", "취소할 작업이 없습니다."); return
        if QMessageBox.question(self, '취소', "마지막 작업을 취소하시겠습니까?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.worker.undo_last_operation()

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, '종료', "작업 진행 중입니다. 종료하시겠습니까?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.worker.cancel(); self.worker.wait(); e.accept()
            else:
                e.ignore()
        else:
            self.save_current_settings(); e.accept()


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform.startswith('win'):
        import multiprocessing; multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = ImageClassifierApp(); window.load_settings(); window.show()
    sys.exit(app.exec_())
