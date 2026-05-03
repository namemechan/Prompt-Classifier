import sys
import os
import shutil
import time
import concurrent.futures
from PyQt5.QtWidgets import (QComboBox, QInputDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, QProgressBar,
                            QMessageBox, QTextEdit, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from settings_manager import SettingsManager
from image_utils import read_info_from_image

def sanitize_for_path(name: str) -> str:
    illegal_chars = r'<>:"/\\|?*'
    sanitized = name
    for char in illegal_chars: sanitized = sanitized.replace(char, '_')
    return sanitized[:100]  # 긴 경로 방지 (Windows MAX_PATH 대비)

def process_single_image_task(image_path, keywords):
    img_file = os.path.basename(image_path)
    try:
        file_size = os.path.getsize(image_path)
        prompt_data = read_info_from_image(image_path)
        if not prompt_data:
            return {"status": "no_prompt", "path": image_path, "log": f"{img_file}: 프롬프트 데이터 없음"}
        if not keywords:
            return {"status": "no_keyword_match", "path": image_path, "size": file_size}
        for keyword in keywords:
            if keyword.lower() in prompt_data.lower():
                return {"status": "success", "path": image_path, "keyword": keyword, "size": file_size}
        return {"status": "no_keyword_match", "path": image_path, "size": file_size, "log": f"{img_file}: 일치하는 키워드 없음"}
    except Exception as e:
        return {"status": "error", "path": image_path, "log": f"{img_file} 처리 중 오류: {str(e)}"}


class ImageClassifierWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    completed = pyqtSignal(int)
    safe_mode_dialog_required = pyqtSignal(int, float)

    def __init__(self, source_dir, prompt_levels, rename_images=False, handle_others=False, resolve_conflicts=False,
                 multicore_enabled=False, multicore_core_count=4, full_tracking_enabled=False, full_tracking_prompt="", 
                 custom_dest_enabled=False, custom_dest_path="", safe_mode_enabled=False, clone_mode_enabled=False):
        super().__init__()
        self.source_dir = source_dir
        self.prompt_levels = prompt_levels
        self.rename_images, self.handle_others, self.resolve_conflicts = rename_images, handle_others, resolve_conflicts
        self.multicore_enabled, self.multicore_core_count = multicore_enabled, multicore_core_count
        self.full_tracking_enabled, self.full_tracking_prompt = full_tracking_enabled, full_tracking_prompt
        self.custom_dest_enabled, self.custom_dest_path = custom_dest_enabled, custom_dest_path
        self.safe_mode_enabled, self.clone_mode_enabled = safe_mode_enabled, clone_mode_enabled
        self.canceled = False
        self.undo_info, self.created_dirs, self.processed_files_info = [], [], []

    def run(self):
        operation_type = 'copy' if self.safe_mode_enabled or self.clone_mode_enabled else 'move'

        if self.full_tracking_enabled:
            self.log_updated.emit("전체추적 모드 활성화: 모든 하위 폴더의 이미지를 검색합니다.")
            images = self._find_all_image_files_recursive(self.source_dir)
            if not images:
                self.completed.emit(0); return
            prompt_keywords = [p.strip() for p in self.full_tracking_prompt.split('|') if p.strip()]
            self._process_images_by_keywords(images, prompt_keywords, operation_type, handle_others_flag=self.handle_others)
        else:
            current_dirs = [self.source_dir]
            active_levels = [(idx, prompt) for idx, (enabled, prompt) in enumerate(self.prompt_levels) if enabled and prompt.strip()]
            
            for level_idx, prompt_string in active_levels:
                self.log_updated.emit(f"레벨 {level_idx+1} 처리 중...")
                level_images = self._collect_level_images(current_dirs)
                if not level_images: break
                
                keywords = [p.strip() for p in prompt_string.split('|') if p.strip()]
                next_dirs = self._process_images_by_keywords(level_images, keywords, operation_type, handle_others_flag=False)
                if next_dirs: current_dirs = next_dirs
                else: break

            if self.handle_others and not self.canceled:
                remaining_images = self._collect_level_images([self.source_dir])
                if remaining_images:
                    self.log_updated.emit(f"{len(remaining_images)}개의 미분류 파일을 'other'로 이동합니다...")
                    counters = {'other': 0}
                    for img_dir, img_file in remaining_images:
                        if self.canceled: break
                        path = os.path.join(img_dir, img_file)
                        self._process_image_file(img_dir, img_file, path, os.path.getsize(path), 'other', counters, operation_type)

        if self.canceled:
            self.log_updated.emit("작업 취소됨."); self.completed.emit(0); return

        if self.safe_mode_enabled:
            total_size_mb = sum(info['size'] for info in self.processed_files_info) / (1024 * 1024) if self.processed_files_info else 0.0
            self.safe_mode_dialog_required.emit(len(self.processed_files_info), total_size_mb)
        else:
            self.completed.emit(len(self.processed_files_info))

    def _collect_level_images(self, directories):
        images = []
        for d in directories:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and os.path.isfile(os.path.join(d, f)):
                    images.append((d, f))
        return images

    def _find_all_image_files_recursive(self, directory):
        images = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): images.append((root, f))
        return images

    def _process_images_by_keywords(self, images, keywords, operation_type, handle_others_flag):
        total, processed, next_dirs, unmatched = len(images), 0, [], []
        counters = {k: 0 for k in keywords}
        paths = [os.path.join(d, f) for d, f in images]

        if self.multicore_enabled and self.multicore_core_count > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.multicore_core_count) as executor:
                future_to_path = {executor.submit(process_single_image_task, path, keywords): path for path in paths}
                for future in concurrent.futures.as_completed(future_to_path):
                    if self.canceled:
                        if sys.version_info >= (3, 9): executor.shutdown(wait=False, cancel_futures=True)
                        else: executor.shutdown(wait=False)
                        return []
                    try:
                        result = future.result()
                        if result.get('log'): self.log_updated.emit(result['log'])
                        
                        img_path, file_size = result["path"], result.get("size", 0)
                        img_dir, img_file = os.path.dirname(img_path), os.path.basename(img_path)
                        
                        if result["status"] == "success":
                            target_dir = self._process_image_file(img_dir, img_file, img_path, file_size, result["keyword"], counters, operation_type)
                            if target_dir and target_dir not in next_dirs: next_dirs.append(target_dir)
                        elif result["status"] in ["no_keyword_match", "no_prompt"]:
                            unmatched.append((img_dir, img_file, img_path, file_size))
                    except Exception as e:
                        self.log_updated.emit(f"처리 중 예외 발생: {e}")
                    processed += 1
                    self.progress_updated.emit(int((processed / total) * 100)) if total > 0 else 0
        else:
            for path in paths:
                if self.canceled: return []
                result = process_single_image_task(path, keywords)
                if result.get('log'): self.log_updated.emit(result['log'])
                img_path, file_size = result["path"], result.get("size", 0)
                img_dir, img_file = os.path.dirname(img_path), os.path.basename(img_path)
                
                if result["status"] == "success":
                    target_dir = self._process_image_file(img_dir, img_file, img_path, file_size, result["keyword"], counters, operation_type)
                    if target_dir and target_dir not in next_dirs: next_dirs.append(target_dir)
                else:
                    unmatched.append((img_dir, img_file, img_path, file_size))
                processed += 1
                self.progress_updated.emit(int((processed / total) * 100)) if total > 0 else 0

        if handle_others_flag and unmatched:
            other_counters = {'other': 0}
            for img_dir, img_file, img_path, file_size in unmatched:
                if self.canceled: break
                self._process_image_file(img_dir, img_file, img_path, file_size, 'other', other_counters, operation_type)
        return next_dirs

    def _process_image_file(self, img_dir, img_file, img_path, file_size, keyword, counters, operation_type):
        sanitized = sanitize_for_path(keyword)
        target_dir = self.custom_dest_path if self.custom_dest_enabled and self.custom_dest_path else os.path.join(img_dir, sanitized)

        if not os.path.exists(target_dir):
            try: os.makedirs(target_dir, exist_ok=True); self.created_dirs.append(target_dir)
            except OSError as e: self.log_updated.emit(f"폴더 생성 실패: {e}"); return None

        counters.setdefault(keyword, 0)
        if self.rename_images:
            counters[keyword] += 1
            dest_filename = f"{sanitized}_{str(counters[keyword]).zfill(6)}{os.path.splitext(img_file)[1]}"
        else:
            dest_filename = img_file

        dest_path = os.path.join(target_dir, dest_filename)

        if os.path.exists(dest_path):
            if not self.resolve_conflicts: return None
            base, ext = os.path.splitext(dest_path)
            c = 1
            while os.path.exists(f"{base} ({str(c).zfill(2)}){ext}"): c += 1
            dest_path = f"{base} ({str(c).zfill(2)}){ext}"

        try:
            if operation_type == 'copy': shutil.copy2(img_path, dest_path)
            else: shutil.move(img_path, dest_path)

            self.processed_files_info.append({'src': img_path, 'dest': dest_path, 'size': file_size})
            if not self.safe_mode_enabled:
                 self.undo_info.append({'src': img_path, 'dest': dest_path, 'op': operation_type})

            action_text = "복사" if operation_type == 'copy' else "이동"
            self.log_updated.emit(f"✅ [{keyword}] {img_file} {action_text} 완료")

            return target_dir
        except Exception as e:
            self.log_updated.emit(f"이동 실패: {e}"); return None

    def finalize_safe_mode(self, choice):
        if choice == "delete":
            self.log_updated.emit("파일 무결성 검증 및 원본 삭제 진행...")
            for info in self.processed_files_info:
                try:
                    src, dest, size = info['src'], info['dest'], info['size']
                    if os.path.exists(dest) and os.path.getsize(dest) == size:
                        if os.path.exists(src): os.remove(src)
                        self.undo_info.append({'src': src, 'dest': dest, 'op': 'move'})
                    else:
                        self.log_updated.emit(f"경고: 복사본 무결성 검증 실패. 원본 보존됨: {src}")
                except Exception as e:
                    self.log_updated.emit(f"오류: {info['src']} 삭제 실패: {e}")
        elif choice == "keep":
             for info in self.processed_files_info: self.undo_info.append({'src': info['src'], 'dest': info['dest'], 'op': 'copy'})
        elif choice == "undo":
            self.log_updated.emit("복사본 삭제(실행 취소) 중...")
            for info in self.processed_files_info:
                try:
                    if os.path.exists(info['dest']): os.remove(info['dest'])
                except: pass
            self.undo_info = []

        self.completed.emit(len(self.processed_files_info) if choice != "undo" else 0)

    def undo_last_operation(self):
        if not self.undo_info: return
        success = 0
        for info in reversed(self.undo_info):
            try:
                dest, src, op = info['dest'], info['src'], info['op']
                if op == 'move' and os.path.exists(dest):
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dest, src)
                    success += 1
                elif op == 'copy' and os.path.exists(dest):
                    os.remove(dest)
                    success += 1
            except: pass

        for d in reversed(self.created_dirs):
            try:
                if os.path.exists(d) and not os.listdir(d): os.rmdir(d)
            except: pass
        self.log_updated.emit(f"작업 취소 완료: {success}개 파일 복원됨.")
        self.undo_info, self.created_dirs, self.processed_files_info = [], [], []

    def cancel(self): self.canceled = True


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.settings_manager = SettingsManager()
        self.init_ui()

    def update_preset_list(self):
        self.preset_combo.clear(); self.preset_combo.addItem("기본 설정")
        for p in self.settings_manager.get_preset_list(): self.preset_combo.addItem(p)

    def load_settings(self):
        s = self.settings_manager.get_settings_for_ui()
        self.source_dir = s[0]; self.dir_path_label.setText(s[0] or "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(s[1]); self.handle_others_check.setChecked(s[2])
        self.resolve_conflicts_check.setChecked(s[3]); self.safe_mode_check.setChecked(s[11])
        self.clone_mode_check.setChecked(s[12])
        self.multicore_check.setChecked(s[4]); self.core_count_spinbox.setValue(s[5])
        self._toggle_multicore_input(s[4])
        self.full_tracking_check.setChecked(s[7]); self.full_tracking_prompt_input.setText(s[8])
        self._toggle_full_tracking_input(s[7])
        self.custom_dest_check.setChecked(s[9]); self.custom_dest_path_input.setText(s[10])
        self._toggle_custom_dest_input(s[9])
        for i, (chk, inp) in enumerate(self.prompt_inputs):
            if i < len(s[6]): chk.setChecked(s[6][i][0]); inp.setText(s[6][i][1])
        self.update_preset_list()

    def save_current_settings(self):
        pl = [(c.isChecked(), i.text()) for c, i in self.prompt_inputs]
        s = self.settings_manager.create_settings_from_ui(
            self.source_dir, self.rename_check.isChecked(), self.handle_others_check.isChecked(),
            self.resolve_conflicts_check.isChecked(), self.multicore_check.isChecked(), self.core_count_spinbox.value(),
            pl, self.full_tracking_check.isChecked(), self.full_tracking_prompt_input.text(),
            self.custom_dest_check.isChecked(), self.custom_dest_path_input.text(),
            self.safe_mode_check.isChecked(), self.clone_mode_check.isChecked()
        )
        self.settings_manager.save_settings(s)

    def load_preset(self, index):
        if index <= 0: return
        p = self.settings_manager.load_preset(self.preset_combo.currentText())
        self.source_dir = p.get("source_directory", ""); self.dir_path_label.setText(self.source_dir or "없음")
        self.rename_check.setChecked(p.get("rename_images", False))
        self.handle_others_check.setChecked(p.get("handle_others", False))
        self.resolve_conflicts_check.setChecked(p.get("resolve_conflicts", False))
        
        self.safe_mode_check.setChecked(p.get("safe_mode_enabled", False))
        self.clone_mode_check.setChecked(p.get("clone_mode_enabled", False))
        self.safe_mode_check.setEnabled(not self.clone_mode_check.isChecked())
        self.clone_mode_check.setEnabled(not self.safe_mode_check.isChecked())

        self.multicore_check.setChecked(p.get("multicore_enabled", False))
        self.core_count_spinbox.setValue(p.get("multicore_core_count", os.cpu_count() or 4))
        self._toggle_multicore_input(p.get("multicore_enabled", False))
        self.full_tracking_check.setChecked(p.get("full_tracking_enabled", False))
        self.full_tracking_prompt_input.setText(p.get("full_tracking_prompt", ""))
        self._toggle_full_tracking_input(p.get("full_tracking_enabled", False))
        self.custom_dest_check.setChecked(p.get("custom_dest_enabled", False))
        self.custom_dest_path_input.setText(p.get("custom_dest_path", ""))
        self._toggle_custom_dest_input(p.get("custom_dest_enabled", False))

        pl = p.get("prompt_levels", [])
        for i, (c, inp) in enumerate(self.prompt_inputs):
            if i < len(pl): c.setChecked(pl[i].get("enabled", False)); inp.setText(pl[i].get("prompt", ""))
            else: c.setChecked(False); inp.setText("")

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
        if QMessageBox.question(self, '삭제', f"'{name}' 삭제?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            if self.settings_manager.delete_preset(name):
                self.preset_combo.removeItem(self.preset_combo.currentIndex())

    def init_ui(self):
        cw = QWidget(); self.setCentralWidget(cw); ml = QVBoxLayout()

        dl = QHBoxLayout(); dl.addWidget(QLabel("소스 디렉토리:"))
        self.dir_path_label = QLabel("디렉토리가 선택되지 않았습니다"); dl.addWidget(self.dir_path_label, 1)
        bb = QPushButton("찾아보기..."); bb.clicked.connect(self.browse_directory); dl.addWidget(bb); ml.addLayout(dl)

        pl = QHBoxLayout(); pl.addWidget(QLabel("프리셋:"))
        self.preset_combo = QComboBox(); self.preset_combo.currentIndexChanged.connect(self.load_preset); pl.addWidget(self.preset_combo, 1)
        self.save_preset_btn = QPushButton("저장"); self.save_preset_btn.clicked.connect(self.show_save_preset_dialog); pl.addWidget(self.save_preset_btn)
        self.delete_preset_btn = QPushButton("삭제"); self.delete_preset_btn.clicked.connect(self.delete_preset); pl.addWidget(self.delete_preset_btn); ml.addLayout(pl)

        ol = QHBoxLayout()
        self.rename_check = QCheckBox("이름 변경"); ol.addWidget(self.rename_check)
        self.handle_others_check = QCheckBox("그 외 처리"); ol.addWidget(self.handle_others_check)
        self.resolve_conflicts_check = QCheckBox("동일명 숫자 추가"); ol.addWidget(self.resolve_conflicts_check)
        self.safe_mode_check = QCheckBox("안전 모드"); self.safe_mode_check.toggled.connect(self._toggle_safety_modes); ol.addWidget(self.safe_mode_check)
        self.clone_mode_check = QCheckBox("복제 모드"); self.clone_mode_check.toggled.connect(self._toggle_safety_modes); ol.addWidget(self.clone_mode_check)
        ol.addStretch(); ml.addLayout(ol)

        mul = QHBoxLayout()
        self.multicore_check = QCheckBox("멀티코어"); self.multicore_check.toggled.connect(self._toggle_multicore_input); mul.addWidget(self.multicore_check)
        self.core_count_spinbox = QSpinBox(); self.core_count_spinbox.setRange(1, os.cpu_count() or 1); self.core_count_spinbox.setSuffix(" 코어"); mul.addWidget(self.core_count_spinbox)
        mul.addStretch(); ml.addLayout(mul)

        fl = QHBoxLayout()
        self.full_tracking_check = QCheckBox("전체추적"); self.full_tracking_check.toggled.connect(self._toggle_full_tracking_input); fl.addWidget(self.full_tracking_check)
        self.full_tracking_prompt_input = QLineEdit(); self.full_tracking_prompt_input.setPlaceholderText("전체추적 키워드 (| 구분)"); fl.addWidget(self.full_tracking_prompt_input, 1)
        ml.addLayout(fl)

        cdl = QHBoxLayout()
        self.custom_dest_check = QCheckBox("사용자 폴더 사용"); self.custom_dest_check.toggled.connect(self._toggle_custom_dest_input); cdl.addWidget(self.custom_dest_check)
        self.custom_dest_path_input = QLineEdit(); self.custom_dest_path_input.setPlaceholderText("이동할 폴더 경로"); cdl.addWidget(self.custom_dest_path_input, 1)
        cb = QPushButton("찾아보기..."); cb.clicked.connect(self._browse_custom_dest_directory); cdl.addWidget(cb); ml.addLayout(cdl)

        self.prompt_inputs = []
        for i in range(5):
            ll = QHBoxLayout(); lc = QCheckBox(f"레벨 {i+1}:"); lc.setChecked(i == 0); ll.addWidget(lc)
            pi = QLineEdit(); pi.setPlaceholderText(f"레벨 {i+1} 프롬프트 (| 구분)"); ll.addWidget(pi, 1)
            ml.addLayout(ll); self.prompt_inputs.append((lc, pi))

        self.progress_bar = QProgressBar(); ml.addWidget(self.progress_bar)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); ml.addWidget(self.log_text)

        bl = QHBoxLayout()
        self.start_btn = QPushButton("분류 시작"); self.start_btn.clicked.connect(self.start_classification); bl.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("취소"); self.cancel_btn.clicked.connect(self.cancel_classification); self.cancel_btn.setEnabled(False); bl.addWidget(self.cancel_btn)
        self.undo_btn = QPushButton("이전 작업 취소"); self.undo_btn.clicked.connect(self.undo_last_operation); bl.addWidget(self.undo_btn)
        ml.addLayout(bl); cw.setLayout(ml)

    def _toggle_safety_modes(self, checked):
        s = self.sender()
        if checked:
            if s == self.safe_mode_check: self.clone_mode_check.setEnabled(False)
            elif s == self.clone_mode_check: self.safe_mode_check.setEnabled(False)
        else:
            self.safe_mode_check.setEnabled(True); self.clone_mode_check.setEnabled(True)

    def undo_last_operation(self):
        if not self.worker or not self.worker.undo_info: QMessageBox.warning(self, "경고", "취소할 작업이 없습니다."); return
        if QMessageBox.question(self, '취소', "작업 취소?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes: self.worker.undo_last_operation()

    def _toggle_multicore_input(self, c): self.core_count_spinbox.setEnabled(c)
    def _toggle_full_tracking_input(self, c):
        self.full_tracking_prompt_input.setEnabled(c)
        for lc, pi in self.prompt_inputs: lc.setEnabled(not c); pi.setEnabled(not c)
    def _toggle_custom_dest_input(self, c): self.custom_dest_path_input.setEnabled(c)
    def _browse_custom_dest_directory(self):
        dp = QFileDialog.getExistingDirectory(self, "대상 선택")
        if dp: self.custom_dest_path_input.setText(dp)
    def browse_directory(self):
        dp = QFileDialog.getExistingDirectory(self, "소스 선택")
        if dp: self.source_dir = dp; self.dir_path_label.setText(dp)

    def start_classification(self):
        if not self.source_dir: QMessageBox.warning(self, "경고", "디렉토리 선택 요망."); return
        self.save_current_settings(); self.start_time = time.time()
        pl = [(c.isChecked(), i.text()) for c, i in self.prompt_inputs]
        if not self.full_tracking_check.isChecked() and not any(e for e, _ in pl) and not self.handle_others_check.isChecked(): return

        self.start_btn.setEnabled(False); self.cancel_btn.setEnabled(True); self.progress_bar.setValue(0); self.log_text.clear()
        self.worker = ImageClassifierWorker(
            self.source_dir, pl, self.rename_check.isChecked(), self.handle_others_check.isChecked(), self.resolve_conflicts_check.isChecked(),
            self.multicore_check.isChecked(), self.core_count_spinbox.value(), self.full_tracking_check.isChecked(), self.full_tracking_prompt_input.text(),
            self.custom_dest_check.isChecked(), self.custom_dest_path_input.text(), self.safe_mode_check.isChecked(), self.clone_mode_check.isChecked()
        )
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(lambda m: self.log_text.append(m))
        self.worker.completed.connect(self.classification_completed)
        self.worker.safe_mode_dialog_required.connect(self.show_safe_mode_popup)
        self.worker.start()

    def show_safe_mode_popup(self, count, total_size_mb):
        if count == 0: 
            self.worker.finalize_safe_mode("undo")
            return
        msg = QMessageBox(self); msg.setWindowTitle("안전 모드"); msg.setText(f"복사 완료: {count}개, {total_size_mb:.2f}MB\n원본 처리?")
        d_btn = msg.addButton("원본 삭제", QMessageBox.YesRole); k_btn = msg.addButton("보존", QMessageBox.NoRole); u_btn = msg.addButton("취소(복사본 삭제)", QMessageBox.RejectRole)
        msg.exec_()
        cb = msg.clickedButton()
        if cb == d_btn: self.worker.finalize_safe_mode("delete")
        elif cb == k_btn: self.worker.finalize_safe_mode("keep")
        else: self.worker.finalize_safe_mode("undo")

    def cancel_classification(self):
        if self.worker and self.worker.isRunning(): self.worker.cancel(); self.log_text.append("취소 중..."); self.cancel_btn.setEnabled(False)

    def classification_completed(self, count):
        self.log_text.append(f"완료! ({time.time()-self.start_time:.2f}초, {count}개)")
        self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False); self.undo_btn.setEnabled(True); self.progress_bar.setValue(100)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, '종료', "진행 중. 종료?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes: self.worker.cancel(); self.worker.wait(); e.accept()
            else: e.ignore()
        else: self.save_current_settings(); e.accept()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        import multiprocessing; multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = ImageClassifierApp(); window.load_settings(); window.show()
    sys.exit(app.exec_())