import sys
import os
import shutil
import gzip
import time
import concurrent.futures
from PyQt5.QtWidgets import (QComboBox, QInputDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, QProgressBar,
                            QMessageBox, QTextEdit, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
from settings_manager import SettingsManager
from image_utils import read_info_from_image

def sanitize_for_path(name: str) -> str:
    """
    Windows 파일/폴더 이름에 사용할 수 없는 문자를 대체합니다.
    """
    illegal_chars = r'<>:"/\\|?*'
    sanitized = name
    for char in illegal_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized

# 멀티프로세싱을 위한 최상위 레벨 함수
def process_single_image_task(image_path, keywords):
    """
    단일 이미지 파일을 처리하는 작업 함수 (CPU 바운드 작업).
    멀티프로세싱 워커에 의해 실행됩니다.
    """
    img_file = os.path.basename(image_path)
    try:
        # 파일 크기 가져오기
        file_size = os.path.getsize(image_path)
        prompt_data = read_info_from_image(image_path)
        if not prompt_data:
            return {"status": "no_prompt", "path": image_path, "log": f"{img_file}: 프롬프트 데이터 없음"}

        if not keywords:
            return {"status": "no_keyword_match", "path": image_path, "prompt": prompt_data, "size": file_size}

        for keyword in keywords:
            if keyword.lower() in prompt_data.lower():
                return {"status": "success", "path": image_path, "keyword": keyword, "prompt": prompt_data, "size": file_size}

        return {"status": "no_keyword_match", "path": image_path, "prompt": prompt_data, "size": file_size, "log": f"{img_file}: 일치하는 키워드 없음"}
    except FileNotFoundError:
        return {"status": "error", "path": image_path, "log": f"{img_file} 파일을 찾을 수 없습니다."}
    except Exception as e:
        return {"status": "error", "path": image_path, "log": f"{img_file} 처리 중 오류 발생: {str(e)}"}


class ImageClassifierWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    completed = pyqtSignal(int)
    safe_mode_dialog_required = pyqtSignal(int, float)

    def __init__(self, source_dir, prompt_levels, rename_images=False, handle_others=False, resolve_conflicts=False,
                 multicore_enabled=False, multicore_core_count=4,
                 full_tracking_enabled=False, full_tracking_prompt="", custom_dest_enabled=False, custom_dest_path="",
                 safe_mode_enabled=False, clone_mode_enabled=False):
        super().__init__()
        self.source_dir = source_dir
        self.prompt_levels = prompt_levels
        self.rename_images = rename_images
        self.handle_others = handle_others
        self.resolve_conflicts = resolve_conflicts
        self.multicore_enabled = multicore_enabled
        self.multicore_core_count = multicore_core_count
        self.full_tracking_enabled = full_tracking_enabled
        self.full_tracking_prompt = full_tracking_prompt
        self.custom_dest_enabled = custom_dest_enabled
        self.custom_dest_path = custom_dest_path
        self.safe_mode_enabled = safe_mode_enabled
        self.clone_mode_enabled = clone_mode_enabled
        self.canceled = False

        self.undo_info = []
        self.created_dirs = []
        self.processed_files_info = []

    def run(self):
        self.undo_info = []
        self.created_dirs = []
        self.processed_files_info = []

        operation_type = 'copy' if self.safe_mode_enabled or self.clone_mode_enabled else 'move'

        if self.full_tracking_enabled:
            self.log_updated.emit("전체추적 모드 활성화: 모든 하위 폴더의 이미지를 검색합니다.")
            image_files_with_paths = self._find_all_image_files_recursive(self.source_dir)
            if not image_files_with_paths:
                self.log_updated.emit("이미지 파일을 찾을 수 없습니다.")
                self.completed.emit(0)
                return

            self.log_updated.emit(f"{len(image_files_with_paths)}개의 이미지를 찾았습니다. 전체추적 분류를 시작합니다...")

            prompt_keywords = [p.strip() for p in self.full_tracking_prompt.split('|') if p.strip()]
            if not prompt_keywords and not self.handle_others:
                self.log_updated.emit("전체추적 프롬프트가 비어있거나 '그 외 처리'가 비활성화되어 작업을 중단합니다.")
                self.completed.emit(0)
                return

            self._process_images_by_keywords(image_files_with_paths, prompt_keywords, operation_type)

        else:
            current_dirs = [self.source_dir]
            for level_idx, (enabled, prompt_string) in enumerate(self.prompt_levels):
                if not enabled or not prompt_string.strip():
                    if level_idx == 0 and self.handle_others:
                        pass
                    else:
                        continue

                self.log_updated.emit(f"레벨 {level_idx+1} 처리 중 - 프롬프트: {prompt_string}")
                level_images = self._collect_level_images(current_dirs)

                if not level_images:
                    self.log_updated.emit("처리할 이미지가 없습니다.")
                    break

                prompt_keywords = [p.strip() for p in prompt_string.split('|') if p.strip()]
                next_dirs = self._process_images_by_keywords(level_images, prompt_keywords, operation_type)

                if next_dirs:
                    current_dirs = next_dirs
                else:
                    self.log_updated.emit("더 이상 처리할 디렉토리가 없습니다.")
                    break

        if self.canceled:
            self.log_updated.emit("작업이 취소되었습니다.")
            self.completed.emit(0)
            return

        if self.safe_mode_enabled:
            total_size_mb = sum(info['size'] for info in self.processed_files_info) / (1024 * 1024) if self.processed_files_info else 0.0
            self.safe_mode_dialog_required.emit(len(self.processed_files_info), total_size_mb)
        else:
            self.completed.emit(len(self.processed_files_info))

    def _collect_level_images(self, directories):
        level_images = []
        for directory in directories:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and os.path.isfile(file_path):
                    level_images.append((directory, file))
        return level_images

    def _find_all_image_files_recursive(self, directory):
        image_files_with_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files_with_paths.append((root, file))
        return image_files_with_paths

    def _process_images_by_keywords(self, images, keywords, operation_type):
        total_images = len(images)
        processed_count = 0
        next_dirs = []
        unmatched_images = []
        keyword_counters = {keyword: 0 for keyword in keywords}

        image_paths = [os.path.join(img_dir, img_file) for img_dir, img_file in images]

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.multicore_core_count if self.multicore_enabled else 1) as executor:
            future_to_path = {executor.submit(process_single_image_task, path, keywords): path for path in image_paths}

            for future in concurrent.futures.as_completed(future_to_path):
                if self.canceled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return []

                try:
                    result = future.result()
                    if result.get('log'):
                        self.log_updated.emit(result['log'])

                    img_path = result["path"]
                    img_dir = os.path.dirname(img_path)
                    img_file = os.path.basename(img_path)

                    if result["status"] == "success":
                        matched_keyword = result["keyword"]
                        file_size = result.get("size", 0)
                        keyword_dir = self._process_image_file(img_dir, img_file, img_path, file_size, matched_keyword, keyword_counters, operation_type)
                        if keyword_dir and keyword_dir not in next_dirs:
                            next_dirs.append(keyword_dir)
                    elif result["status"] in ["no_keyword_match", "no_prompt"]:
                        unmatched_images.append((img_dir, img_file, img_path, result.get("size", 0)))

                except Exception as e:
                    path = future_to_path[future]
                    img_file = os.path.basename(path)
                    self.log_updated.emit(f"{img_file} 처리 중 심각한 오류 발생: {e}")

                processed_count += 1
                progress = int((processed_count / total_images) * 100) if total_images > 0 else 0
                self.progress_updated.emit(progress)

        if self.handle_others and unmatched_images:
            self.log_updated.emit(f"{len(unmatched_images)}개의 분류되지 않은 파일을 'other' 폴더로 이동합니다...")
            other_counters = {'other': 0}
            for img_dir, img_file, img_path, file_size in unmatched_images:
                if self.canceled: break
                self._process_image_file(img_dir, img_file, img_path, file_size, 'other', other_counters, operation_type)

        return next_dirs

    def _process_image_file(self, img_dir, img_file, img_path, file_size, keyword, counters, operation_type):
        sanitized_keyword = sanitize_for_path(keyword)

        if self.custom_dest_enabled and self.custom_dest_path:
            target_dir = self.custom_dest_path
        else:
            target_dir = os.path.join(img_dir, sanitized_keyword)

        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
            except OSError as e:
                self.log_updated.emit(f"오류: 대상 폴더를 생성할 수 없습니다: {target_dir}. 건너뜁니다. ({e})")
                return None

        if self.rename_images:
            counters[keyword] += 1
            dest_filename = f"{sanitized_keyword}_{str(counters[keyword]).zfill(6)}{os.path.splitext(img_file)[1]}"
        else:
            dest_filename = img_file

        dest_path = os.path.join(target_dir, dest_filename)

        if os.path.exists(dest_path) and not self.resolve_conflicts:
            self.log_updated.emit(f"경고: '{os.path.basename(dest_path)}' 파일이 이미 존재하여 건너뜁니다.")
            return None
        elif os.path.exists(dest_path) and self.resolve_conflicts:
            base, ext = os.path.splitext(dest_path)
            counter = 1
            new_dest_path = f"{base} ({str(counter).zfill(2)}){ext}"
            while os.path.exists(new_dest_path):
                counter += 1
                new_dest_path = f"{base} ({str(counter).zfill(2)}){ext}"
            dest_path = new_dest_path
            self.log_updated.emit(f"알림: 이름 충돌로 '{os.path.basename(dest_path)}'(으)로 저장")

        try:
            if operation_type == 'copy':
                shutil.copy2(img_path, dest_path)
            else: # 'move'
                shutil.move(img_path, dest_path)

            self.processed_files_info.append({'src': img_path, 'dest': dest_path, 'size': file_size})

            if not self.safe_mode_enabled:
                 self.undo_info.append({'src': img_path, 'dest': dest_path, 'op': operation_type})

            self.log_updated.emit(f"{img_file} -> {os.path.relpath(dest_path, self.source_dir)}")
            return target_dir
        except Exception as e:
            self.log_updated.emit(f"오류: {img_file}을(를) {dest_path}(으)로 처리하는 중 오류 발생: {e}")
            return None

    def finalize_safe_mode(self, choice):
        if choice == "delete": # 원본 삭제
            self.log_updated.emit("원본 파일을 삭제합니다...")
            for info in self.processed_files_info:
                try:
                    if os.path.exists(info['src']):
                        os.remove(info['src'])
                    self.undo_info.append({'src': info['src'], 'dest': info['dest'], 'op': 'move'})
                except Exception as e:
                    self.log_updated.emit(f"오류: 원본 파일 {info['src']} 삭제 실패: {e}")
            self.log_updated.emit("원본 파일 삭제 완료.")
        elif choice == "keep": # 모두 보존
             self.log_updated.emit("원본과 복사본을 모두 보존합니다.")
             for info in self.processed_files_info:
                 self.undo_info.append({'src': info['src'], 'dest': info['dest'], 'op': 'copy'})
        elif choice == "undo": # 실행 취소 (복사본 삭제)
            self.log_updated.emit("복사된 파일을 삭제하여 실행을 취소합니다...")
            for info in self.processed_files_info:
                try:
                    if os.path.exists(info['dest']):
                        os.remove(info['dest'])
                except Exception as e:
                    self.log_updated.emit(f"오류: 복사본 {info['dest']} 삭제 실패: {e}")
            self.undo_info = [] # Undo is done, clear list.
            self.log_updated.emit("복사본 삭제 완료.")

        self.completed.emit(len(self.processed_files_info) if choice != "undo" else 0)

    def undo_last_operation(self):
        if not self.undo_info:
            self.log_updated.emit("취소할 작업이 없습니다.")
            return

        success_count = 0
        self.log_updated.emit("이전 작업을 취소하는 중...")

        for info in reversed(self.undo_info):
            try:
                dest_path = info['dest']
                src_path = info['src']
                op_type = info['op']

                if op_type == 'move':
                    if os.path.exists(dest_path):
                        os.makedirs(os.path.dirname(src_path), exist_ok=True)
                        shutil.move(dest_path, src_path)
                        success_count += 1
                elif op_type == 'copy':
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                        success_count += 1
            except Exception as e:
                self.log_updated.emit(f"파일 복원/삭제 중 오류 발생: {str(e)}")

        # 생성된 빈 디렉토리 정리
        for dir_path in reversed(self.created_dirs):
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                self.log_updated.emit(f"디렉토리 제거 중 오류 발생: {str(e)}")

        self.log_updated.emit(f"{success_count}개 파일에 대한 작업을 취소했습니다.")
        self.undo_info = []
        self.created_dirs = []
        self.processed_files_info = []

    def cancel(self):
        self.canceled = True


class ImageClassifierApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.source_dir = ""
        self.worker = None
        self.start_time = 0
        self.settings_manager = SettingsManager()
        self.init_ui()

    def update_preset_list(self):
        self.preset_combo.clear()
        self.preset_combo.addItem("기본 설정")
        presets = self.settings_manager.get_preset_list()
        if presets:
            for preset in presets:
                self.preset_combo.addItem(preset)

    def load_settings(self):
        (source_dir, rename_images, handle_others, resolve_conflicts,
         multicore_enabled, multicore_core_count, prompt_levels,
         full_tracking_enabled, full_tracking_prompt, custom_dest_enabled, custom_dest_path,
         safe_mode_enabled, clone_mode_enabled) = self.settings_manager.get_settings_for_ui()

        self.source_dir = source_dir
        self.dir_path_label.setText(source_dir if source_dir else "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(rename_images)
        self.handle_others_check.setChecked(handle_others)
        self.resolve_conflicts_check.setChecked(resolve_conflicts)
        self.safe_mode_check.setChecked(safe_mode_enabled)
        self.clone_mode_check.setChecked(clone_mode_enabled)

        self.multicore_check.setChecked(multicore_enabled)
        self.core_count_spinbox.setValue(multicore_core_count)
        self._toggle_multicore_input(multicore_enabled)

        self.full_tracking_check.setChecked(full_tracking_enabled)
        self.full_tracking_prompt_input.setText(full_tracking_prompt)
        self._toggle_full_tracking_input(full_tracking_enabled)

        self.custom_dest_check.setChecked(custom_dest_enabled)
        self.custom_dest_path_input.setText(custom_dest_path)
        self._toggle_custom_dest_input(custom_dest_enabled)

        for i, (level_check, prompt_input) in enumerate(self.prompt_inputs):
            if i < len(prompt_levels):
                level_check.setChecked(prompt_levels[i][0])
                prompt_input.setText(prompt_levels[i][1])

        self.update_preset_list()

    def save_current_settings(self):
        prompt_levels = []
        for level_check, prompt_input in self.prompt_inputs:
            prompt_levels.append((level_check.isChecked(), prompt_input.text()))

        settings = self.settings_manager.create_settings_from_ui(
            self.source_dir,
            self.rename_check.isChecked(),
            self.handle_others_check.isChecked(),
            self.resolve_conflicts_check.isChecked(),
            self.multicore_check.isChecked(),
            self.core_count_spinbox.value(),
            prompt_levels,
            self.full_tracking_check.isChecked(),
            self.full_tracking_prompt_input.text(),
            self.custom_dest_check.isChecked(),
            self.custom_dest_path_input.text(),
            self.safe_mode_check.isChecked(),
            self.clone_mode_check.isChecked()
        )
        self.settings_manager.save_settings(settings)

    def show_save_preset_dialog(self):
        name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름:")
        if ok and name:
            prompt_levels = []
            for level_check, prompt_input in self.prompt_inputs:
                prompt_levels.append((level_check.isChecked(), prompt_input.text()))

            settings = self.settings_manager.create_settings_from_ui(
                self.source_dir,
                self.rename_check.isChecked(),
                self.handle_others_check.isChecked(),
                self.resolve_conflicts_check.isChecked(),
                self.multicore_check.isChecked(),
                self.core_count_spinbox.value(),
                prompt_levels,
                self.full_tracking_check.isChecked(),
                self.full_tracking_prompt_input.text(),
                self.custom_dest_check.isChecked(),
                self.custom_dest_path_input.text(),
                self.safe_mode_check.isChecked(),
                self.clone_mode_check.isChecked()
            )

            if self.settings_manager.save_preset(name, settings):
                QMessageBox.information(self, "성공", f"프리셋 '{name}'이(가) 저장되었습니다.")
                self.update_preset_list()
                index = self.preset_combo.findText(name)
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
            else:
                QMessageBox.warning(self, "오류", "프리셋 저장 중 오류가 발생했습니다.")

    def load_preset(self, index):
        if index <= 0: return
        preset_name = self.preset_combo.currentText()
        preset = self.settings_manager.load_preset(preset_name)

        self.source_dir = preset.get("source_directory", "")
        self.dir_path_label.setText(self.source_dir if self.source_dir else "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(preset.get("rename_images", False))
        self.handle_others_check.setChecked(preset.get("handle_others", False))
        self.resolve_conflicts_check.setChecked(preset.get("resolve_conflicts", False))
        self.safe_mode_check.setChecked(preset.get("safe_mode_enabled", False))
        self.clone_mode_check.setChecked(preset.get("clone_mode_enabled", False))

        multicore_enabled = preset.get("multicore_enabled", False)
        multicore_core_count = preset.get("multicore_core_count", os.cpu_count() or 4)
        self.multicore_check.setChecked(multicore_enabled)
        self.core_count_spinbox.setValue(multicore_core_count)
        self._toggle_multicore_input(multicore_enabled)

        full_tracking_enabled = preset.get("full_tracking_enabled", False)
        full_tracking_prompt = preset.get("full_tracking_prompt", "")
        self.full_tracking_check.setChecked(full_tracking_enabled)
        self.full_tracking_prompt_input.setText(full_tracking_prompt)
        self._toggle_full_tracking_input(full_tracking_enabled)

        custom_dest_enabled = preset.get("custom_dest_enabled", False)
        custom_dest_path = preset.get("custom_dest_path", "")
        self.custom_dest_check.setChecked(custom_dest_enabled)
        self.custom_dest_path_input.setText(custom_dest_path)
        self._toggle_custom_dest_input(custom_dest_enabled)

        prompt_levels = preset.get("prompt_levels", [])
        for i, (level_check, prompt_input) in enumerate(self.prompt_inputs):
            if i < len(prompt_levels):
                level_check.setChecked(prompt_levels[i].get("enabled", False))
                prompt_input.setText(prompt_levels[i].get("prompt", ""))
            else:
                level_check.setChecked(False)
                prompt_input.setText("")

        self.log_text.append(f"프리셋 '{preset_name}'을(를) 로드했습니다.")

    def delete_preset(self):
        if self.preset_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "경고", "기본 설정은 삭제할 수 없습니다.")
            return

        preset_name = self.preset_combo.currentText()
        reply = QMessageBox.question(self, '프리셋 삭제', f"프리셋 '{preset_name}'을(를) 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.settings_manager.delete_preset(preset_name):
                self.preset_combo.removeItem(self.preset_combo.currentIndex())
                QMessageBox.information(self, "성공", f"프리셋 '{preset_name}'이(가) 삭제되었습니다.")
            else:
                QMessageBox.warning(self, "오류", "프리셋 삭제 중 오류가 발생했습니다.")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("소스 디렉토리:"))
        self.dir_path_label = QLabel("디렉토리가 선택되지 않았습니다")
        dir_layout.addWidget(self.dir_path_label, 1)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        main_layout.addLayout(dir_layout)

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("프리셋:"))
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo, 1)
        self.save_preset_btn = QPushButton("저장")
        self.save_preset_btn.clicked.connect(self.show_save_preset_dialog)
        preset_layout.addWidget(self.save_preset_btn)
        self.delete_preset_btn = QPushButton("삭제")
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        preset_layout.addWidget(self.delete_preset_btn)
        main_layout.addLayout(preset_layout)

        options_layout = QHBoxLayout()
        self.rename_check = QCheckBox("프롬프트에 맞게 이미지 이름 변경")
        options_layout.addWidget(self.rename_check)
        self.handle_others_check = QCheckBox("그 외 처리")
        options_layout.addWidget(self.handle_others_check)
        self.resolve_conflicts_check = QCheckBox("동일명 파일 숫자 추가")
        options_layout.addWidget(self.resolve_conflicts_check)

        self.safe_mode_check = QCheckBox("안전 모드")
        self.safe_mode_check.toggled.connect(self._toggle_safety_modes)
        options_layout.addWidget(self.safe_mode_check)

        self.clone_mode_check = QCheckBox("복제 모드")
        self.clone_mode_check.toggled.connect(self._toggle_safety_modes)
        options_layout.addWidget(self.clone_mode_check)

        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        multicore_layout = QHBoxLayout()
        self.multicore_check = QCheckBox("멀티코어 처리 사용")
        self.multicore_check.toggled.connect(self._toggle_multicore_input)
        multicore_layout.addWidget(self.multicore_check)
        self.core_count_spinbox = QSpinBox()
        self.core_count_spinbox.setRange(1, os.cpu_count() or 1)
        self.core_count_spinbox.setSuffix(" 개 코어")
        multicore_layout.addWidget(self.core_count_spinbox)
        multicore_layout.addStretch()
        main_layout.addLayout(multicore_layout)

        full_tracking_layout = QHBoxLayout()
        self.full_tracking_check = QCheckBox("전체추적 활성화")
        self.full_tracking_check.toggled.connect(self._toggle_full_tracking_input)
        full_tracking_layout.addWidget(self.full_tracking_check)
        self.full_tracking_prompt_input = QLineEdit()
        self.full_tracking_prompt_input.setPlaceholderText("전체추적 프롬프트를 | 문자로 구분하여 입력")
        full_tracking_layout.addWidget(self.full_tracking_prompt_input, 1)
        main_layout.addLayout(full_tracking_layout)

        custom_dest_layout = QHBoxLayout()
        self.custom_dest_check = QCheckBox("사용자 지정 대상 폴더 사용")
        self.custom_dest_check.toggled.connect(self._toggle_custom_dest_input)
        custom_dest_layout.addWidget(self.custom_dest_check)
        self.custom_dest_path_input = QLineEdit()
        self.custom_dest_path_input.setPlaceholderText("이미지를 이동할 사용자 지정 폴더 경로")
        custom_dest_layout.addWidget(self.custom_dest_path_input, 1)
        self.browse_custom_dest_btn = QPushButton("찾아보기...")
        self.browse_custom_dest_btn.clicked.connect(self._browse_custom_dest_directory)
        custom_dest_layout.addWidget(self.browse_custom_dest_btn)
        main_layout.addLayout(custom_dest_layout)

        self.prompt_inputs = []
        for i in range(5):
            level_layout = QHBoxLayout()
            level_check = QCheckBox(f"레벨 {i+1}:")
            level_check.setChecked(i == 0)
            level_layout.addWidget(level_check)
            prompt_input = QLineEdit()
            prompt_input.setPlaceholderText(f"레벨 {i+1} 프롬프트를 | 문자로 구분하여 입력")
            level_layout.addWidget(prompt_input, 1)
            main_layout.addLayout(level_layout)
            self.prompt_inputs.append((level_check, prompt_input))

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(QLabel("로그:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("분류 시작")
        self.start_btn.clicked.connect(self.start_classification)
        buttons_layout.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.cancel_classification)
        self.cancel_btn.setEnabled(False)
        buttons_layout.addWidget(self.cancel_btn)
        self.undo_btn = QPushButton("이전 작업 취소")
        self.undo_btn.clicked.connect(self.undo_last_operation)
        buttons_layout.addWidget(self.undo_btn)
        main_layout.addLayout(buttons_layout)

        central_widget.setLayout(main_layout)

    def _toggle_safety_modes(self, checked):
        source = self.sender()
        if checked:
            if source == self.safe_mode_check:
                self.clone_mode_check.setEnabled(False)
            elif source == self.clone_mode_check:
                self.safe_mode_check.setEnabled(False)
        else:
            self.safe_mode_check.setEnabled(True)
            self.clone_mode_check.setEnabled(True)

    def undo_last_operation(self):
        if not self.worker or not hasattr(self.worker, 'undo_info') or not self.worker.undo_info:
            QMessageBox.warning(self, "경고", "취소할 수 있는 작업이 없습니다.")
            return

        reply = QMessageBox.question(self, '작업 취소',
                                     "최근 분류 작업을 취소하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.worker.undo_last_operation()
            self.progress_bar.setValue(0)

    def _toggle_multicore_input(self, checked):
        self.core_count_spinbox.setEnabled(checked)

    def _toggle_full_tracking_input(self, checked):
        self.full_tracking_prompt_input.setEnabled(checked)
        for level_check, prompt_input in self.prompt_inputs:
            level_check.setEnabled(not checked)
            prompt_input.setEnabled(not checked)

    def _toggle_custom_dest_input(self, checked):
        self.custom_dest_path_input.setEnabled(checked)
        self.browse_custom_dest_btn.setEnabled(checked)

    def _browse_custom_dest_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "대상 디렉토리 선택")
        if dir_path:
            self.custom_dest_path_input.setText(dir_path)
            self.save_current_settings()

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "소스 디렉토리 선택")
        if dir_path:
            self.source_dir = dir_path
            self.dir_path_label.setText(dir_path)
            self.save_current_settings()

    def start_classification(self):
        if not self.source_dir:
            QMessageBox.warning(self, "경고", "소스 디렉토리가 선택되지 않았습니다.")
            return

        self.save_current_settings()
        self.start_time = time.time()

        prompt_levels = [(chk.isChecked(), inp.text()) for chk, inp in self.prompt_inputs]
        is_any_level_active = any(enabled for enabled, _ in prompt_levels)
        is_full_tracking_active = self.full_tracking_check.isChecked() and self.full_tracking_prompt_input.text().strip()

        if not is_full_tracking_active and not is_any_level_active and not self.handle_others_check.isChecked():
            QMessageBox.warning(self, "경고", "실행할 작업이 없습니다...")
            return

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.undo_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = ImageClassifierWorker(
            self.source_dir,
            prompt_levels,
            rename_images=self.rename_check.isChecked(),
            handle_others=self.handle_others_check.isChecked(),
            resolve_conflicts=self.resolve_conflicts_check.isChecked(),
            multicore_enabled=self.multicore_check.isChecked(),
            multicore_core_count=self.core_count_spinbox.value(),
            full_tracking_enabled=self.full_tracking_check.isChecked(),
            full_tracking_prompt=self.full_tracking_prompt_input.text(),
            custom_dest_enabled=self.custom_dest_check.isChecked(),
            custom_dest_path=self.custom_dest_path_input.text(),
            safe_mode_enabled=self.safe_mode_check.isChecked(),
            clone_mode_enabled=self.clone_mode_check.isChecked()
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_updated.connect(self.update_log)
        self.worker.completed.connect(self.classification_completed)
        self.worker.safe_mode_dialog_required.connect(self.show_safe_mode_popup)
        self.worker.start()

    def show_safe_mode_popup(self, count, total_size_mb):
        if count == 0:
            self.classification_completed(0)
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("안전 모드 확인")
        msg_box.setText(f"파일 복사가 완료되었습니다.\n\n- 파일 수: {count}개\n- 총 용량: {total_size_mb:.2f} MB\n\n원본 파일을 어떻게 처리하시겠습니까?")

        delete_btn = msg_box.addButton("원본 삭제", QMessageBox.YesRole)
        keep_btn = msg_box.addButton("모두 보존", QMessageBox.NoRole)
        undo_btn = msg_box.addButton("실행 취소 (복사본 삭제)", QMessageBox.RejectRole)

        msg_box.exec_()

        clicked_button = msg_box.clickedButton()
        if clicked_button == delete_btn:
            self.worker.finalize_safe_mode("delete")
        elif clicked_button == keep_btn:
            self.worker.finalize_safe_mode("keep")
        else: # undo_btn or closed
            self.worker.finalize_safe_mode("undo")

    def cancel_classification(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.log_text.append("작업 취소 중...")
            self.cancel_btn.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def classification_completed(self, classified_count):
        duration = time.time() - self.start_time
        self.log_text.append(f"분류가 완료되었습니다! (총 소요 시간: {duration:.2f}초, 처리된 이미지: {classified_count}개)")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.undo_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        if not (self.worker and self.worker.safe_mode_enabled) or classified_count > 0:
             QMessageBox.information(self, "완료", "이미지 분류가 완료되었습니다.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '작업 중단', "작업이 진행 중입니다. 종료하시겠습니까?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            self.save_current_settings()
            event.accept()

if __name__ == "__main__":
    # PyInstaller/Windows에서 멀티프로세싱을 위한 필수 코드
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.load_settings()
    window.show()
    sys.exit(app.exec_())