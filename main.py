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

# 멀티프로세싱을 위한 최상위 레벨 함수
def process_single_image_task(image_path, keywords):
    """
    단일 이미지 파일을 처리하는 작업 함수 (CPU 바운드 작업).
    멀티프로세싱 워커에 의해 실행됩니다.
    """
    img_file = os.path.basename(image_path)
    try:
        prompt_data = read_info_from_image(image_path)
        if not prompt_data:
            return {"status": "no_prompt", "path": image_path, "log": f"{img_file}: 프롬프트 데이터 없음"}

        if not keywords:
            return {"status": "no_keyword_match", "path": image_path, "prompt": prompt_data}

        for keyword in keywords:
            if keyword.lower() in prompt_data.lower():
                return {"status": "success", "path": image_path, "keyword": keyword, "prompt": prompt_data}
        
        return {"status": "no_keyword_match", "path": image_path, "prompt": prompt_data, "log": f"{img_file}: 일치하는 키워드 없음"}
    except Exception as e:
        return {"status": "error", "path": image_path, "log": f"{img_file} 처리 중 오류 발생: {str(e)}"}


class ImageClassifierWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    completed = pyqtSignal()
    
    def undo_classification(self):
        """분류 작업 취소"""
        if not self.moved_files:
            self.log_updated.emit("취소할 작업이 없습니다.")
            return
        
        success_count = 0
        for src_path, dest_path in reversed(self.moved_files):
            try:
                if os.path.exists(dest_path):
                    os.makedirs(os.path.dirname(src_path), exist_ok=True)
                    shutil.move(dest_path, src_path)
                    success_count += 1
            except Exception as e:
                self.log_updated.emit(f"파일 복원 중 오류 발생: {str(e)}")
        
        for dir_path in reversed(self.created_dirs):
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                self.log_updated.emit(f"디렉토리 제거 중 오류 발생: {str(e)}")
        
        self.log_updated.emit(f"{success_count}개 파일을 원래 위치로 복원했습니다.")
        self.moved_files = []
        self.created_dirs = []

    def __init__(self, source_dir, prompt_levels, rename_images=False, handle_others=False, resolve_conflicts=False, 
                 multicore_enabled=False, multicore_core_count=4,
                 full_tracking_enabled=False, full_tracking_prompt="", custom_dest_enabled=False, custom_dest_path=""):
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
        self.canceled = False
        
    def run(self):
        start_time = time.time()
        self.moved_files = []
        self.created_dirs = []

        if self.full_tracking_enabled:
            self.log_updated.emit("전체추적 모드 활성화: 모든 하위 폴더의 이미지를 검색합니다.")
            image_files_with_paths = self._find_all_image_files_recursive(self.source_dir)
            if not image_files_with_paths:
                self.log_updated.emit("이미지 파일을 찾을 수 없습니다.")
                self.completed.emit()
                return
            
            self.log_updated.emit(f"{len(image_files_with_paths)}개의 이미지를 찾았습니다. 전체추적 분류를 시작합니다...")
            
            prompt_keywords = [p.strip() for p in self.full_tracking_prompt.split('|') if p.strip()]
            if not prompt_keywords and not self.handle_others:
                self.log_updated.emit("전체추적 프롬프트가 비어있거나 '그 외 처리'가 비활성화되어 작업을 중단합니다.")
                self.completed.emit()
                return
                
            self._process_images_by_keywords(image_files_with_paths, prompt_keywords)

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
                next_dirs = self._process_images_by_keywords(level_images, prompt_keywords)
                
                if next_dirs:
                    current_dirs = next_dirs
                else:
                    self.log_updated.emit("더 이상 처리할 디렉토리가 없습니다.")
                    break
        
        end_time = time.time()
        duration = end_time - start_time
        self.log_updated.emit(f"분류가 완료되었습니다! (총 소요 시간: {duration:.2f}초)")
        self.completed.emit()
    
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
    
    def _process_images_by_keywords(self, images, keywords):
        total_images = len(images)
        processed_count = 0
        next_dirs = []
        unmatched_images = []
        keyword_counters = {keyword: 0 for keyword in keywords}
        
        image_paths = [os.path.join(img_dir, img_file) for img_dir, img_file in images]

        if self.multicore_enabled and total_images > 0:
            self.log_updated.emit(f"멀티코어 처리 활성화 ({self.multicore_core_count}개 코어 사용)...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.multicore_core_count) as executor:
                future_to_path = {executor.submit(process_single_image_task, path, keywords): path for path in image_paths}
                for future in concurrent.futures.as_completed(future_to_path):
                    if self.canceled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.log_updated.emit("작업이 취소되었습니다.")
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
                            keyword_dir = self._move_image(img_dir, img_file, img_path, matched_keyword, keyword_counters)
                            if keyword_dir and keyword_dir not in next_dirs:
                                next_dirs.append(keyword_dir)
                        elif result["status"] in ["no_keyword_match", "no_prompt"]:
                            unmatched_images.append((img_dir, img_file, img_path))

                    except Exception as e:
                        path = future_to_path[future]
                        img_file = os.path.basename(path)
                        self.log_updated.emit(f"{img_file} 처리 중 심각한 오류 발생: {e}")
                    
                    processed_count += 1
                    progress = int((processed_count / total_images) * 100)
                    self.progress_updated.emit(progress)
        else:
            if total_images > 0:
                self.log_updated.emit("단일 코어 처리 중...")
            for path in image_paths:
                if self.canceled:
                    self.log_updated.emit("작업이 취소되었습니다.")
                    return []
                
                result = process_single_image_task(path, keywords)
                if result.get('log'):
                    self.log_updated.emit(result['log'])

                img_path = result["path"]
                img_dir = os.path.dirname(img_path)
                img_file = os.path.basename(img_path)

                if result["status"] == "success":
                    matched_keyword = result["keyword"]
                    keyword_dir = self._move_image(img_dir, img_file, img_path, matched_keyword, keyword_counters)
                    if keyword_dir and keyword_dir not in next_dirs:
                        next_dirs.append(keyword_dir)
                elif result["status"] in ["no_keyword_match", "no_prompt"]:
                    unmatched_images.append((img_dir, img_file, img_path))

                processed_count += 1
                progress = int((processed_count / total_images) * 100)
                self.progress_updated.emit(progress)

        if self.handle_others and unmatched_images:
            self.log_updated.emit(f"{len(unmatched_images)}개의 분류되지 않은 파일을 'other' 폴더로 이동합니다...")
            other_counters = {'other': 0}
            for img_dir, img_file, img_path in unmatched_images:
                if self.canceled: break
                self._move_image(img_dir, img_file, img_path, 'other', other_counters)
        
        return next_dirs
    
    def _move_image(self, img_dir, img_file, img_path, keyword, counters):
        if self.custom_dest_enabled and self.custom_dest_path:
            target_dir = self.custom_dest_path
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
        else:
            target_dir = os.path.join(img_dir, keyword)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
        
        if self.rename_images:
            counters[keyword] += 1
            dest_filename = f"{keyword}_{str(counters[keyword]).zfill(6)}{os.path.splitext(img_file)[1]}"
        else:
            dest_filename = img_file
            
        dest_path = os.path.join(target_dir, dest_filename)

        if os.path.exists(dest_path):
            if self.resolve_conflicts:
                base, ext = os.path.splitext(dest_path)
                counter = 1
                new_dest_path = f"{base} ({str(counter).zfill(2)}){ext}"
                while os.path.exists(new_dest_path):
                    counter += 1
                    new_dest_path = f"{base} ({str(counter).zfill(2)}){ext}"
                dest_path = new_dest_path
                self.log_updated.emit(f"알림: 이름 충돌로 '{os.path.basename(dest_path)}'(으)로 저장")
            else:
                self.log_updated.emit(f"경고: '{os.path.basename(dest_path)}' 파일이 이미 존재하여 건너뜁니다.")
                return None

        self.moved_files.append((img_path, dest_path))
        shutil.move(img_path, dest_path)
        self.log_updated.emit(f"{img_file} -> {os.path.relpath(dest_path, self.source_dir)}")
        
        return target_dir
    
    def cancel(self):
        self.canceled = True


class ImageClassifierApp(QMainWindow):

    def undo_classification(self):
        if not self.worker or not hasattr(self.worker, 'moved_files') or not self.worker.moved_files:
            QMessageBox.warning(self, "경고", "취소할 수 있는 작업이 없습니다.")
            return
        
        reply = QMessageBox.question(self, '작업 취소', 
                                     "최근 분류 작업을 취소하시겠습니까? 이동된 파일이 원래 위치로 복원됩니다.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.log_text.append("이전 작업을 취소하는 중...")
            self.worker.undo_classification()
            self.progress_bar.setValue(0)
            QMessageBox.information(self, "완료", "이전 작업이 취소되었습니다.")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.source_dir = ""
        self.worker = None
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
         full_tracking_enabled, full_tracking_prompt, custom_dest_enabled, custom_dest_path) = self.settings_manager.get_settings_for_ui()
        
        self.source_dir = source_dir
        self.dir_path_label.setText(source_dir if source_dir else "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(rename_images)
        self.handle_others_check.setChecked(handle_others)
        self.resolve_conflicts_check.setChecked(resolve_conflicts)
        
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
            self.custom_dest_path_input.text()
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
                self.custom_dest_path_input.text()
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
        self.undo_btn.clicked.connect(self.undo_classification)
        buttons_layout.addWidget(self.undo_btn)
        main_layout.addLayout(buttons_layout)
        
        central_widget.setLayout(main_layout)
    
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

        prompt_levels = [(chk.isChecked(), inp.text()) for chk, inp in self.prompt_inputs]
        is_any_level_active = any(enabled for enabled, _ in prompt_levels)
        is_full_tracking_active = self.full_tracking_check.isChecked() and self.full_tracking_prompt_input.text().strip()

        if not is_full_tracking_active and not is_any_level_active and not self.handle_others_check.isChecked():
            QMessageBox.warning(self, "경고", "실행할 작업이 없습니다...")
            return

        # ... (other validation checks)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.worker = ImageClassifierWorker(
            self.source_dir, 
            prompt_levels,
            self.rename_check.isChecked(),
            self.handle_others_check.isChecked(),
            self.resolve_conflicts_check.isChecked(),
            self.multicore_check.isChecked(),
            self.core_count_spinbox.value(),
            self.full_tracking_check.isChecked(),
            self.full_tracking_prompt_input.text(),
            self.custom_dest_check.isChecked(),
            self.custom_dest_path_input.text()
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_updated.connect(self.update_log)
        self.worker.completed.connect(self.classification_completed)
        self.worker.start()
        
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
        
    def classification_completed(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
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