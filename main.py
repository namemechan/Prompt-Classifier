import sys
import os
import shutil
import gzip
from PyQt5.QtWidgets import (QComboBox, QInputDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, QProgressBar,
                            QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
from settings_manager import SettingsManager
from image_utils import read_info_from_image




class ImageClassifierWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    completed = pyqtSignal()
    
    # ImageClassifierWorker 클래스에 새로운 메소드 추가
    def undo_classification(self):
        """분류 작업 취소"""
        if not self.moved_files:
            self.log_updated.emit("취소할 작업이 없습니다.")
            return
        
        # 이동된 파일들 복원
        success_count = 0
        for src_path, dest_path in reversed(self.moved_files):
            try:
                if os.path.exists(dest_path):
                    # 원본 디렉토리가 없으면 생성
                    os.makedirs(os.path.dirname(src_path), exist_ok=True)
                    shutil.move(dest_path, src_path)
                    success_count += 1
            except Exception as e:
                self.log_updated.emit(f"파일 복원 중 오류 발생: {str(e)}")
        
        # 생성된 빈 디렉토리 제거
        for dir_path in reversed(self.created_dirs):
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                self.log_updated.emit(f"디렉토리 제거 중 오류 발생: {str(e)}")
        
        self.log_updated.emit(f"{success_count}개 파일을 원래 위치로 복원했습니다.")
        self.moved_files = []
        self.created_dirs = []

    def __init__(self, source_dir, prompt_levels, rename_images=False, full_tracking_enabled=False, full_tracking_prompt="", custom_dest_enabled=False, custom_dest_path=""):
        super().__init__()
        self.source_dir = source_dir
        self.prompt_levels = prompt_levels  # List of tuples (enabled, prompt_string)
        self.rename_images = rename_images
        self.full_tracking_enabled = full_tracking_enabled
        self.full_tracking_prompt = full_tracking_prompt
        self.custom_dest_enabled = custom_dest_enabled
        self.custom_dest_path = custom_dest_path
        self.canceled = False
        
    # ImageClassifierWorker 클래스의 run 메소드 수정
    def run(self):
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
            if not prompt_keywords:
                self.log_updated.emit("전체추적 프롬프트가 비어있습니다. 작업을 중단합니다.")
                self.completed.emit()
                return
                
            self._process_images_by_keywords(image_files_with_paths, prompt_keywords)

        else:
            # 기존 프롬프트 레벨 처리 로직
            image_files = self._find_image_files(self.source_dir)
            
            if not image_files:
                self.log_updated.emit("이미지 파일을 찾을 수 없습니다.")
                self.completed.emit()
                return
            
            self.log_updated.emit(f"{len(image_files)}개의 이미지를 찾았습니다. 분류를 시작합니다...")
            
            # 초기 디렉토리 설정
            current_dirs = [self.source_dir]
            
            # 각 레벨에 대해 처리
            for level_idx, (enabled, prompt_string) in enumerate(self.prompt_levels):
                if not enabled or not prompt_string.strip():
                    continue
                    
                self.log_updated.emit(f"레벨 {level_idx+1} 처리 중 - 프롬프트: {prompt_string}")
                
                # 현재 디렉토리에서 모든 이미지 파일 가져오기 
                # - 이 부분은 이제 하위 디렉토리는 검색하지 않음
                level_images = self._collect_level_images(current_dirs)
                
                if not level_images:
                    self.log_updated.emit("처리할 이미지가 없습니다.")
                    break
                    
                # 프롬프트 키워드 분리
                prompt_keywords = [p.strip() for p in prompt_string.split('|') if p.strip()]
                
                # 프롬프트 키워드에 따라 이미지 처리
                next_dirs = self._process_images_by_keywords(level_images, prompt_keywords)
                
                # 다음 레벨을 위한 현재 디렉토리 업데이트
                if next_dirs:
                    current_dirs = next_dirs
                else:
                    # 생성된 폴더가 없으면 더 이상 진행하지 않음
                    self.log_updated.emit("더 이상 처리할 디렉토리가 없습니다.")
                    break
        
        self.log_updated.emit("분류가 완료되었습니다!")
        self.completed.emit()
    
    def _find_image_files(self, directory):
        """디렉토리에서 이미지 파일 찾기 - 최상위 디렉토리만 스캔"""
        return [f for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) 
                and os.path.isfile(os.path.join(directory, f))]
    
    def _collect_level_images(self, directories):
        """여러 디렉토리에서 이미지 파일 수집 - 지정된 디렉토리만 스캔"""
        level_images = []
        for directory in directories:
            # 각 디렉토리의 직접적인 이미지 파일만 가져옴(하위 폴더 검색 안함)
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and os.path.isfile(file_path):
                    level_images.append((directory, file))
        return level_images

    def _find_all_image_files_recursive(self, directory):
        """지정된 디렉토리와 모든 하위 디렉토리에서 이미지 파일 찾기"""
        image_files_with_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files_with_paths.append((root, file))
        return image_files_with_paths
    
    def _process_images_by_keywords(self, images, keywords):
        """키워드에 따라 이미지 처리"""
        total_images = len(images)
        processed = 0
        next_dirs = []
        
        # 키워드별 카운터 초기화
        keyword_counters = {keyword: 0 for keyword in keywords}
        
        for img_dir, img_file in images:
            if self.canceled:
                self.log_updated.emit("작업이 취소되었습니다.")
                return []
            
            img_path = os.path.join(img_dir, img_file)
            try:
                # 이미지에서 프롬프트 데이터 읽기
                prompt_data = read_info_from_image(img_path)
                
                if prompt_data:
                    # 일치하는 키워드 찾기
                    matched_keyword = self._find_matching_keyword(prompt_data, keywords)
                    
                    if matched_keyword:
                        # 이미지 분류 및 이동
                        keyword_dir = self._move_image(img_dir, img_file, img_path, matched_keyword, keyword_counters)
                        
                        # 다음 레벨을 위한 디렉토리 추가
                        if keyword_dir not in next_dirs:
                            next_dirs.append(keyword_dir)
                    else:
                        self.log_updated.emit(f"{img_file}에서 일치하는 프롬프트를 찾을 수 없습니다.")
                else:
                    self.log_updated.emit(f"{img_file}에서 프롬프트 데이터를 찾을 수 없습니다.")
            except Exception as e:
                self.log_updated.emit(f"{img_file} 처리 중 오류 발생: {str(e)}")
            
            processed += 1
            progress = int((processed / total_images) * 100)
            self.progress_updated.emit(progress)
        
        return next_dirs
    
    def _find_matching_keyword(self, prompt_data, keywords):
        """프롬프트 데이터에서 일치하는 키워드 찾기"""
        for keyword in keywords:
            if keyword.lower() in prompt_data.lower():
                return keyword
        return None
    
    # _move_image 메소드 수정
    def _move_image(self, img_dir, img_file, img_path, keyword, counters):
        """이미지를 해당 키워드 폴더 또는 사용자 지정 폴더로 이동"""
        if self.custom_dest_enabled and self.custom_dest_path:
            # 사용자 지정 대상 폴더가 활성화된 경우
            target_dir = self.custom_dest_path
            # 사용자 지정 대상 폴더가 없으면 생성
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
        else:
            # 기존 로직: 키워드에 대한 하위 디렉토리 생성
            target_dir = os.path.join(img_dir, keyword)
            # 새로 생성된 디렉토리 추적
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.created_dirs.append(target_dir)
            else:
                os.makedirs(target_dir, exist_ok=True)
        
        # 이미지 이름 변경 처리
        if self.rename_images:
            # 사용자 지정 대상 폴더로 이동 시에도 키워드 기반 이름 변경 유지
            counters[keyword] += 1
            new_filename = f"{keyword}_{str(counters[keyword]).zfill(6)}{os.path.splitext(img_file)[1]}"
            dest_path = os.path.join(target_dir, new_filename)
        else:
            dest_path = os.path.join(target_dir, img_file)
        
        # 이동 전 원본 경로와 대상 경로를 저장 (취소용)
        self.moved_files.append((img_path, dest_path))
        
        # 이미지 이동
        shutil.move(img_path, dest_path)
        self.log_updated.emit(f"{img_file}을(를) {target_dir} 폴더로 이동했습니다.")
        
        return target_dir
    
    def cancel(self):
        self.canceled = True


class ImageClassifierApp(QMainWindow):

    # ImageClassifierApp 클래스에 undo_classification 메소드 추가
    def undo_classification(self):
        """최근 분류 작업 취소"""
        if not self.worker or not hasattr(self.worker, 'moved_files'):
            QMessageBox.warning(self, "경고", "취소할 수 있는 작업이 없습니다.")
            return
        
        reply = QMessageBox.question(self, '작업 취소', 
                                     "최근 분류 작업을 취소하시겠습니까? 이동된 파일이 원래 위치로 복원됩니다.",
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)
        
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
        """프리셋 콤보박스 업데이트"""
        self.preset_combo.clear()
        self.preset_combo.addItem("기본 설정")
        presets = self.settings_manager.get_preset_list()
        if presets:
            for preset in presets:
                self.preset_combo.addItem(preset)
    
    def load_settings(self):
        """마지막 설정 로드"""
        source_dir, rename_images, prompt_levels, full_tracking_enabled, full_tracking_prompt, custom_dest_enabled, custom_dest_path = self.settings_manager.get_settings_for_ui()
        
        # UI에 설정 적용
        self.source_dir = source_dir
        self.dir_path_label.setText(source_dir if source_dir else "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(rename_images)
        
        # 전체추적 설정
        self.full_tracking_check.setChecked(full_tracking_enabled)
        self.full_tracking_prompt_input.setText(full_tracking_prompt)
        self._toggle_full_tracking_input(full_tracking_enabled) # UI 상태 업데이트
        
        # 사용자 지정 대상 폴더 설정
        self.custom_dest_check.setChecked(custom_dest_enabled)
        self.custom_dest_path_input.setText(custom_dest_path)
        self._toggle_custom_dest_input(custom_dest_enabled) # UI 상태 업데이트
        
        # 프롬프트 레벨 설정
        for i, (level_check, prompt_input) in enumerate(self.prompt_inputs):
            if i < len(prompt_levels):
                level_check.setChecked(prompt_levels[i][0])
                prompt_input.setText(prompt_levels[i][1])
        
        # 프리셋 목록 업데이트
        self.update_preset_list()
    
    def save_current_settings(self):
        """현재 설정 저장"""
        # UI에서 현재 설정 수집
        prompt_levels = []
        for level_check, prompt_input in self.prompt_inputs:
            prompt_levels.append((level_check.isChecked(), prompt_input.text()))
    
        # 설정 저장
        settings = self.settings_manager.create_settings_from_ui(
            self.source_dir,
            self.rename_check.isChecked(),
            prompt_levels,
            self.full_tracking_check.isChecked(),
            self.full_tracking_prompt_input.text(),
            self.custom_dest_check.isChecked(),
            self.custom_dest_path_input.text()
        )
        self.settings_manager.save_settings(settings)
    
    def show_save_preset_dialog(self):
        """프리셋 저장 대화상자 표시"""
        name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름:")
        if ok and name:
            # 현재 설정 수집
            prompt_levels = []
            for level_check, prompt_input in self.prompt_inputs:
                prompt_levels.append((level_check.isChecked(), prompt_input.text()))
            
            # 설정 저장
            settings = self.settings_manager.create_settings_from_ui(
                self.source_dir,
                self.rename_check.isChecked(),
                prompt_levels
            )
            
            # 프리셋으로 저장
            if self.settings_manager.save_preset(name, settings):
                QMessageBox.information(self, "성공", f"프리셋 '{name}'이(가) 저장되었습니다.")
                self.update_preset_list()
                # 저장된 프리셋 선택
                index = self.preset_combo.findText(name)
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
            else:
                QMessageBox.warning(self, "오류", "프리셋 저장 중 오류가 발생했습니다.")
    
    def load_preset(self, index):
        """프리셋 로드"""
        if index <= 0:  # 기본 설정
            return
        
        preset_name = self.preset_combo.currentText()
        preset = self.settings_manager.load_preset(preset_name)
        
        # UI에 설정 적용
        self.source_dir = preset.get("source_directory", "")
        self.dir_path_label.setText(self.source_dir if self.source_dir else "디렉토리가 선택되지 않았습니다")
        self.rename_check.setChecked(preset.get("rename_images", False))
        
        # 프롬프트 레벨 설정
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
        """프리셋 삭제"""
        if self.preset_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "경고", "기본 설정은 삭제할 수 없습니다.")
            return
        
        preset_name = self.preset_combo.currentText()
        reply = QMessageBox.question(self, '프리셋 삭제', 
                                     f"프리셋 '{preset_name}'을(를) 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)
    
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
        
        # 소스 디렉토리 선택
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("소스 디렉토리:"))
        self.dir_path_label = QLabel("디렉토리가 선택되지 않았습니다")
        dir_layout.addWidget(self.dir_path_label, 1)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        main_layout.addLayout(dir_layout)

        # 프리셋 컨트롤
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
        
        # 이름 변경 옵션
        rename_layout = QHBoxLayout()
        self.rename_check = QCheckBox("프롬프트에 맞게 이미지 이름 변경")
        rename_layout.addWidget(self.rename_check)
        rename_layout.addStretch()
        main_layout.addLayout(rename_layout)
        
        # 전체추적 기능
        full_tracking_layout = QHBoxLayout()
        self.full_tracking_check = QCheckBox("전체추적 활성화")
        self.full_tracking_check.setChecked(False) # 기본적으로 비활성화
        self.full_tracking_check.toggled.connect(self._toggle_full_tracking_input)
        full_tracking_layout.addWidget(self.full_tracking_check)
        
        self.full_tracking_prompt_input = QLineEdit()
        self.full_tracking_prompt_input.setPlaceholderText("전체추적 프롬프트를 | 문자로 구분하여 입력")
        self.full_tracking_prompt_input.setEnabled(False) # 초기에는 비활성화
        full_tracking_layout.addWidget(self.full_tracking_prompt_input, 1)
        main_layout.addLayout(full_tracking_layout)
        
        # 사용자 지정 대상 폴더 기능
        custom_dest_layout = QHBoxLayout()
        self.custom_dest_check = QCheckBox("사용자 지정 대상 폴더 사용")
        self.custom_dest_check.setChecked(False) # 기본적으로 비활성화
        self.custom_dest_check.toggled.connect(self._toggle_custom_dest_input)
        custom_dest_layout.addWidget(self.custom_dest_check)
        
        self.custom_dest_path_input = QLineEdit()
        self.custom_dest_path_input.setPlaceholderText("이미지를 이동할 사용자 지정 폴더 경로")
        self.custom_dest_path_input.setEnabled(False) # 초기에는 비활성화
        custom_dest_layout.addWidget(self.custom_dest_path_input, 1)
        
        self.browse_custom_dest_btn = QPushButton("찾아보기...")
        self.browse_custom_dest_btn.clicked.connect(self._browse_custom_dest_directory)
        self.browse_custom_dest_btn.setEnabled(False) # 초기에는 비활성화
        custom_dest_layout.addWidget(self.browse_custom_dest_btn)
        main_layout.addLayout(custom_dest_layout)
        
        # 프롬프트 레벨 입력 (최대 5개)
        self.prompt_inputs = []
        
        for i in range(5):
            level_layout = QHBoxLayout()
            
            # 이 레벨을 활성화/비활성화하는 체크박스
            level_check = QCheckBox(f"레벨 {i+1}:")
            level_check.setChecked(i == 0)  # 기본적으로 첫 번째 레벨 활성화
            level_layout.addWidget(level_check)
            
            # 프롬프트 입력 필드
            prompt_input = QLineEdit()
            prompt_input.setPlaceholderText(f"레벨 {i+1} 프롬프트를 | 문자로 구분하여 입력")
            level_layout.addWidget(prompt_input, 1)
            
            main_layout.addLayout(level_layout)
            self.prompt_inputs.append((level_check, prompt_input))
        
        # 진행 상태 표시줄
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 로그 영역
        main_layout.addWidget(QLabel("로그:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        
        # init_ui 메소드의 버튼 레이아웃 부분 수정
        # 실행 버튼
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("분류 시작")
        self.start_btn.clicked.connect(self.start_classification)
        buttons_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.cancel_classification)
        self.cancel_btn.setEnabled(False)
        buttons_layout.addWidget(self.cancel_btn)
        
        # 작업 취소 버튼 추가
        self.undo_btn = QPushButton("이전 작업 취소")
        self.undo_btn.clicked.connect(self.undo_classification)
        buttons_layout.addWidget(self.undo_btn)
        
        main_layout.addLayout(buttons_layout)
        
        central_widget.setLayout(main_layout)
    
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
            self.save_current_settings() # 디렉토리 변경 시 설정 저장
        
    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "소스 디렉토리 선택")
        if dir_path:
            self.source_dir = dir_path
            self.dir_path_label.setText(dir_path)
            self.save_current_settings()  # 디렉토리 변경 시 설정 저장

    def start_classification(self):
        if not self.source_dir:
            QMessageBox.warning(self, "경고", "소스 디렉토리가 선택되지 않았습니다.")
            return
    
        # 현재 설정 저장
        self.save_current_settings()

        # 유효성 검사
        if self.full_tracking_check.isChecked():
            if not self.full_tracking_prompt_input.text().strip():
                QMessageBox.warning(self, "경고", "전체추적 프롬프트가 비어있습니다.")
                return
        else:
            prompt_levels = []
            for level_check, prompt_input in self.prompt_inputs:
                prompt_levels.append((level_check.isChecked(), prompt_input.text()))
                
            # 최소한 하나의 프롬프트 레벨이 활성화되었는지 확인
            if not any(enabled for enabled, _ in prompt_levels):
                QMessageBox.warning(self, "경고", "최소한 하나의 프롬프트 레벨을 활성화해야 합니다.")
                return
                
            # 활성화된 모든 레벨에 프롬프트가 입력되었는지 확인
            for i, (enabled, prompt) in enumerate(prompt_levels):
                if enabled and not prompt.strip():
                    QMessageBox.warning(self, "경고", f"레벨 {i+1}의 프롬프트가 비어있습니다.")
                    return

        # 사용자 지정 대상 폴더가 활성화된 경우 경로 유효성 검사
        if self.custom_dest_check.isChecked():
            if not self.custom_dest_path_input.text().strip():
                QMessageBox.warning(self, "경고", "사용자 지정 대상 폴더 경로가 비어있습니다.")
                return
            if not os.path.isdir(self.custom_dest_path_input.text()):
                QMessageBox.warning(self, "경고", "사용자 지정 대상 폴더 경로가 유효하지 않습니다.")
                return
                
        # UI 상태 업데이트
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 워커 스레드 생성 및 시작
        self.worker = ImageClassifierWorker(
            self.source_dir, 
            prompt_levels if not self.full_tracking_check.isChecked() else [], # 전체추적 모드일 때는 빈 리스트 전달
            self.rename_check.isChecked(),
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
        # 스크롤을 항상 아래로 유지
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def classification_completed(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        QMessageBox.information(self, "완료", "이미지 분류가 완료되었습니다.")
        
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '작업 중단', 
                                         "작업이 진행 중입니다. 종료하시겠습니까?",
                                         QMessageBox.Yes | QMessageBox.No, 
                                         QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.load_settings()  # 앱 시작 시 마지막 설정 로드
    window.show()
    sys.exit(app.exec_())