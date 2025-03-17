"""
애플리케이션 설정 및 프리셋을 관리하는 모듈
설정 저장, 로드 및 프리셋 관리 기능 제공
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union


class SettingsManager:
    """
    애플리케이션 설정 및 프리셋을 관리하는 클래스
    """
    def __init__(self, app_name: str = "ImageClassifier"):
        """
        설정 관리자 초기화
        
        Args:
            app_name: 애플리케이션 이름 (폴더명으로 사용)
        """
        self.app_name = app_name
        
        # 현재 실행 파일이 있는 디렉토리를 기준으로 경로 설정
        self.settings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app_name)
        self.settings_file = os.path.join(self.settings_dir, "settings.json")
        self.presets_dir = os.path.join(self.settings_dir, "presets")
        
        # 로깅 설정
        self.logger = logging.getLogger(app_name)
        
        # 필요한 디렉토리 생성
        self._ensure_directories()
        
        # 기본 설정
        self.default_settings = self._get_default_settings()
        
        # 현재 설정 로드
        self.current_settings = self.load_settings()
    
    def _ensure_directories(self) -> None:
        """필요한 디렉토리가 존재하는지 확인하고 생성"""
        os.makedirs(self.settings_dir, exist_ok=True)
        os.makedirs(self.presets_dir, exist_ok=True)
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "source_directory": "",
            "rename_images": False,
            "prompt_levels": [
                {"enabled": True, "prompt": ""},
                {"enabled": False, "prompt": ""},
                {"enabled": False, "prompt": ""},
                {"enabled": False, "prompt": ""},
                {"enabled": False, "prompt": ""}
            ]
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """
        마지막으로 사용한 설정을 로드
        
        Returns:
            설정 정보를 담은 딕셔너리
        """
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                return self._validate_settings(settings)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"설정 로드 중 오류 발생: {e}")
                return self.default_settings.copy()
        else:
            return self.default_settings.copy()
    
    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        로드된 설정 유효성 검사 및 필요 시 기본값으로 보완
        
        Args:
            settings: 검증할 설정 딕셔너리
            
        Returns:
            검증된 설정 딕셔너리
        """
        validated = self.default_settings.copy()
        
        # 기본 필드 검증
        if "source_directory" in settings and isinstance(settings["source_directory"], str):
            validated["source_directory"] = settings["source_directory"]
            
        if "rename_images" in settings and isinstance(settings["rename_images"], bool):
            validated["rename_images"] = settings["rename_images"]
            
        # 프롬프트 레벨 검증
        if "prompt_levels" in settings and isinstance(settings["prompt_levels"], list):
            for i, level in enumerate(settings["prompt_levels"]):
                if i < len(validated["prompt_levels"]):
                    if isinstance(level, dict):
                        if "enabled" in level and isinstance(level["enabled"], bool):
                            validated["prompt_levels"][i]["enabled"] = level["enabled"]
                        if "prompt" in level and isinstance(level["prompt"], str):
                            validated["prompt_levels"][i]["prompt"] = level["prompt"]
                
        return validated
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        현재 설정을 저장
        
        Args:
            settings: 저장할 설정 딕셔너리
            
        Returns:
            성공 여부
        """
        try:
            # 설정 유효성 검사
            validated_settings = self._validate_settings(settings)
            self.current_settings = validated_settings
            
            # 파일 저장
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(validated_settings, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"설정 저장 중 오류 발생: {e}")
            return False
    
    def get_preset_list(self) -> List[str]:
        """
        사용 가능한 프리셋 목록 반환
        
        Returns:
            프리셋 이름 목록
        """
        try:
            if not os.path.exists(self.presets_dir):
                return []
                
            presets = [f[:-5] for f in os.listdir(self.presets_dir) 
                      if f.endswith('.json') and os.path.isfile(os.path.join(self.presets_dir, f))]
            return sorted(presets)
        except IOError as e:
            self.logger.error(f"프리셋 목록 로드 중 오류 발생: {e}")
            return []
    
    def save_preset(self, name: str, settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        현재 설정을 프리셋으로 저장
        
        Args:
            name: 프리셋 이름
            settings: 저장할 설정 딕셔너리. None이면 현재 설정 사용
            
        Returns:
            성공 여부
        """
        if not name or not isinstance(name, str):
            return False
            
        if settings is None:
            settings = self.current_settings
            
        try:
            # 유효성 검사된 설정
            validated_settings = self._validate_settings(settings)
            
            # 저장
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(validated_settings, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"프리셋 '{name}' 저장 중 오류 발생: {e}")
            return False
    
    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        저장된 프리셋 로드
        
        Args:
            name: 프리셋 이름
            
        Returns:
            프리셋 설정 딕셔너리. 로드 실패 시 현재 설정 반환
        """
        if not name or not isinstance(name, str):
            return self.current_settings
            
        try:
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            if os.path.exists(preset_path):
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset = json.load(f)
                return self._validate_settings(preset)
            else:
                self.logger.warning(f"프리셋 '{name}'을(를) 찾을 수 없습니다.")
                return self.current_settings
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"프리셋 '{name}' 로드 중 오류 발생: {e}")
            return self.current_settings
    
    def delete_preset(self, name: str) -> bool:
        """
        저장된 프리셋 삭제
        
        Args:
            name: 삭제할 프리셋 이름
            
        Returns:
            성공 여부
        """
        if not name or not isinstance(name, str):
            return False
            
        try:
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            if os.path.exists(preset_path):
                os.remove(preset_path)
                return True
            else:
                self.logger.warning(f"프리셋 '{name}'을(를) 찾을 수 없습니다.")
                return False
        except IOError as e:
            self.logger.error(f"프리셋 '{name}' 삭제 중 오류 발생: {e}")
            return False
    
    def get_settings_for_ui(self) -> Tuple[str, bool, List[Tuple[bool, str]]]:
        """
        UI에 쉽게 적용할 수 있는 형태로 현재 설정 반환
        
        Returns:
            (소스 디렉토리, 이름 변경 여부, 프롬프트 레벨 목록) 튜플
        """
        source_dir = self.current_settings.get("source_directory", "")
        rename_images = self.current_settings.get("rename_images", False)
        
        prompt_levels = []
        for level in self.current_settings.get("prompt_levels", []):
            prompt_levels.append((level.get("enabled", False), level.get("prompt", "")))
        
        # 항상 5개의 레벨이 있도록 보장
        while len(prompt_levels) < 5:
            prompt_levels.append((False, ""))
        
        return source_dir, rename_images, prompt_levels
    
    def create_settings_from_ui(self, source_dir: str, rename_images: bool, 
                              prompt_levels: List[Tuple[bool, str]]) -> Dict[str, Any]:
        """
        UI 값에서 설정 딕셔너리 생성
        
        Args:
            source_dir: 소스 디렉토리 경로
            rename_images: 이름 변경 여부
            prompt_levels: 프롬프트 레벨 목록
            
        Returns:
            설정 딕셔너리
        """
        levels = [
            {
                "enabled": enabled,
                "prompt": prompt
            } 
            for enabled, prompt in prompt_levels
        ]
        
        settings = {
            "source_directory": source_dir or "",
            "rename_images": bool(rename_images),
            "prompt_levels": levels
        }
        
        return settings