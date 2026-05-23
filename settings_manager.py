"""
애플리케이션 설정 및 프리셋을 관리하는 모듈
설정 저장, 로드 및 프리셋 관리 기능 제공
"""
import os
import json
import logging
import sys
from typing import Dict, List, Tuple, Any, Optional


class SettingsManager:
    def __init__(self, app_name: str = "ImageClassifier"):
        self.app_name = app_name
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
            self.settings_dir = os.path.join(base_dir, app_name)
        else:
            self.settings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app_name)
        self.settings_file = os.path.join(self.settings_dir, "settings.json")
        self.presets_dir = os.path.join(self.settings_dir, "presets")
        self.logger = logging.getLogger(app_name)
        self._ensure_directories()
        self.default_settings = self._get_default_settings()
        self.current_settings = self.load_settings()

    def _ensure_directories(self) -> None:
        os.makedirs(self.settings_dir, exist_ok=True)
        os.makedirs(self.presets_dir, exist_ok=True)

    def _get_default_settings(self) -> Dict[str, Any]:
        return {
            "source_directory": "",
            "rename_images": False,          # 하위 호환 유지(로드 시 rename_mode로 변환)
            "rename_mode": "keep",           # "keep" | "prompt" | "custom"
            "rename_custom_name": "",        # rename_mode == "custom" 일 때 사용
            "handle_others": "off",          # "off" | "after" | "per_level"
            "resolve_conflicts": False,
            "multicore_enabled": False,
            "multicore_core_count": os.cpu_count() or 4,
            "prompt_levels": [
                {"enabled": True,  "prompt": "", "exclude_prompt": ""},
                {"enabled": False, "prompt": "", "exclude_prompt": ""},
                {"enabled": False, "prompt": "", "exclude_prompt": ""},
                {"enabled": False, "prompt": "", "exclude_prompt": ""},
                {"enabled": False, "prompt": "", "exclude_prompt": ""},
            ],
            "full_tracking_enabled": False,
            "full_tracking_prompt": "",
            "full_tracking_exclude_prompt": "",
            "custom_dest_enabled": False,
            "custom_dest_path": "",
            "safe_mode_enabled": False,
            "clone_mode_enabled": False,
            # ── 해상도 분류 ──────────────────────────────────────────
            "res_enabled": False,
            "res_standalone": False,        # True 이면 프롬프트 분류 없이 해상도만 실행
            "res_timing": "after",          # "before" | "after"
            "res_folder_format": "wh",      # "wh" | "wh_dir" | "ratio"
            "res_group_by_ratio": False,
            "res_ignore_orientation": False,
            "res_tolerance_enabled": False,
            "res_tolerance_mode": "percent", # "percent" | "pixel"
            "res_tolerance_value": 5,
            "res_min_count_enabled": False,
            "res_min_count": 3,
            "res_cutoff_enabled": False,
            "res_cutoff_min": 0,            # 0 = 비활성
            "res_cutoff_max": 0,            # 0 = 비활성
            "res_whitelist_enabled": False,
            "res_blacklist_enabled": False,
            "res_whitelist": [],            # [{"w": 1216, "h": 832}, ...]
            "res_blacklist": [],
        }

    def load_settings(self) -> Dict[str, Any]:
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return self._validate_settings(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"설정 로드 오류: {e}")
        return self.default_settings.copy()

    def _validate_settings(self, s: Dict[str, Any]) -> Dict[str, Any]:
        v = self.default_settings.copy()

        for key in ("source_directory", "full_tracking_prompt", "full_tracking_exclude_prompt",
                    "custom_dest_path", "res_timing", "res_folder_format", "res_tolerance_mode",
                    "rename_mode", "rename_custom_name"):
            if key in s and isinstance(s[key], str):
                v[key] = s[key]

        # handle_others: 하위 호환 (구버전 bool → 신버전 str)
        if "handle_others" in s:
            ho = s["handle_others"]
            if isinstance(ho, bool):
                v["handle_others"] = "after" if ho else "off"
            elif isinstance(ho, str) and ho in ("off", "after", "per_level"):
                v["handle_others"] = ho

        # 하위 호환: 구버전 rename_images(bool) → rename_mode 변환
        if "rename_images" in s and isinstance(s["rename_images"], bool):
            if "rename_mode" not in s:
                v["rename_mode"] = "prompt" if s["rename_images"] else "keep"

        for key in ("rename_images", "resolve_conflicts", "multicore_enabled",
                    "full_tracking_enabled", "custom_dest_enabled", "safe_mode_enabled", "clone_mode_enabled",
                    "res_enabled", "res_standalone", "res_group_by_ratio", "res_ignore_orientation",
                    "res_tolerance_enabled", "res_min_count_enabled", "res_cutoff_enabled",
                    "res_whitelist_enabled", "res_blacklist_enabled"):
            if key in s and isinstance(s[key], bool):
                v[key] = s[key]

        for key in ("multicore_core_count", "res_tolerance_value", "res_min_count",
                    "res_cutoff_min", "res_cutoff_max"):
            if key in s and isinstance(s[key], (int, float)):
                v[key] = int(s[key])

        if "prompt_levels" in s and isinstance(s["prompt_levels"], list):
            for i, level in enumerate(s["prompt_levels"]):
                if i < len(v["prompt_levels"]) and isinstance(level, dict):
                    if "enabled" in level and isinstance(level["enabled"], bool):
                        v["prompt_levels"][i]["enabled"] = level["enabled"]
                    if "prompt" in level and isinstance(level["prompt"], str):
                        v["prompt_levels"][i]["prompt"] = level["prompt"]
                    if "exclude_prompt" in level and isinstance(level["exclude_prompt"], str):
                        v["prompt_levels"][i]["exclude_prompt"] = level["exclude_prompt"]

        for key in ("res_whitelist", "res_blacklist"):
            if key in s and isinstance(s[key], list):
                cleaned = []
                for item in s[key]:
                    if isinstance(item, dict) and "w" in item and "h" in item:
                        try:
                            cleaned.append({"w": int(item["w"]), "h": int(item["h"])})
                        except (ValueError, TypeError):
                            pass
                v[key] = cleaned

        return v

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        try:
            validated = self._validate_settings(settings)
            self.current_settings = validated
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(validated, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"설정 저장 오류: {e}")
            return False

    def get_preset_list(self) -> List[str]:
        try:
            if not os.path.exists(self.presets_dir):
                return []
            return sorted(f[:-5] for f in os.listdir(self.presets_dir)
                          if f.endswith('.json') and os.path.isfile(os.path.join(self.presets_dir, f)))
        except IOError as e:
            self.logger.error(f"프리셋 목록 오류: {e}")
            return []

    def save_preset(self, name: str, settings: Optional[Dict[str, Any]] = None) -> bool:
        if not name or not isinstance(name, str):
            return False
        if settings is None:
            settings = self.current_settings
        try:
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(self._validate_settings(settings), f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"프리셋 '{name}' 저장 오류: {e}")
            return False

    def load_preset(self, name: str) -> Dict[str, Any]:
        if not name or not isinstance(name, str):
            return self.current_settings
        try:
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            if os.path.exists(preset_path):
                with open(preset_path, 'r', encoding='utf-8') as f:
                    return self._validate_settings(json.load(f))
            self.logger.warning(f"프리셋 '{name}' 없음")
            return self.current_settings
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"프리셋 '{name}' 로드 오류: {e}")
            return self.current_settings

    def delete_preset(self, name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        try:
            preset_path = os.path.join(self.presets_dir, f"{name}.json")
            if os.path.exists(preset_path):
                os.remove(preset_path)
                return True
            self.logger.warning(f"프리셋 '{name}' 없음")
            return False
        except IOError as e:
            self.logger.error(f"프리셋 '{name}' 삭제 오류: {e}")
            return False

    def get_settings_for_ui(self) -> tuple:
        s = self.current_settings
        prompt_levels = []
        for level in s.get("prompt_levels", []):
            prompt_levels.append((level.get("enabled", False),
                                  level.get("prompt", ""),
                                  level.get("exclude_prompt", "")))
        while len(prompt_levels) < 5:
            prompt_levels.append((False, "", ""))

        return (
            s.get("source_directory", ""),           # 0
            s.get("rename_mode", "keep"),            # 1  ← 변경 (구: rename_images bool)
            s.get("rename_custom_name", ""),         # 2  ← 신규
            s.get("handle_others", "off"),           # 3
            s.get("resolve_conflicts", False),       # 4  ← 구:3
            s.get("multicore_enabled", False),       # 5  ← 구:4
            s.get("multicore_core_count", os.cpu_count() or 4),  # 6  ← 구:5
            prompt_levels,                           # 7  ← 구:6
            s.get("full_tracking_enabled", False),  # 8  ← 구:7
            s.get("full_tracking_prompt", ""),       # 9  ← 구:8
            s.get("full_tracking_exclude_prompt", ""),  # 10 ← 구:9
            s.get("custom_dest_enabled", False),    # 11 ← 구:10
            s.get("custom_dest_path", ""),          # 12 ← 구:11
            s.get("safe_mode_enabled", False),      # 13 ← 구:12
            s.get("clone_mode_enabled", False),     # 14 ← 구:13
            # 해상도 분류
            s.get("res_enabled", False),            # 15 ← 구:14
            s.get("res_standalone", False),         # 16 ← 신규
            s.get("res_timing", "after"),           # 17 ← 구:15
            s.get("res_folder_format", "wh"),       # 18 ← 구:16
            s.get("res_group_by_ratio", False),     # 19 ← 구:17
            s.get("res_ignore_orientation", False), # 20 ← 구:18
            s.get("res_tolerance_enabled", False),  # 21 ← 구:19
            s.get("res_tolerance_mode", "percent"), # 22 ← 구:20
            s.get("res_tolerance_value", 5),        # 23 ← 구:21
            s.get("res_min_count_enabled", False),  # 24 ← 구:22
            s.get("res_min_count", 3),              # 25 ← 구:23
            s.get("res_cutoff_enabled", False),     # 26 ← 구:24
            s.get("res_cutoff_min", 0),             # 27 ← 구:25
            s.get("res_cutoff_max", 0),             # 28 ← 구:26
            s.get("res_whitelist_enabled", False),  # 29 ← 구:27
            s.get("res_blacklist_enabled", False),  # 30 ← 구:28
            s.get("res_whitelist", []),             # 31 ← 구:29
            s.get("res_blacklist", []),             # 32 ← 구:30
        )

    def create_settings_from_ui(self, source_dir, rename_mode, rename_custom_name,
                                handle_others, resolve_conflicts,
                                multicore_enabled, multicore_core_count, prompt_levels,
                                full_tracking_enabled, full_tracking_prompt, full_tracking_exclude_prompt,
                                custom_dest_enabled, custom_dest_path, safe_mode_enabled, clone_mode_enabled,
                                res_enabled, res_standalone, res_timing, res_folder_format, res_group_by_ratio,
                                res_ignore_orientation, res_tolerance_enabled, res_tolerance_mode,
                                res_tolerance_value, res_min_count_enabled, res_min_count,
                                res_cutoff_enabled, res_cutoff_min, res_cutoff_max,
                                res_whitelist_enabled, res_blacklist_enabled,
                                res_whitelist, res_blacklist) -> Dict[str, Any]:
        levels = []
        for item in prompt_levels:
            levels.append({
                "enabled": item[0],
                "prompt": item[1] if len(item) > 1 else "",
                "exclude_prompt": item[2] if len(item) > 2 else "",
            })
        return {
            "source_directory": source_dir or "",
            "rename_mode": rename_mode or "keep",
            "rename_custom_name": rename_custom_name or "",
            "rename_images": (rename_mode == "prompt"),  # 하위 호환
            "handle_others": handle_others if handle_others in ("off", "after", "per_level") else "off",
            "resolve_conflicts": bool(resolve_conflicts),
            "multicore_enabled": bool(multicore_enabled),
            "multicore_core_count": int(multicore_core_count),
            "prompt_levels": levels,
            "full_tracking_enabled": bool(full_tracking_enabled),
            "full_tracking_prompt": full_tracking_prompt or "",
            "full_tracking_exclude_prompt": full_tracking_exclude_prompt or "",
            "custom_dest_enabled": bool(custom_dest_enabled),
            "custom_dest_path": custom_dest_path or "",
            "safe_mode_enabled": bool(safe_mode_enabled),
            "clone_mode_enabled": bool(clone_mode_enabled),
            "res_enabled": bool(res_enabled),
            "res_standalone": bool(res_standalone),
            "res_timing": res_timing or "after",
            "res_folder_format": res_folder_format or "wh",
            "res_group_by_ratio": bool(res_group_by_ratio),
            "res_ignore_orientation": bool(res_ignore_orientation),
            "res_tolerance_enabled": bool(res_tolerance_enabled),
            "res_tolerance_mode": res_tolerance_mode or "percent",
            "res_tolerance_value": int(res_tolerance_value),
            "res_min_count_enabled": bool(res_min_count_enabled),
            "res_min_count": int(res_min_count),
            "res_cutoff_enabled": bool(res_cutoff_enabled),
            "res_cutoff_min": int(res_cutoff_min),
            "res_cutoff_max": int(res_cutoff_max),
            "res_whitelist_enabled": bool(res_whitelist_enabled),
            "res_blacklist_enabled": bool(res_blacklist_enabled),
            "res_whitelist": list(res_whitelist),
            "res_blacklist": list(res_blacklist),
        }
