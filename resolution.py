"""
해상도 분류 관련 기능 모듈
- 표준 해상도 기준셋 및 그룹 매칭 로직
- 필터(컷오프/화이트리스트/블랙리스트) 처리
- 폴더명 생성
- 화이트/블랙리스트 편집 다이얼로그
"""

from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)


# ──────────────────────────────────────────────────────────────────
#  표준 해상도 기준셋
# ──────────────────────────────────────────────────────────────────
_BASE_RESOLUTIONS = [
    (1024, 1024),
    (1152, 896),  (896,  1152),
    (1216, 832),  (832,  1216),
    (1344, 768),  (768,  1344),
    (1536, 640),  (640,  1536),
    (1280, 720),  (720,  1280),
    (1920, 1080), (1080, 1920),
    (2560, 1440), (1440, 2560),
    (3840, 2160), (2160, 3840),
]
_SCALES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

STANDARD_RESOLUTIONS: list[tuple[int, int]] = []
for _bw, _bh in _BASE_RESOLUTIONS:
    for _sc in _SCALES:
        _sw, _sh = int(round(_bw * _sc)), int(round(_bh * _sc))
        if _sw > 0 and _sh > 0 and (_sw, _sh) not in STANDARD_RESOLUTIONS:
            STANDARD_RESOLUTIONS.append((_sw, _sh))


# ──────────────────────────────────────────────────────────────────
#  내부 유틸
# ──────────────────────────────────────────────────────────────────
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _simplified_ratio(w: int, h: int) -> tuple[int, int]:
    g = _gcd(w, h)
    return w // g, h // g


def _normalize(w: int, h: int, ignore_orientation: bool) -> tuple[int, int]:
    """ignore_orientation 이 True 이면 항상 가로 >= 세로 순으로 반환."""
    if ignore_orientation and w < h:
        return h, w
    return w, h


# ──────────────────────────────────────────────────────────────────
#  공개 API
# ──────────────────────────────────────────────────────────────────
def make_res_folder_name(w: int, h: int, fmt: str, ignore_orientation: bool) -> str:
    """
    폴더명 생성.

    fmt:
        'wh'     → '1216x832'
        'wh_dir' → '1216x832_(가로)' 또는 '1216x832_(세로)'
        'ratio'  → '19:13'
    ignore_orientation 이 True 이면 표기 기준을 항상 가로 >= 세로로 맞춤.
    """
    nw, nh = _normalize(w, h, ignore_orientation)
    if fmt == "ratio":
        rw, rh = _simplified_ratio(nw, nh)
        return f"{rw}:{rh}"
    elif fmt == "wh_dir":
        direction = "가로" if w >= h else "세로"
        return f"{nw}x{nh}_({direction})"
    else:  # "wh"
        return f"{nw}x{nh}"


def find_res_group(
    w: int, h: int, cfg: dict
) -> tuple[int, int]:
    """
    (w, h) 가 속하는 대표 해상도 그룹을 반환.
    매칭 표준이 없으면 자기 자신(정규화된 값)을 반환.
    """
    ignore_ori  = cfg.get("res_ignore_orientation", False)
    group_ratio = cfg.get("res_group_by_ratio", False)
    tol_en      = cfg.get("res_tolerance_enabled", False)
    tol_mode    = cfg.get("res_tolerance_mode", "percent")
    tol_val     = cfg.get("res_tolerance_value", 5)

    nw, nh = _normalize(w, h, ignore_ori)

    # ── 비율 분류 ───────────────────────────────────────────────
    if group_ratio:
        target = _simplified_ratio(nw, nh)
        for sw, sh in STANDARD_RESOLUTIONS:
            snw, snh = _normalize(sw, sh, ignore_ori)
            if _simplified_ratio(snw, snh) == target:
                return _normalize(sw, sh, ignore_ori)
        # 표준에 없으면 비율 최소 단위 자체를 대표로
        return target

    # ── 허용 오차 ───────────────────────────────────────────────
    if tol_en:
        best: Optional[tuple[int, int]] = None
        best_dist = float("inf")
        for sw, sh in STANDARD_RESOLUTIONS:
            snw, snh = _normalize(sw, sh, ignore_ori)
            if tol_mode == "percent":
                dw = abs(nw - snw) / snw * 100 if snw else float("inf")
                dh = abs(nh - snh) / snh * 100 if snh else float("inf")
                dist = max(dw, dh)
            else:  # pixel
                dist = max(abs(nw - snw), abs(nh - snh))
            if dist <= tol_val and dist < best_dist:
                best_dist = dist
                best = (snw, snh)
        return best if best is not None else (nw, nh)

    # ── 정확 일치 ───────────────────────────────────────────────
    for sw, sh in STANDARD_RESOLUTIONS:
        snw, snh = _normalize(sw, sh, ignore_ori)
        if (nw, nh) == (snw, snh):
            return snw, snh

    return nw, nh  # 표준 미등록 → 자기 크기 그대로


def passes_res_filters(
    w: int, h: int, cfg: dict
) -> tuple[bool, Optional[str]]:
    """
    컷오프 / 블랙리스트 / 화이트리스트 검사.

    반환:
        (True,  None)           — 통과, 정상 분류
        (False, 'too_small')    — 컷오프 미달
        (False, 'too_large')    — 컷오프 초과
        (False, 'blacklisted')  — 블랙리스트 해당
        (False, 'not_whitelisted') — 화이트리스트 미포함
    """
    ignore_ori = cfg.get("res_ignore_orientation", False)
    total_px   = w * h

    def _match(item: dict) -> bool:
        bw, bh = item["w"], item["h"]
        if ignore_ori:
            return sorted([w, h]) == sorted([bw, bh])
        return w == bw and h == bh

    # 컷오프
    if cfg.get("res_cutoff_enabled", False):
        cmin = cfg.get("res_cutoff_min", 0)
        cmax = cfg.get("res_cutoff_max", 0)
        if cmin > 0 and total_px < cmin:
            return False, "too_small"
        if cmax > 0 and total_px > cmax:
            return False, "too_large"

    # 블랙리스트
    if cfg.get("res_blacklist_enabled", False):
        for item in cfg.get("res_blacklist", []):
            if _match(item):
                return False, "blacklisted"

    # 화이트리스트
    if cfg.get("res_whitelist_enabled", False):
        for item in cfg.get("res_whitelist", []):
            if _match(item):
                return True, None
        return False, "not_whitelisted"

    return True, None


# ── 특수 사유 → 폴더명 매핑 ───────────────────────────────────────
SPECIAL_FOLDER_MAP: dict[str, str] = {
    "too_small":       "res_too_small",
    "too_large":       "res_too_large",
    "blacklisted":     "res_blacklisted",
    "not_whitelisted": "res_not_whitelisted",
    "below_min":       "res_below_min",
    "no_size":         "res_no_size",
}


# ──────────────────────────────────────────────────────────────────
#  화이트/블랙리스트 편집 다이얼로그
# ──────────────────────────────────────────────────────────────────
_QUICK_PRESETS = [
    ("1024²",    1024, 1024),
    ("1152×896", 1152, 896),
    ("1216×832", 1216, 832),
    ("832×1216", 832,  1216),
    ("1344×768", 1344, 768),
    ("1280×720", 1280, 720),
    ("1920×1080",1920, 1080),
    ("2560×1440",2560, 1440),
    ("3840×2160",3840, 2160),
]


class ResListDialog(QDialog):
    """화이트리스트 또는 블랙리스트 편집 창."""

    def __init__(self, title: str, items: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(420, 440)
        layout = QVBoxLayout(self)

        hint = QLabel("분류에 포함하거나 제외할 해상도를 추가하세요.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 표
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["너비 (W)", "높이 (H)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        for item in items:
            self._add_row(item["w"], item["h"])

        # 행 추가 / 삭제
        edit_row = QHBoxLayout()
        add_btn = QPushButton("행 추가")
        add_btn.clicked.connect(lambda: self._add_row())
        del_btn = QPushButton("선택 삭제")
        del_btn.clicked.connect(self._del_row)
        edit_row.addWidget(add_btn)
        edit_row.addWidget(del_btn)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        # 빠른 추가
        layout.addWidget(QLabel("빠른 추가:"))
        quick_row = QHBoxLayout()
        for label, w, h in _QUICK_PRESETS:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, w=w, h=h: self._add_row(w, h))
            quick_row.addWidget(btn)
        layout.addLayout(quick_row)

        # 확인 / 취소
        ok_row = QHBoxLayout()
        ok_btn  = QPushButton("확인"); ok_btn.clicked.connect(self.accept)
        can_btn = QPushButton("취소"); can_btn.clicked.connect(self.reject)
        ok_row.addStretch()
        ok_row.addWidget(ok_btn)
        ok_row.addWidget(can_btn)
        layout.addLayout(ok_row)

    def _add_row(self, w: int = 1024, h: int = 1024):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(w)))
        self.table.setItem(row, 1, QTableWidgetItem(str(h)))

    def _del_row(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def get_items(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            try:
                w = int(self.table.item(row, 0).text())
                h = int(self.table.item(row, 1).text())
                if w > 0 and h > 0:
                    result.append({"w": w, "h": h})
            except (ValueError, AttributeError):
                pass
        return result