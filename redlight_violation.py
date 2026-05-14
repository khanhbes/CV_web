"""
Red Light Violation Detection System
=====================================
Phát hiện vi phạm vượt đèn đỏ sử dụng YOLOv8 Segmentation + Object Tracking

Features:
1. Calibration stopline bằng RANSAC (5 giây đầu)
2. Hệ tọa độ tương đối: Âm = đang tới, Dương = đã qua
3. Phát hiện vi phạm khi xe chuyển từ Âm sang Dương lúc đèn đỏ

Author: Traffic Violation Detection System
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path so we can import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_bottom_center(xyxy: np.ndarray) -> Tuple[float, float]:
    """Lấy điểm đáy-giữa của bounding box"""
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float(y2)


def get_signed_distance(px: float, py: float, a: float, b: float, c: float) -> float:
    """
    Tính khoảng cách có dấu từ điểm (px, py) đến đường thẳng ax + by + c = 0
    
    Returns:
        Giá trị có dấu: âm = một phía, dương = phía kia
    """
    return (a * px + b * py + c) / (np.sqrt(a * a + b * b) + 1e-12)


def get_distance_to_line(px: float, py: float, a: float, b: float, c: float) -> float:
    """Tính khoảng cách tuyệt đối từ điểm đến đường thẳng"""
    return abs(get_signed_distance(px, py, a, b, c))

# =============================================================================
# STOPLINE MASK LINE FITTING (from reference: minAreaRect + walk-along-mask)
# =============================================================================

def boxes_overlap(box_a, box_b):
    """Kiểm tra 2 bounding box có overlap không"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def merge_overlapping_detections(detections):
    """Gộp các detection overlap thành 1 (box + mask)"""
    merged = []
    used = [False] * len(detections)

    for i in range(len(detections)):
        if used[i]:
            continue
        used[i] = True
        current_box = detections[i]["box"].copy()
        current_conf = detections[i]["conf"]
        current_mask = detections[i]["mask"].copy() if detections[i]["mask"] is not None else None

        changed = True
        while changed:
            changed = False
            for j in range(len(detections)):
                if used[j]:
                    continue
                if boxes_overlap(current_box, detections[j]["box"]):
                    box_j = detections[j]["box"]
                    current_box = np.array([
                        min(current_box[0], box_j[0]),
                        min(current_box[1], box_j[1]),
                        max(current_box[2], box_j[2]),
                        max(current_box[3], box_j[3]),
                    ], dtype=np.float32)
                    current_conf = max(current_conf, detections[j]["conf"])
                    if current_mask is None:
                        if detections[j]["mask"] is not None:
                            current_mask = detections[j]["mask"].copy()
                    elif detections[j]["mask"] is not None:
                        current_mask = np.logical_or(current_mask, detections[j]["mask"])
                    used[j] = True
                    changed = True

        merged.append({"box": current_box, "conf": current_conf, "mask": current_mask})
    return merged


def line_inside_mask(mask):
    """Tìm đường centerline bên trong mask bằng minAreaRect + walk.
    
    Returns:
        (x1, y1, x2, y2, length) hoặc None
    """
    if mask is None:
        return None

    mask_u8 = (mask.astype(np.uint8) * 255)
    if cv2.countNonZero(mask_u8) == 0:
        return None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    theta = np.deg2rad(angle if rw >= rh else angle + 90.0)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None
    direction /= norm

    h, w = mask_u8.shape
    center_x = int(round(cx))
    center_y = int(round(cy))
    if (center_x < 0 or center_x >= w or center_y < 0 or center_y >= h
            or mask_u8[center_y, center_x] == 0):
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return None
        idx = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        cx = float(xs[idx])
        cy = float(ys[idx])
    else:
        cx = float(center_x)
        cy = float(center_y)

    def walk(sign):
        x, y = cx, cy
        last_x, last_y = x, y
        max_steps = int(max(h, w) * 4)
        step_size = 0.5
        for _ in range(max_steps):
            x += direction[0] * step_size * sign
            y += direction[1] * step_size * sign
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or xi >= w or yi < 0 or yi >= h or mask_u8[yi, xi] == 0:
                break
            last_x = x
            last_y = y
        return last_x, last_y

    p1x, p1y = walk(-1)
    p2x, p2y = walk(1)
    length = float(np.hypot(p2x - p1x, p2y - p1y))
    if length < 2.0:
        return None

    return int(round(p1x)), int(round(p1y)), int(round(p2x)), int(round(p2y)), length


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrackState:
    """Trạng thái tracking của mỗi xe"""
    last_region: Optional[int] = None     # -1: approaching, 0: near/touching, +1: fully crossed
    first_seen: bool = False
    last_event_frame: int = -1000
    last_warning_frame: int = -1000       # Frame cuối cùng set WARNING (debounce riêng)
    last_crossing_frame: int = -1000      # Frame cuối cùng phát hiện crossing (debounce riêng)
    label: str = "Safe"
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))
    positions: List[Tuple[float, float]] = field(default_factory=list)
    direction: str = "UNKNOWN"
    warned: bool = False                  # Đã hiển thị WARNING chạm vạch chưa
    prev_region_at_crossing: Optional[int] = None  # Vùng trước đó khi set VIOLATION (để track escalation)


@dataclass
class TrafficLightState:
    """Trạng thái chi tiết của tất cả đèn giao thông"""
    circle_red: bool = False
    circle_yellow: bool = False
    circle_green: bool = False
    left_red: bool = False
    left_yellow: bool = False
    left_green: bool = False
    straight_red: bool = False
    straight_yellow: bool = False
    straight_green: bool = False
    right_green: bool = False
    
    def get_allowed_directions(self) -> Dict[str, bool]:
        """Trả về các hướng được phép đi dựa trên tổ hợp đèn giao thông."""
        if self.circle_red:
            allowed = {'STRAIGHT': False, 'LEFT': False, 'RIGHT': False}
            if self.right_green:
                allowed['RIGHT'] = True
            if self.left_green:
                allowed['LEFT'] = True
            if self.straight_green:
                allowed['STRAIGHT'] = True
            return allowed
        
        if self.circle_green:
            allowed = {'STRAIGHT': True, 'LEFT': True, 'RIGHT': True}
            if self.left_red:
                allowed['LEFT'] = False
            if self.straight_red:
                allowed['STRAIGHT'] = False
            return allowed
        
        allowed = {'STRAIGHT': True, 'LEFT': True, 'RIGHT': True}
        if self.straight_red:
            allowed['STRAIGHT'] = False
        if self.left_red:
            allowed['LEFT'] = False
        if self.straight_green:
            allowed['STRAIGHT'] = True
        if self.left_green:
            allowed['LEFT'] = True
        if self.right_green:
            allowed['RIGHT'] = True
        return allowed
    
    def has_any_yellow(self) -> bool:
        return self.circle_yellow or self.left_yellow or self.straight_yellow
    
    def has_any_red(self) -> bool:
        return self.circle_red or self.left_red or self.straight_red
    
    def get_simple_state(self) -> str:
        if self.has_any_red():
            return "RED"
        if self.has_any_yellow():
            return "YELLOW"
        return "GREEN"


@dataclass  
class LightMemory:
    """Bộ nhớ trạng thái đèn giao thông"""
    last_state: TrafficLightState = field(default_factory=TrafficLightState)
    last_update_time: float = 0.0
    
    def update(self, state: TrafficLightState, current_time: float):
        if state.has_any_red() or state.has_any_yellow() or state.circle_green or state.left_green or state.straight_green or state.right_green:
            self.last_state = state
            self.last_update_time = current_time
    
    def get(self, current_time: float) -> TrafficLightState:
        if (current_time - self.last_update_time) <= config.LIGHT_MEMORY_DURATION:
            return self.last_state
        return TrafficLightState()


# =============================================================================
# STOPLINE CALIBRATOR
# =============================================================================

class StoplineCalibrator:
    """Bộ calibration stopline: tìm centerline trong mask, khóa sau N giây"""
    
    def __init__(self, duration: float = 5.0):
        self.duration = duration
        self.start_time = None
        self.line_abc: Optional[Tuple[float, float, float]] = None
        self.sign_flip: float = 1.0
        self.min_x: int = 0
        self.max_x: int = 0
        # Best line found during calibration
        self.best_line: Optional[Tuple[int, int, int, int]] = None
        self.best_line_length: float = 0.0
        
    def is_calibrated(self) -> bool:
        return self.line_abc is not None
    
    def start(self):
        self.start_time = time.time()
    
    def update_line(self, line_info):
        """Cập nhật line candidate từ line_inside_mask().
        Giữ lại line dài nhất (phủ nhiều mặt đường nhất).
        """
        if line_info is None:
            return
        x1, y1, x2, y2, length = line_info
        if length > self.best_line_length:
            self.best_line = (x1, y1, x2, y2)
            self.best_line_length = length
    
    def maybe_finish(self, frame_height: int, frame_width: int):
        """Kiểm tra và hoàn thành calibration"""
        if self.start_time is None or self.is_calibrated():
            return
        
        elapsed = time.time() - self.start_time
        if elapsed < self.duration:
            return
        
        if self.best_line is None:
            print("[ERROR] No stopline found during calibration!")
            return
        
        x1, y1, x2, y2 = self.best_line
        self.min_x = min(x1, x2)
        self.max_x = max(x1, x2)
        
        # Convert line segment → ax + by + c = 0
        a = float(y1 - y2)
        b = float(x2 - x1)
        c = float(x1 * y2 - x2 * y1)
        norm = np.sqrt(a * a + b * b) + 1e-12
        a, b, c = a / norm, b / norm, c / norm
        
        self.line_abc = (a, b, c)
        
        # Auto-orient: điểm dưới frame phải là ÂM (xe đang tới)
        ref_x, ref_y = frame_width / 2.0, frame_height - 10
        signed = a * ref_x + b * ref_y + c
        self.sign_flip = -1.0 if signed > 0 else 1.0
        
        print(f"[✓] Stopline calibrated: ({x1},{y1})->({x2},{y2}), length={self.best_line_length:.0f}px")
        print(f"    Line eq: {a:.4f}x + {b:.4f}y + {c:.2f} = 0")


# =============================================================================
# TRAFFIC LIGHT DETECTION
# =============================================================================

def get_all_light_classes():
    """Trả về tất cả class ID của đèn giao thông"""
    return (config.RED_LIGHTS + config.YELLOW_LIGHTS + config.GREEN_LIGHTS +
            config.ARROW_LEFT_RED + config.ARROW_LEFT_YELLOW + config.ARROW_LEFT_GREEN +
            config.ARROW_STRAIGHT_RED + config.ARROW_STRAIGHT_YELLOW + config.ARROW_STRAIGHT_GREEN +
            config.ARROW_RIGHT_GREEN)


def get_light_color(cls_id: int) -> Tuple[Tuple[int, int, int], str]:
    """Trả về màu và label chi tiết cho đèn giao thông"""
    # Mapping cls_id -> (color_bgr, short_english_label)
    LIGHT_LABELS = {
        # Đèn tròn (circle)
        18: ((0, 0, 255),   "RED"),              # light_straight_circle_red
        19: ((0, 255, 255), "YELLOW"),            # light_straight_circle_yellow
        17: ((0, 255, 0),   "GREEN"),             # light_straight_circle_green
        # Đèn mũi tên rẽ trái
        11: ((0, 0, 255),   "RED LEFT"),           # light_left_red
        12: ((0, 255, 255), "YEL LEFT"),           # light_left_yellow
        10: ((0, 255, 0),   "GRN LEFT"),           # light_left_green
        # Đèn mũi tên đi thẳng
        15: ((0, 0, 255),   "RED STR"),            # light_straight_arrow_red
        16: ((0, 255, 255), "YEL STR"),            # light_straight_arrow_yellow
        14: ((0, 255, 0),   "GRN STR"),            # light_straight_arrow_green
        # Đèn mũi tên rẽ phải
        13: ((0, 255, 0),   "GRN RIGHT"),          # light_right_green
    }
    
    if cls_id in LIGHT_LABELS:
        return LIGHT_LABELS[cls_id]
    return (128, 128, 128), "UNKNOWN"


def draw_traffic_lights(frame, boxes, cls_ids, confs):
    """Vẽ bounding box cho đèn giao thông - giữ best confidence per cls_id, NMS overlap"""
    all_light_classes = get_all_light_classes()
    
    # Giữ đèn có confidence cao nhất cho mỗi cls_id cụ thể
    # (tránh trùng lặp cùng loại, nhưng cho phép khác loại tồn tại song song)
    best_per_cls: Dict[int, Tuple] = {}
    
    for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
        if cls_id not in all_light_classes:
            continue
        if cls_id not in best_per_cls or conf > best_per_cls[cls_id][2]:
            best_per_cls[cls_id] = (xyxy, cls_id, conf)
    
    if not best_per_cls:
        return
    
    # Helper IoU
    def calc_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    # NMS: nếu 2 đèn khác loại nhưng box chồng nhau → giữ conf cao hơn
    lights_sorted = sorted(best_per_cls.values(), key=lambda x: x[2], reverse=True)
    final_lights = []
    for light_data in lights_sorted:
        xyxy, cls_id, conf = light_data
        is_overlap = False
        for existing in final_lights:
            if calc_iou(xyxy, existing[0]) > 0.5:  # Overlap > 50% → cùng vật thể
                is_overlap = True
                break
        if not is_overlap:
            final_lights.append(light_data)
    
    # Vẽ
    for xyxy, cls_id, conf in final_lights:
        color, label = get_light_color(cls_id)
        draw_bbox_with_label(frame, tuple(xyxy), f"{label} {conf:.2f}", color)


def detect_traffic_lights(cls_ids: np.ndarray) -> TrafficLightState:
    """Phát hiện chi tiết trạng thái các đèn giao thông"""
    state = TrafficLightState()
    
    if cls_ids is None or len(cls_ids) == 0:
        return state
    
    # Đèn tròn (Circle lights)
    state.circle_red = np.isin(cls_ids, config.RED_LIGHTS).any()        # class 18
    state.circle_yellow = np.isin(cls_ids, config.YELLOW_LIGHTS).any()  # class 19
    state.circle_green = np.isin(cls_ids, config.GREEN_LIGHTS).any()    # class 17
    
    # Đèn mũi tên rẽ trái
    state.left_red = np.isin(cls_ids, config.ARROW_LEFT_RED).any()      # class 11
    state.left_yellow = np.isin(cls_ids, config.ARROW_LEFT_YELLOW).any() # class 12
    state.left_green = np.isin(cls_ids, config.ARROW_LEFT_GREEN).any()  # class 10
    
    # Đèn mũi tên đi thẳng
    state.straight_red = np.isin(cls_ids, config.ARROW_STRAIGHT_RED).any()      # class 15
    state.straight_yellow = np.isin(cls_ids, config.ARROW_STRAIGHT_YELLOW).any() # class 16
    state.straight_green = np.isin(cls_ids, config.ARROW_STRAIGHT_GREEN).any()  # class 14
    
    # Đèn mũi tên rẽ phải
    state.right_green = np.isin(cls_ids, config.ARROW_RIGHT_GREEN).any()  # class 13
    
    return state


# Direction detection constants (tuned like wrong_lane_violation.py)
STRAIGHT_ANGLE_DEG = 26.0   # <= this angle from ref = STRAIGHT
TURN_ANGLE_DEG = 34.0       # >= this angle from ref = LEFT/RIGHT
MIN_STEP_MAG = 2.0          # Minimum pixel movement per step
MIN_MOVE_MAG_SUM = 60.0     # Minimum total movement magnitude
MIN_FORWARD_STEPS = 6       # Minimum forward-motion steps

# Stopline crossing buffer (pixels)
STOPLINE_TOUCH_DIST = 15.0  # Khoảng cách chạm vạch (WARNING zone)

# Debounce frames - separated by event type (not global cooldown)
DEBOUNCE_WARNING_FRAMES = 8       # Tránh toggle WARNING liên TỤC
DEBOUNCE_CROSSING_FRAMES = 4      # Tránh phát hiện crossing duplicate từ cùng motion
# Quy tắc: WARNING + CROSSING dùng debounce khác nhau
# - Warning debounce: khi từ approaching → touching dưới light red/yellow
# - Crossing debounce: khi từ (-1|0) → +1 (vượt hoàn toàn)


def detect_vehicle_direction(
    positions: List[Tuple[float, float]],
    ref_vector: Optional[Tuple[float, float]] = None,
    min_frames: int = 10
) -> str:
    """
    Phát hiện hướng đi của xe bằng vector/angle (cải tiến từ wrong_lane_violation.py).
    
    Sử dụng stopline normal làm reference vector (hướng đi thẳng chuẩn).
    So sánh góc giữa motion vector và ref vector:
      - angle <= STRAIGHT_ANGLE_DEG → STRAIGHT
      - angle >= TURN_ANGLE_DEG → LEFT/RIGHT (dựa vào cross product)
      - vùng mơ hồ giữa → bias STRAIGHT (giảm false positive)
    
    Args:
        positions: Danh sách (x, y) liên tục
        ref_vector: (vx, vy) hướng đi thẳng chuẩn (từ stopline normal), None = dùng (0, -1)
        min_frames: Số frame tối thiểu để phân tích
    
    Returns:
        'STRAIGHT', 'LEFT', 'RIGHT', hoặc 'UNKNOWN'
    """
    if len(positions) < min_frames:
        return "UNKNOWN"
    
    # Reference vector (mặc định đi lên = (0, -1))
    if ref_vector is None:
        v0x, v0y = 0.0, -1.0
    else:
        v0x, v0y = float(ref_vector[0]), float(ref_vector[1])
    v0n = np.hypot(v0x, v0y) + 1e-9
    v0x /= v0n
    v0y /= v0n
    
    # Tính các step di chuyển (lọc bước quá nhỏ)
    steps = []
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if np.hypot(dx, dy) < MIN_STEP_MAG:
            continue
        steps.append((dx, dy))
    
    if len(steps) < MIN_FORWARD_STEPS:
        return "UNKNOWN"
    
    # Tính hướng sơ bộ bằng median (ổn định hơn mean)
    dx0 = float(np.median([s[0] for s in steps]))
    dy0 = float(np.median([s[1] for s in steps]))
    if dx0 * dx0 + dy0 * dy0 < 1e-6:
        return "UNKNOWN"
    
    # Flip ref vector để cùng chiều motion
    if (dx0 * v0x + dy0 * v0y) < 0:
        v0x, v0y = -v0x, -v0y
    
    # Lọc forward steps (dot > 0 vs ref)
    fsteps = []
    mag_sum = 0.0
    for dx, dy in steps:
        if (dx * v0x + dy * v0y) <= 0.0:
            continue  # Bước ngược, bỏ qua
        fsteps.append((dx, dy))
        mag_sum += float(np.hypot(dx, dy))
    
    if len(fsteps) < MIN_FORWARD_STEPS or mag_sum < MIN_MOVE_MAG_SUM:
        return "UNKNOWN"
    
    # Tính motion vector cuối cùng (median of forward steps)
    dx_med = float(np.median([s[0] for s in fsteps]))
    dy_med = float(np.median([s[1] for s in fsteps]))
    vn = np.hypot(dx_med, dy_med) + 1e-9
    vx = dx_med / vn
    vy = dy_med / vn
    
    # Tính angle giữa motion và ref
    dot = float(vx * v0x + vy * v0y)
    dot = max(-1.0, min(1.0, dot))
    cross = float(v0x * vy - v0y * vx)  # > 0 → RIGHT, < 0 → LEFT
    angle = float(np.degrees(np.arctan2(abs(cross), dot)))
    
    # Quyết định hướng
    if angle <= STRAIGHT_ANGLE_DEG:
        return "STRAIGHT"
    elif angle >= TURN_ANGLE_DEG:
        return "RIGHT" if cross > 0 else "LEFT"
    else:
        return "STRAIGHT"  # Vùng mơ hồ → bias STRAIGHT (giảm false positive)


# =============================================================================


def check_violation(
    track_state: TrackState,
    signed_distance: float,
    light_state: TrafficLightState,
    frame_idx: int,
    px: float,
    py: float,
    ref_vector: Optional[Tuple[float, float]] = None,
    debounce_frames: int = 8,
    is_priority_vehicle: bool = False
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Kiểm tra vi phạm với 3 vùng (NEW logic với debounce tách):
      - Vùng -1 (approaching): signed_distance < -STOPLINE_TOUCH_DIST → xe đang tới
      - Vùng  0 (touching):    |signed_distance| <= STOPLINE_TOUCH_DIST → xe chạm vạch
      - Vùng +1 (crossed):     signed_distance > +STOPLINE_TOUCH_DIST → xe vượt qua hoàn toàn
    
    DEBOUNCE LOGIC (NEW):
      - WARNING debounce (DEBOUNCE_WARNING_FRAMES): (-1→0) state transition không toggle liên tục
      - CROSSING debounce (DEBOUNCE_CROSSING_FRAMES): (0→+1) hoặc (-1→+1) state transition
      - Quy tắc KEY: nếu đang WARNING (prev=0, frame_idx - last_warning < DEBOUNCE_WARNING)
        và hiện tại crossing (curr=+1) → BỎ QUA WARNING debounce, kiểm tra CROSSING debounce riêng
        → cho phép ESCALATION: WARNING → VIOLATION ngay cả nếu warning debounce chưa hết
    
    Priority vehicle override:
      - Nếu is_priority_vehicle=True và đèn RED/YELLOW → set label "Priority", bỏ qua WARNING/VIOLATION.
    
    Args:
        track_state: Trạng thái tracking
        signed_distance: Khoảng cách có dấu đến stopline
        light_state: Trạng thái đèn chi tiết
        frame_idx: Frame hiện tại
        px, py: Vị trí hiện tại
        ref_vector: Hướng đi thẳng chuẩn (stopline normal)
        debounce_frames: (deprecated) — sử dụng hằng số global thay vì
        is_priority_vehicle: Xe ưu tiên (ambulance, fire_truck, police_car) — miễn vi phạm đèn đỏ/vàng
    """
    # Xác định vùng 3 mức
    if signed_distance < -STOPLINE_TOUCH_DIST:
        current_region = -1    # Approaching
    elif signed_distance > STOPLINE_TOUCH_DIST:
        current_region = +1    # Fully crossed
    else:
        current_region = 0     # Touching / near stopline
    
    # Cập nhật trajectory
    track_state.positions.append((px, py))
    if len(track_state.positions) > config.MAX_TRACK_HISTORY:
        track_state.positions.pop(0)
    
    # Lần đầu thấy xe
    if not track_state.first_seen:
        track_state.first_seen = True
        track_state.last_region = current_region
        return track_state.label, track_state.color
    
    # === PRIORITY VEHICLE OVERRIDE ===
    # Xe ưu tiên (ambulance, fire_truck, police_car) được miễn vi phạm đèn đỏ/vàng
    if is_priority_vehicle:
        if light_state.has_any_red() or light_state.has_any_yellow():
            track_state.label = "Priority"
            track_state.color = config.COLOR_PRIORITY
            track_state.last_region = current_region
            return track_state.label, track_state.color
        else:
            # Đèn xanh → reset về Safe bình thường
            if track_state.label == "Priority":
                track_state.label = "Safe"
                track_state.color = config.COLOR_SAFE
    
    prev_region = track_state.last_region
    
    # === REGION TRANSITION ANALYSIS ===
    
    # === APPROACHING → TOUCHING (chạm vạch) ===
    if prev_region == -1 and current_region == 0:
        # Check warning debounce: giữ khoảng cách để tránh toggle WARNING liên tục
        if (frame_idx - track_state.last_warning_frame) >= DEBOUNCE_WARNING_FRAMES:
            if light_state.has_any_red() or light_state.has_any_yellow():
                track_state.label = "WARNING"
                track_state.color = config.COLOR_WARNING
                track_state.warned = True
                track_state.last_warning_frame = frame_idx
                track_state.last_event_frame = frame_idx  # Cập nhật cả last_event_frame
    
    # === APPROACHING → FULLY CROSSED (vượt thẳng, bỏ qua touch zone) ===
    # HOẶC: TOUCHING → FULLY CROSSED (đã chạm, giờ vượt hẳn) ===
    # → 2 trường hợp này dùng CROSSING debounce RIÊNG (không phụ thuộc WARNING debounce)
    elif (prev_region == -1 and current_region == +1) or (prev_region == 0 and current_region == +1):
        # Check crossing debounce riêng biệt
        if (frame_idx - track_state.last_crossing_frame) >= DEBOUNCE_CROSSING_FRAMES:
            _check_crossing(track_state, light_state, frame_idx, ref_vector)
            track_state.last_crossing_frame = frame_idx
            track_state.last_event_frame = frame_idx
    
    # === TOUCHING lâu khi đèn xanh → reset warning ===
    elif current_region == 0 and not light_state.has_any_red() and not light_state.has_any_yellow():
        if track_state.label == "WARNING":
            track_state.label = "Safe"
            track_state.color = config.COLOR_SAFE
            track_state.warned = False
    
    # === Corner case: từ crossed về touching (xe đang lùi?) → reset nếu cần ===
    elif current_region == 0 and prev_region == +1:
        # Nếu đã set VIOLATION → giữ nguyên (không revert)
        # Nếu chỉ WARNING → giữ hoặc reset tùy logic
        pass
    
    track_state.last_region = current_region
    return track_state.label, track_state.color


def _check_crossing(
    track_state: TrackState,
    light_state: TrafficLightState,
    frame_idx: int,
    ref_vector: Optional[Tuple[float, float]] = None
):
    """Kiểm tra vi phạm khi xe vượt qua hoàn toàn vạch dừng (CROSSING PHASE)
    
    Quy tắc:
    1. Đèn vàng → WARNING (chưa vi phạm pháp lý, chỉ cảnh báo)
    2. Đèn đỏ + hướng vi phạm → VIOLATION
    3. Đèn xanh → Safe (không vi phạm dù vượt vạch)
    """
    # Đèn vàng → WARNING (thận trọng pháp lý: vàng crossing = cảnh báo, chưa phạt)
    if light_state.has_any_yellow():
        if track_state.label != "VIOLATION":  # Không downgrade từ VIOLATION
            track_state.label = "WARNING"
            track_state.color = config.COLOR_WARNING
        return
    
    # Nếu không có red light nào → safe
    if not light_state.has_any_red():
        track_state.label = "Safe"
        track_state.color = config.COLOR_SAFE
        track_state.warned = False
        return
    
    
    # Phát hiện hướng đi (dùng ref_vector từ stopline normal)
    direction = detect_vehicle_direction(track_state.positions, ref_vector=ref_vector)
    track_state.direction = direction
    
    # Lấy hướng được phép
    allowed = light_state.get_allowed_directions()
    
    # Xét vi phạm
    is_violation = False
    
    if direction == "STRAIGHT" and not allowed['STRAIGHT']:
        is_violation = True
    elif direction == "LEFT" and not allowed['LEFT']:
        is_violation = True
    elif direction == "RIGHT" and not allowed['RIGHT']:
        is_violation = True
    elif direction == "UNKNOWN":
        # Không xác định hướng → kiểm tra đèn đỏ tròn
        if light_state.circle_red:
            # Đèn đỏ tròn + không có mũi tên xanh nào → vi phạm chắc chắn
            if not light_state.right_green and not light_state.left_green and not light_state.straight_green:
                is_violation = True
            # Có mũi tên xanh → benefit of doubt
    
    if is_violation:
        track_state.label = "VIOLATION"
        track_state.color = config.COLOR_VIOLATION
    else:
        track_state.label = "Safe"
        track_state.color = config.COLOR_SAFE
        track_state.warned = False


def dedup_boxes(xyxy: np.ndarray, cls_ids: np.ndarray, confs: np.ndarray, track_ids: np.ndarray):
    """
    Loại bỏ box trùng lặp theo track_id (giữ box có conf cao nhất)
    """
    if track_ids is None or len(track_ids) == 0:
        return xyxy, cls_ids, confs, track_ids
    
    best = {}
    for i, tid in enumerate(track_ids):
        if tid is None or tid < 0:
            continue
        tid = int(tid)
        if tid not in best or confs[i] > confs[best[tid]]:
            best[tid] = i
    
    if not best:
        return xyxy, cls_ids, confs, track_ids
        
    keep_idx = np.array(sorted(best.values()), dtype=int)
    return xyxy[keep_idx], cls_ids[keep_idx], confs[keep_idx], track_ids[keep_idx]


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run(
    video_path: str = None,
    output_path: str = None,
    model_path: str = None,
    show_preview: bool = True
):
    """
    Chạy detection vi phạm vượt đèn đỏ
    
    Args:
        video_path: Đường dẫn video input
        output_path: Đường dẫn video output  
        model_path: Đường dẫn model YOLO
        show_preview: Có hiển thị preview không
    """
    import os
    from datetime import datetime
    from pathlib import Path
    
    video_path = video_path or config.DEFAULT_VIDEO
    model_path = model_path or config.MODEL_PATH
    
    # Tự động tạo output folder nếu chưa có
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo tên file với timestamp để không bị ghi đè
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"redlight_violation_{timestamp}.mp4"
        output_path = str(output_dir / output_filename)
    else:
        # Đảm bảo output_path nằm trong output folder
        output_path = str(output_dir / Path(output_path).name)
    
    print(f"[INFO] Output will be saved to: {output_path}")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer - sử dụng kích thước gốc của video
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)  # Giữ nguyên độ phân giải gốc
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")
    
    # Load model
    print(f"[INIT] Loading model: {model_path}")
    model = YOLO(model_path)
    # Model riêng cho vật thể cố định (đèn, vạch, biển) — predict, không track
    static_model = YOLO(model_path)
    
    # Initialize components
    calibrator = StoplineCalibrator(duration=config.STOPLINE_CALIBRATION_DURATION)
    calibrator.start()
    
    light_memory = LightMemory()
    tracks: Dict[int, TrackState] = {}
    
    # Counters
    violations = 0
    warnings = 0
    frame_idx = 0
    fps_frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    print(f"\n[START] Processing video...")
    print("Press 'q' to quit, 'd' to toggle debug")
    
    debug_on = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        fps_frame_count += 1
        current_time = time.time()
        
        # Cập nhật FPS mỗi giây
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
        
        # === Track: vật thể chuyển động (xe, người) ===
        results = model.track(
            frame,
            persist=True,
            conf=config.CONF_THRESHOLD_VEHICLE,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            tracker=config.TRACKER,
            verbose=False
        )
        r0_track = results[0]
        
        # === Predict: vật thể cố định (đèn, vạch, biển) ===
        static_results = static_model.predict(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD_STOPLINE,
            retina_masks=True,
            verbose=False
        )
        r0_static = static_results[0]
        
        # Parse VEHICLES from track
        if r0_track.boxes is None or len(r0_track.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int32)
            confs = np.zeros((0,), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
        else:
            boxes = r0_track.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = r0_track.boxes.cls.cpu().numpy().astype(np.int32)
            confs = r0_track.boxes.conf.cpu().numpy().astype(np.float32)
            track_ids = r0_track.boxes.id.cpu().numpy().astype(np.int32) if r0_track.boxes.id is not None else np.array([-1] * len(cls_ids))
        
        # Parse STATIC objects from predict
        if r0_static.boxes is None or len(r0_static.boxes) == 0:
            static_cls = np.zeros((0,), dtype=np.int32)
            static_boxes = np.zeros((0, 4), dtype=np.float32)
            static_confs = np.zeros((0,), dtype=np.float32)
        else:
            static_cls = r0_static.boxes.cls.cpu().numpy().astype(np.int32)
            static_boxes = r0_static.boxes.xyxy.cpu().numpy().astype(np.float32)
            static_confs = r0_static.boxes.conf.cpu().numpy().astype(np.float32)
        
        # Traffic light state - từ static predict
        light_state = detect_traffic_lights(static_cls)
        light_memory.update(light_state, current_time)
        light_state = light_memory.get(current_time)
        
        # Stopline detection — merge overlapping + line_inside_mask
        stopline_detections = []
        stopline_mask = np.zeros((h, w), dtype=np.uint8)
        
        if r0_static.masks is not None and r0_static.masks.data is not None:
            stopline_idxs = np.where(np.isin(static_cls, config.STOPLINE_CLASS))[0]
            masks_data = r0_static.masks.data.cpu().numpy()
            
            for idx in stopline_idxs:
                if idx < len(masks_data):
                    mask_raw = masks_data[idx]
                    # Resize to frame resolution if needed
                    if mask_raw.shape[0] != h or mask_raw.shape[1] != w:
                        mask_raw = cv2.resize(mask_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = (mask_raw > 0.5)
                    stopline_detections.append({
                        'box': static_boxes[idx].copy(),
                        'conf': float(static_confs[idx]),
                        'mask': mask_bool
                    })
                    stopline_mask = np.maximum(stopline_mask, mask_bool.astype(np.uint8))
        
        # Merge overlapping detections + find centerline
        merged_dets = merge_overlapping_detections(stopline_detections)
        best_line_this_frame = None
        for det in merged_dets:
            line_info = line_inside_mask(det['mask'])
            if line_info is not None:
                if best_line_this_frame is None or line_info[4] > best_line_this_frame[4]:
                    best_line_this_frame = line_info
        
        # Frame để vẽ
        frame_vis = frame.copy()
        
        # Vẽ traffic light boxes - từ static predict
        draw_traffic_lights(frame_vis, static_boxes, static_cls, static_confs)
        
        # Calibration phase
        if not calibrator.is_calibrated():
            if best_line_this_frame is not None:
                calibrator.update_line(best_line_this_frame)
            calibrator.maybe_finish(h, w)
            
            # Hiển thị mask trong quá trình calibration
            if np.any(stopline_mask > 0):
                overlay = frame_vis.copy()
                overlay[stopline_mask > 0] = [0, 255, 0]
                frame_vis = cv2.addWeighted(overlay, 0.4, frame_vis, 0.6, 0)
                contours, _ = cv2.findContours(stopline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_vis, contours, -1, (0, 255, 0), 2)
            
            # Hiển thị best line tạm thời
            if best_line_this_frame is not None:
                lx1, ly1, lx2, ly2, _ = best_line_this_frame
                cv2.line(frame_vis, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
            
            elapsed = time.time() - calibrator.start_time
            progress = min(elapsed / calibrator.duration * 100, 100)
            remaining = max(0, calibrator.duration - elapsed)
        
        # Vẽ stopline cố định (đã lock)
        if calibrator.is_calibrated() and calibrator.best_line is not None:
            lx1, ly1, lx2, ly2 = calibrator.best_line
            cv2.line(frame_vis, (lx1, ly1), (lx2, ly2), config.COLOR_STOPLINE, 3)
        
        # Dedup boxes
        boxes, cls_ids, confs, track_ids = dedup_boxes(boxes, cls_ids, confs, track_ids)
        
        # Process vehicles
        for xyxy, cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
            # Chỉ xử lý xe
            if cls_id not in config.VEHICLE_CLASSES:
                continue
            if conf < config.CONF_THRESHOLD_VEHICLE:
                continue
            if tid < 0:
                continue
            
            tid = int(tid)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            px, py = get_bottom_center(xyxy)
            
            # Init track state
            if tid not in tracks:
                tracks[tid] = TrackState()
            
            label = tracks[tid].label
            color = tracks[tid].color
            
            # Check violation nếu đã calibrate
            if calibrator.is_calibrated():
                a, b, c = calibrator.line_abc
                signed = get_signed_distance(px, py, a, b, c) * calibrator.sign_flip
                
                # Stopline normal = hướng đi thẳng chuẩn (vuông góc với vạch, hướng xe đang tới)
                # Normal vector = (a, b) * sign_flip (hướng từ vạch về phía camera)
                ref_vec = (a * calibrator.sign_flip, b * calibrator.sign_flip)
                
                label, color = check_violation(
                    tracks[tid], signed, light_state, frame_idx, px, py,
                    ref_vector=ref_vec,
                    is_priority_vehicle=(cls_id in config.PRIORITY_VEHICLE_CLASSES)
                )
                
                # Cập nhật counter
                if label == "VIOLATION" and tracks[tid].last_event_frame == frame_idx:
                    violations += 1
                    direction = tracks[tid].direction
                    # Chụp screenshot ngay khi phát hiện violation (dùng frame gốc)
                    vclass = config.CLASS_NAMES.get(cls_id, "vehicle").lower()
                    save_violation_snapshot(frame, "redlight", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                    print(f"[VIOLATION] Vehicle ID {tid} | Direction: {direction}")
                elif label == "WARNING" and tracks[tid].last_event_frame == frame_idx:
                    warnings += 1
            
            # Vẽ box và label - ngắn gọn
            vehicle_name = config.CLASS_NAMES.get(cls_id, "Vehicle")
            # Chỉ hiện Warning hoặc Violation ngắn gọn
            if label == "VIOLATION":
                display_label = "Violation"
            elif label == "WARNING":
                display_label = "Warning"
            elif label == "Priority":
                display_label = "Priority"
            else:
                display_label = f"{vehicle_name}:{tid}"
            draw_bbox_with_label(frame_vis, (x1, y1, x2, y2), display_label, color)
            cv2.circle(frame_vis, (int(px), int(py)), 4, color, -1)
            
            # Debug overlay - hiển thị thông tin chi tiết
            if debug_on:
                track = tracks[tid]
                # Hiển thị signed distance và state
                if calibrator.is_calibrated():
                    a, b, c = calibrator.line_abc
                    signed = get_signed_distance(px, py, a, b, c) * calibrator.sign_flip
                    debug_text = f"d={signed:.0f} dir={track.direction}"
                    cv2.putText(frame_vis, debug_text, (int(x1), int(y2) + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # HUD - sử dụng draw_utils
        simple_light = light_state.get_simple_state()
        light_color = config.COLOR_VIOLATION if simple_light == "RED" else config.COLOR_WARNING if simple_light == "YELLOW" else config.COLOR_SAFE
        
        hud_lines = [
            (f"Light: {simple_light}", light_color),
            (f"Violations: {violations}", config.COLOR_VIOLATION),
            (f"Warnings: {warnings}", config.COLOR_WARNING),
            (f"FPS: {current_fps:.1f}", config.HUD_TEXT_COLOR),
        ]
        draw_info_hud(frame_vis, hud_lines, title="RED LIGHT DETECTION", title_color=config.COLOR_VIOLATION)
        
        # Output - giữ nguyên độ phân giải gốc (không resize)
        writer.write(frame_vis)
        
        # Preview
        if show_preview:
            cv2.imshow("Red Light Violation Detection", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                debug_on = not debug_on
                print(f"[DEBUG] Debug mode: {'ON' if debug_on else 'OFF'}")
        
        # Progress log
        if frame_idx % 100 == 0:
            print(f"[PROGRESS] Frame {frame_idx} | Violations: {violations} | Warnings: {warnings}")
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total Frames: {frame_idx}")
    print(f"Red Light Violations: {violations}")
    print(f"Yellow Warnings: {warnings}")
    print(f"\nOutput saved: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run()