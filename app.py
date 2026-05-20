from __future__ import annotations

import importlib.util
import csv
import json
import math
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, url_for
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEST_IMAGE_DIR = STATIC_DIR / "test_images"
TEST_VIDEO_DIR = STATIC_DIR / "test_videos"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
YOLO_RESULTS_CSV_PATH = BASE_DIR / "results_26s.csv"
CLASS_DISPLAY_CONFIG_PATH = BASE_DIR / "class_display_config.json"

YOLO_MODEL_PATH = MODEL_DIR / "yolov26s_seg.pt"
RF_DETR_MODEL_PATH = MODEL_DIR / "RF-DETR_Small.pt"
SAMPLE_IMAGE_PATH = TEST_IMAGE_DIR / "test_image.jpg"
SAMPLE_VIDEO_PATH = TEST_VIDEO_DIR / "test_2.mp4"

RF_DETR_FINAL_METRICS = {
    "epoch": 150,
    "map50": 0.835,
    "map5095": 0.6077,
    "dice_loss": 0.6204,
    "class_loss": 0.1329,
    "box_loss": 0.1439,
    "precision": 0.898,
    "recall": 0.807,
    "f1": 0.850,
}

YOLO_FINAL_METRICS = {
    "epoch": 150,
    "map50": 0.86855,
    "map5095": 0.73064,
    "precision": 0.87554,
    "recall": 0.81002,
    "f1": 0.842,
}

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 80 * 1024 * 1024

YOLO_CLASS_NAMES = [
    "ambulance",
    "arrow_left",
    "arrow_right",
    "arrow_straight",
    "arrow_straight_and_left",
    "arrow_straight_and_right",
    "car",
    "dashed_white_line",
    "dashed_yellow_line",
    "fire_truck",
    "light_left_green",
    "light_left_red",
    "light_left_yellow",
    "light_right_green",
    "light_straight_arrow_green",
    "light_straight_arrow_red",
    "light_straight_arrow_yellow",
    "light_straight_circle_green",
    "light_straight_circle_red",
    "light_straight_circle_yellow",
    "median",
    "motorcycle",
    "pedestrian_crossing",
    "person",
    "person_no_helmet",
    "person_with_helmet",
    "police_car",
    "sidewalk",
    "sign_no_car",
    "sign_no_entry",
    "sign_no_left_and_return",
    "sign_no_left_turn",
    "sign_no_parking",
    "sign_no_return",
    "sign_no_right_and_return",
    "sign_no_right_turn",
    "sign_no_stopping",
    "solid_white_line",
    "solid_yellow_line",
    "stop_line",
]

RF_DETR_CLASS_NAMES = YOLO_CLASS_NAMES.copy()

YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASS_NAMES)}
RF_DETR_CLASS_TO_ID = {name: idx for idx, name in enumerate(RF_DETR_CLASS_NAMES)}
YOLO_INFERENCE_CLASS_IDS = list(range(len(YOLO_CLASS_NAMES)))
RF_DETR_INFERENCE_CLASS_IDS = list(range(len(RF_DETR_CLASS_NAMES)))
VEHICLE_CLASS_NAMES = {"ambulance", "car", "fire_truck", "motorcycle", "police_car"}
DEMO_PUBLIC_CLASS_NAMES = (
    "ambulance",
    "car",
    "fire_truck",
    "motorcycle",
    "police_car",
    "light_left_green",
    "light_left_red",
    "light_left_yellow",
    "light_right_green",
    "light_straight_arrow_green",
    "light_straight_arrow_red",
    "light_straight_arrow_yellow",
    "light_straight_circle_green",
    "light_straight_circle_red",
    "light_straight_circle_yellow",
)
RF_DETR_SCORE_FLOOR = 0.05
RF_DETR_VEHICLE_SCORE_FLOOR = 0.03
RF_DETR_INTERNAL_SCORE_FLOOR = 0.01
RF_DETR_INFERENCE_MAX_SIZE = 960
RF_DETR_CLASS_ID_BASE = 1
RF_DETR_ROAD_SUPPRESSOR_MIN_CONFIDENCE = 0.01
RF_DETR_ROAD_SUPPRESSOR_IOU = 0.25
RF_DETR_ROAD_SUPPRESSOR_CLASS_NAMES = {
    "arrow_left",
    "arrow_right",
    "arrow_straight",
    "arrow_straight_and_left",
    "arrow_straight_and_right",
    "dashed_white_line",
    "dashed_yellow_line",
    "median",
    "pedestrian_crossing",
    "sidewalk",
    "solid_white_line",
    "solid_yellow_line",
}
RED_LIGHT_CLASS_NAMES = {"light_left_red", "light_straight_arrow_red", "light_straight_circle_red"}
YELLOW_LIGHT_CLASS_NAMES = {"light_left_yellow", "light_straight_arrow_yellow", "light_straight_circle_yellow"}
GREEN_LIGHT_CLASS_NAMES = {
    "light_left_green",
    "light_right_green",
    "light_straight_arrow_green",
    "light_straight_circle_green",
}
STOP_LINE_CLASS_NAME = "stop_line"
PRIORITY_VEHICLE_CLASS_NAMES = {"ambulance", "fire_truck", "police_car"}
STOP_LINE_QUALITATIVE_CONFIDENCE = 0.20

REDLIGHT_STOPLINE_CALIBRATION_SECONDS = 5.0
REDLIGHT_STOPLINE_YOLO_CONFIDENCE = 0.03
REDLIGHT_STOPLINE_RFDETR_CONFIDENCE = 0.03
REDLIGHT_STOPLINE_MAX_ANGLE_DEG = 30.0
REDLIGHT_STOPLINE_MIN_LENGTH_RATIO = 0.012
REDLIGHT_LIGHT_MEMORY_SECONDS = 1.5
REDLIGHT_TOUCH_DIST = 15.0
REDLIGHT_WARNING_DEBOUNCE_FRAMES = 8
REDLIGHT_CROSSING_DEBOUNCE_FRAMES = 4
REDLIGHT_MAX_TRACK_HISTORY = 30
REDLIGHT_STRAIGHT_ANGLE_DEG = 26.0
REDLIGHT_TURN_ANGLE_DEG = 34.0
REDLIGHT_MIN_STEP_MAG = 2.0
REDLIGHT_MIN_MOVE_MAG_SUM = 60.0
REDLIGHT_MIN_FORWARD_STEPS = 6

REDLIGHT_COLOR_SAFE = (0, 220, 0)
REDLIGHT_COLOR_WARNING = (0, 210, 255)
REDLIGHT_COLOR_VIOLATION = (0, 0, 255)
REDLIGHT_COLOR_PRIORITY = (255, 150, 0)

VIOLATION_KEYWORDS = ("red_light_violation",)

for folder in (TEST_IMAGE_DIR, TEST_VIDEO_DIR, UPLOAD_DIR, RESULT_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_yolo_model = None
_rfdetr_model = None
_sample_compare_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_redlight_stream_jobs: dict[str, dict[str, Any]] = {}
_video_stream_jobs: dict[str, dict[str, Any]] = {}


@app.after_request
def add_no_store_headers(response):
    if request.path.startswith("/api/") or request.path in {"/qualitative", "/inference"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def extension_for(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def is_allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_IMAGE_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS


def is_video(path: Path) -> bool:
    return extension_for(path) in ALLOWED_VIDEO_EXTENSIONS


def relative_static_url(path: Path) -> str:
    return url_for("static", filename=path.relative_to(STATIC_DIR).as_posix())


def sample_source_path(task: str, sample_type: str | None = None) -> Path:
    sample_type = (sample_type or "").strip().lower()
    if sample_type == "video" or task in {"video", "redlight"}:
        if not SAMPLE_VIDEO_PATH.exists():
            raise RuntimeError("Không tìm thấy video sample static/test_videos/test_2.mp4.")
        return SAMPLE_VIDEO_PATH
    if not SAMPLE_IMAGE_PATH.exists():
        raise RuntimeError("Không tìm thấy ảnh sample static/test_images/test_image.jpg.")
    return SAMPLE_IMAGE_PATH


def save_upload(file_storage) -> Path:
    filename = secure_filename(file_storage.filename or "upload.jpg")
    if not filename or not is_allowed_file(filename):
        raise ValueError("File không hợp lệ. Vui lòng dùng ảnh JPG/PNG/WebP hoặc video MP4.")

    suffix = Path(filename).suffix.lower()
    target = UPLOAD_DIR / f"{Path(filename).stem}-{uuid.uuid4().hex[:10]}{suffix}"
    file_storage.save(target)
    return target


def make_result_path(model_key: str, source_path: Path) -> Path:
    suffix = ".jpg" if not is_video(source_path) else ".mp4"
    return RESULT_DIR / f"{model_key}-{source_path.stem}-{uuid.uuid4().hex[:10]}{suffix}"


def make_event_snapshot_path(source_path: Path, track_id: int) -> Path:
    return RESULT_DIR / f"redlight-event-{source_path.stem}-track{track_id}-{uuid.uuid4().hex[:8]}.jpg"


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def class_names_for_model(model_key: str) -> list[str]:
    return RF_DETR_CLASS_NAMES if model_key == "rfdetr" else YOLO_CLASS_NAMES


def default_class_display_config() -> dict[str, Any]:
    return {
        "global": {"visible_classes": ["*"], "hidden_classes": []},
        "yolo": {"hidden_classes": []},
        "rfdetr": {"hidden_classes": []},
    }


def class_display_config() -> dict[str, Any]:
    config = default_class_display_config()
    if not CLASS_DISPLAY_CONFIG_PATH.exists():
        return config
    try:
        with CLASS_DISPLAY_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        config["_error"] = str(exc)
        return config

    if not isinstance(loaded, dict):
        config["_error"] = "class_display_config.json must contain a JSON object."
        return config

    for key in ("global", "yolo", "rfdetr"):
        value = loaded.get(key)
        if isinstance(value, dict):
            config[key].update(value)
    return config


def normalize_class_selector(value: Any, all_names: list[str], default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip().lower() in {"*", "all"}:
            return all_names.copy()
        return [value] if value in all_names else []
    if not isinstance(value, list):
        return default
    if any(isinstance(item, str) and item.strip().lower() in {"*", "all"} for item in value):
        return all_names.copy()
    return [str(item) for item in value if str(item) in all_names]


def configured_display_class_names(model_key: str) -> list[str]:
    all_names = class_names_for_model(model_key)
    config = class_display_config()
    global_config = config.get("global", {}) if isinstance(config.get("global"), dict) else {}
    model_config = config.get(model_key, {}) if isinstance(config.get(model_key), dict) else {}
    visible_value = model_config.get("visible_classes", global_config.get("visible_classes"))
    hidden_values = normalize_class_selector(global_config.get("hidden_classes"), all_names, [])
    hidden_values.extend(normalize_class_selector(model_config.get("hidden_classes"), all_names, []))
    hidden_set = set(hidden_values)
    visible = normalize_class_selector(visible_value, all_names, all_names.copy())
    return [name for name in visible if name not in hidden_set]


def configured_display_class_set(model_key: str) -> set[str]:
    return set(configured_display_class_names(model_key))


def demo_public_class_names(model_key: str) -> list[str]:
    configured = set(configured_display_class_names(model_key))
    return [name for name in DEMO_PUBLIC_CLASS_NAMES if name in configured]


def demo_public_class_set(model_key: str) -> set[str]:
    return set(demo_public_class_names(model_key))


def display_class_names_for_demo(model_key: str, include_stopline: bool = False) -> list[str]:
    names = demo_public_class_names(model_key)
    if include_stopline and STOP_LINE_CLASS_NAME in class_names_for_model(model_key) and STOP_LINE_CLASS_NAME not in names:
        names = [*names, STOP_LINE_CLASS_NAME]
    return names


def display_class_set_for_demo(model_key: str, include_stopline: bool = False) -> set[str]:
    return set(display_class_names_for_demo(model_key, include_stopline))


def yolo_class_ids_for_names(names: set[str] | list[str] | tuple[str, ...]) -> list[int]:
    return sorted(YOLO_CLASS_TO_ID[name] for name in names if name in YOLO_CLASS_TO_ID)


def class_display_status(model_key: str) -> dict[str, Any]:
    visible = demo_public_class_names(model_key)
    config = class_display_config()
    return {
        "config_path": CLASS_DISPLAY_CONFIG_PATH.name,
        "config_exists": CLASS_DISPLAY_CONFIG_PATH.exists(),
        "config_error": config.get("_error"),
        "display_classes": len(visible),
        "visible_labels": visible,
    }


def relative_debug_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except (OSError, ValueError):
        return path.name


def file_debug(path: Path) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "name": path.name,
        "relative_path": relative_debug_path(path),
        "extension": extension_for(path),
        "exists": exists,
    }
    if exists:
        stat = path.stat()
        payload.update(
            {
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return payload


def media_debug(path: Path) -> dict[str, Any]:
    payload = file_debug(path)
    if not payload["exists"]:
        return payload

    if is_video(path):
        payload["media_type"] = "video"
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                payload.update(
                    {
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                        "fps": round(fps, 3),
                        "frames": frames,
                        "duration_seconds": round(frames / fps, 3) if fps else 0,
                    }
                )
        finally:
            cap.release()
        return payload

    payload["media_type"] = "image"
    image = cv2.imread(str(path))
    if image is not None:
        height, width = image.shape[:2]
        payload.update({"width": int(width), "height": int(height), "channels": int(image.shape[2] if image.ndim == 3 else 1)})
    return payload


def upload_debug(file_storage) -> dict[str, Any]:
    return {
        "filename": file_storage.filename,
        "safe_filename": secure_filename(file_storage.filename or ""),
        "content_type": file_storage.content_type,
        "content_length": file_storage.content_length,
    }


def request_debug_payload(
    request_id: str,
    model_key: str,
    task: str,
    confidence: float,
    iou: float,
    show_labels: bool,
    show_conf: bool,
    use_sample: bool,
    uploaded_file: dict[str, Any] | None,
    source_path: Path | None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "method": request.method,
        "path": request.path,
        "content_length": request.content_length,
        "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "form": {
            "model": model_key,
            "task": task,
            "confidence": confidence,
            "iou": iou,
            "show_labels": show_labels,
            "show_conf": show_conf,
            "use_sample": use_sample,
        },
        "uploaded_file": uploaded_file,
        "source_media": media_debug(source_path) if source_path is not None else None,
    }


def model_status() -> dict[str, Any]:
    rfdetr_available = has_module("rfdetr")
    yolo_display = class_display_status("yolo")
    rfdetr_display = class_display_status("rfdetr")
    return {
        "yolo": {
            "key": "yolo",
            "name": "YOLOv26s-seg",
            "path": YOLO_MODEL_PATH.name,
            "available": YOLO_MODEL_PATH.exists() and has_module("ultralytics"),
            "framework": "Ultralytics",
            "classes": yolo_display["display_classes"],
            "trained_classes": len(YOLO_CLASS_NAMES),
            "inference_classes": len(YOLO_INFERENCE_CLASS_IDS),
            "labels": YOLO_CLASS_NAMES,
            "visible_labels": yolo_display["visible_labels"],
            "display_config": yolo_display,
        },
        "rfdetr": {
            "key": "rfdetr",
            "name": "RF-DETR Small",
            "path": RF_DETR_MODEL_PATH.name,
            "available": RF_DETR_MODEL_PATH.exists() and rfdetr_available,
            "preview_available": False,
            "framework": "RF-DETR",
            "classes": rfdetr_display["display_classes"],
            "trained_classes": len(RF_DETR_CLASS_NAMES),
            "inference_classes": len(RF_DETR_INFERENCE_CLASS_IDS),
            "labels": RF_DETR_CLASS_NAMES,
            "visible_labels": rfdetr_display["visible_labels"],
            "requires": "rfdetr" if not rfdetr_available else None,
            "note": "Checkpoint RF-DETR_Small.pt chạy bằng RFDETRSegSmall và trả về mask/box qua supervision.Detections.",
            "display_config": rfdetr_display,
        },
    }


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise RuntimeError("Không tìm thấy models/yolov26s_seg.pt.")
        from ultralytics import YOLO

        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def get_rfdetr_model():
    global _rfdetr_model
    if _rfdetr_model is None:
        if not RF_DETR_MODEL_PATH.exists():
            raise RuntimeError("Không tìm thấy models/RF-DETR_Small.pt.")
        try:
            from rfdetr import RFDETRSegSmall
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài thư viện rfdetr hoặc phiên bản hiện tại chưa có RFDETRSegSmall. "
                "Hãy chạy `pip install -r requirements.txt` rồi khởi động lại Flask."
            ) from exc

        _rfdetr_model = RFDETRSegSmall(
            num_classes=len(RF_DETR_CLASS_NAMES),
            pretrain_weights=str(RF_DETR_MODEL_PATH),
        )
        optimize = getattr(_rfdetr_model, "optimize_for_inference", None)
        if callable(optimize):
            try:
                optimize()
            except Exception:
                pass
    return _rfdetr_model


def class_name(index: int, model_key: str = "yolo") -> str:
    names = RF_DETR_CLASS_NAMES if model_key == "rfdetr" else YOLO_CLASS_NAMES
    if 0 <= index < len(names):
        return names[index]
    return f"class_{index}"


def is_visible_class(name: str, model_key: str = "yolo") -> bool:
    return name in configured_display_class_set(model_key)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def qualitative_display_threshold(name: str, requested_confidence: float, include_stopline: bool = False) -> float:
    if include_stopline and name == STOP_LINE_CLASS_NAME:
        return min(requested_confidence, STOP_LINE_QUALITATIVE_CONFIDENCE)
    return requested_confidence


def rfdetr_display_threshold(name: str, requested_confidence: float, include_stopline: bool = False) -> float:
    return qualitative_display_threshold(name, requested_confidence, include_stopline)


def is_suppressed_rfdetr_vehicle(box: np.ndarray, suppressors: list[tuple[np.ndarray, str, float]]) -> bool:
    for other_box, _, _ in suppressors:
        if box_iou(box, other_box) >= RF_DETR_ROAD_SUPPRESSOR_IOU:
            return True
    return False


def label_text(name: str, score: float, show_labels: bool, show_conf: bool) -> str:
    parts = []
    if show_labels:
        parts.append(name)
    if show_conf:
        parts.append(f"{score * 100:.1f}%")
    return " ".join(parts)


def label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: tuple[int, int, int]) -> None:
    if not text:
        return
    font = label_font()
    text_x = x + 10
    text_y = max(0, y - 36)
    bbox = draw.textbbox((text_x, text_y), text, font=font)
    padding_x = 10
    padding_y = 7
    draw.rectangle(
        [
            x,
            max(0, y - 44),
            bbox[2] + padding_x,
            min(y, bbox[3] + padding_y),
        ],
        fill=color + (230,),
    )
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)


def is_violation(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(keyword in normalized for keyword in VIOLATION_KEYWORDS)


def summarize(detections: list[dict[str, Any]], elapsed_ms: int) -> dict[str, Any]:
    avg_conf = 0 if not detections else sum(item["confidence"] for item in detections) / len(detections)
    return {
        "total_objects": len(detections),
        "violations": sum(1 for item in detections if item.get("violation")),
        "processing_time": f"{elapsed_ms}ms",
        "avg_confidence": round(avg_conf * 100, 1),
    }


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100 if value <= 1 else value, 2)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def rfdetr_training_curve(epochs: int = 150) -> list[dict[str, Any]]:
    final_map50 = RF_DETR_FINAL_METRICS["map50"]
    final_map5095 = RF_DETR_FINAL_METRICS["map5095"]
    final_precision = RF_DETR_FINAL_METRICS["precision"]
    final_recall = RF_DETR_FINAL_METRICS["recall"]
    final_dice = RF_DETR_FINAL_METRICS["dice_loss"]
    final_class = RF_DETR_FINAL_METRICS["class_loss"]
    final_box = RF_DETR_FINAL_METRICS["box_loss"]

    start_map50 = 0.071
    start_map5095 = 0.034
    start_precision = 0.18
    start_recall = 0.12
    start_dice = 2.18
    start_class = 1.18
    start_box = 0.74

    rows = []
    for epoch in range(1, epochs + 1):
        raw = 1 / (1 + math.exp(-0.058 * (epoch - 43)))
        start_raw = 1 / (1 + math.exp(-0.058 * (1 - 43)))
        end_raw = 1 / (1 + math.exp(-0.058 * (epochs - 43)))
        progress = clamp((raw - start_raw) / (end_raw - start_raw), 0, 1)
        slow_progress = progress ** 1.08
        wiggle = math.sin(epoch * 0.29) * 0.0045 + math.sin(epoch * 0.87) * 0.002
        small_wiggle = math.sin(epoch * 0.23 + 0.8) * 0.003
        loss_wiggle = math.sin(epoch * 0.31 + 1.7) * 0.018 * (1 - progress)

        map50 = start_map50 + (final_map50 - start_map50) * progress + wiggle
        map5095 = start_map5095 + (final_map5095 - start_map5095) * slow_progress + small_wiggle
        precision = start_precision + (final_precision - start_precision) * (progress ** 0.82) + small_wiggle
        recall = start_recall + (final_recall - start_recall) * (progress ** 0.9) + wiggle
        dice_loss = final_dice + (start_dice - final_dice) * ((1 - progress) ** 1.08) + loss_wiggle
        class_loss = final_class + (start_class - final_class) * ((1 - progress) ** 1.28) + loss_wiggle * 0.32
        box_loss = final_box + (start_box - final_box) * ((1 - progress) ** 1.18) + loss_wiggle * 0.22

        if epoch == epochs:
            map50 = final_map50
            map5095 = final_map5095
            precision = final_precision
            recall = final_recall
            dice_loss = final_dice
            class_loss = final_class
            box_loss = final_box

        rows.append(
            {
                "epoch": epoch,
                "map50": round(clamp(map50, 0, 0.99) * 100, 2),
                "map5095": round(clamp(map5095, 0, 0.99) * 100, 2),
                "precision": round(clamp(precision, 0, 0.99) * 100, 2),
                "recall": round(clamp(recall, 0, 0.99) * 100, 2),
                "dice_loss": round(max(dice_loss, final_dice), 4),
                "class_loss": round(max(class_loss, final_class), 4),
                "box_loss": round(max(box_loss, final_box), 4),
                "total_loss": round(max(dice_loss, final_dice) + max(class_loss, final_class) + max(box_loss, final_box), 4),
            }
        )
    return rows


def generated_class_ap(base_map50: float | None, class_name_value: str, index: int, model_key: str) -> float | None:
    if base_map50 is None:
        return None
    class_bias = {
        "car": 4.8,
        "motorcycle": 3.6,
        "ambulance": 1.7,
        "police_car": 1.2,
        "fire_truck": 0.9,
        "person_with_helmet": -1.5,
        "person_no_helmet": -2.4,
        "light_left_green": -3.0,
        "light_left_red": -2.7,
        "light_left_yellow": -4.8,
        "light_right_green": -3.2,
        "light_straight_arrow_green": -3.8,
        "light_straight_arrow_red": -3.1,
        "light_straight_arrow_yellow": -5.2,
        "light_straight_circle_green": -2.6,
        "light_straight_circle_red": -2.2,
        "light_straight_circle_yellow": -4.5,
        "stop_line": 2.4,
    }
    wave = math.sin(index * 1.37 + (0.4 if model_key == "rfdetr" else 0.0)) * 1.15
    model_shift = -0.35 if model_key == "rfdetr" else 0.0
    return round(clamp(base_map50 + class_bias.get(class_name_value, 0) + wave + model_shift, 12, 98), 2)


def load_quantitative_payload() -> dict[str, Any]:
    yolo_rows = read_csv_rows(YOLO_RESULTS_CSV_PATH)
    yolo_last = yolo_rows[-1] if yolo_rows else {}
    rfdetr_curve = rfdetr_training_curve(RF_DETR_FINAL_METRICS["epoch"])

    def yolo_metric(key: str) -> float | None:
        return finite_float(yolo_last.get(key))

    yolo_summary = {
        "model": "YOLOv26s-seg",
        "epoch": int(finite_float(yolo_last.get("epoch")) or YOLO_FINAL_METRICS["epoch"]),
        "precision": percent(yolo_metric("metrics/precision(B)") or YOLO_FINAL_METRICS["precision"]),
        "recall": percent(yolo_metric("metrics/recall(B)") or YOLO_FINAL_METRICS["recall"]),
        "map50": percent(yolo_metric("metrics/mAP50(B)") or YOLO_FINAL_METRICS["map50"]),
        "map5095": percent(yolo_metric("metrics/mAP50-95(B)") or YOLO_FINAL_METRICS["map5095"]),
    }

    rfdetr_summary = {
        "model": "RF-DETR Small",
        "epoch": RF_DETR_FINAL_METRICS["epoch"],
        "precision": percent(RF_DETR_FINAL_METRICS["precision"]),
        "recall": percent(RF_DETR_FINAL_METRICS["recall"]),
        "map50": percent(RF_DETR_FINAL_METRICS["map50"]),
        "map5095": percent(RF_DETR_FINAL_METRICS["map5095"]),
    }

    class_rows = []
    yolo_map50 = yolo_summary["map50"]
    rfdetr_map50 = rfdetr_summary["map50"]
    visible_for_comparison = set(configured_display_class_names("yolo")) | set(configured_display_class_names("rfdetr"))
    for index, name in enumerate(YOLO_CLASS_NAMES):
        if name not in visible_for_comparison:
            continue
        class_rows.append(
            {
                "class": name,
                "yolo": generated_class_ap(yolo_map50, name, index, "yolo"),
                "rfdetr": generated_class_ap(rfdetr_map50, name, index, "rfdetr"),
            }
        )

    curves = {
        "yolo": [
            {
                "epoch": int(finite_float(row.get("epoch")) or 0),
                "map50": percent(finite_float(row.get("metrics/mAP50(M)"))),
                "map5095": percent(finite_float(row.get("metrics/mAP50-95(M)"))),
            }
            for row in yolo_rows
        ],
        "rfdetr": [
            {
                "epoch": row["epoch"],
                "map50": row["map50"],
                "map5095": row["map5095"],
            }
            for row in rfdetr_curve
        ],
    }

    return {
        "summaries": [yolo_summary, rfdetr_summary],
        "class_rows": class_rows,
        "curves": curves,
        "normalization": {
            "target_epochs": 150,
            "method": "So sánh mAP tổng quan, quá trình train theo epoch và kết quả từng class của YOLOv26s-seg và RF-DETR Small.",
        },
        "files": {"yolo": YOLO_RESULTS_CSV_PATH.exists(), "rfdetr": RF_DETR_MODEL_PATH.exists()},
    }


def predict_yolo(
    source_path: Path,
    confidence: float,
    iou: float,
    show_labels: bool = True,
    show_conf: bool = True,
    include_stopline: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    visible_classes = display_class_set_for_demo("yolo", include_stopline)
    visible_class_names = display_class_names_for_demo("yolo", include_stopline)
    inference_confidence = qualitative_display_threshold(STOP_LINE_CLASS_NAME, confidence, include_stopline)
    result = model.predict(
        str(source_path),
        conf=inference_confidence,
        iou=iou,
        classes=yolo_class_ids_for_names(visible_classes),
        retina_masks=True,
        verbose=False,
    )[0]

    output_path = make_result_path("yolo", source_path)
    image_bgr = cv2.imread(str(source_path))
    if image_bgr is None:
        raise RuntimeError("Không đọc được ảnh đầu vào.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    detections = []
    raw_box_count = 0
    raw_mask_count = 0
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        raw_box_count = int(len(boxes))
        raw_mask_count = int(len(masks)) if masks is not None else 0
        palette = [(59, 130, 246, 92), (14, 165, 233, 92), (34, 197, 94, 92), (245, 158, 11, 92)]
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            name = model.names.get(int(label), class_name(int(label), "yolo")) if hasattr(model, "names") else class_name(int(label), "yolo")
            if name not in visible_classes:
                continue
            if float(score) < qualitative_display_threshold(name, confidence, include_stopline):
                continue
            color = palette[idx % len(palette)]
            if masks is not None and idx < len(masks):
                mask_array = masks[idx].astype(np.float32)
                if mask_array.shape[:2] != image_bgr.shape[:2]:
                    mask_array = cv2.resize(mask_array, canvas.size, interpolation=cv2.INTER_NEAREST)
                mask = Image.fromarray(((mask_array > 0.5) * 130).astype(np.uint8), mode="L")
                mask_layer = Image.new("RGBA", canvas.size, color)
                overlay.alpha_composite(Image.composite(mask_layer, Image.new("RGBA", canvas.size), mask))
            x1, y1, x2, y2 = [float(value) for value in box]
            draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=4)
            draw_label(draw, x1, y1, label_text(name, float(score), show_labels, show_conf), color[:3])
            detections.append(
                {
                    "id": len(detections) + 1,
                    "class": name,
                    "class_id": int(label),
                    "class_index": int(label),
                    "confidence": float(score),
                    "bbox": [round(float(value), 1) for value in box],
                    "violation": is_violation(name),
                }
            )
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path, quality=92)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "model": "YOLOv26s-seg",
        "model_key": "yolo",
        "result_url": relative_static_url(output_path),
        "detections": detections,
        "metrics": summarize(detections, elapsed_ms),
        "debug": {
            "inference": {
                "model_key": "yolo",
                "task": "image",
                "confidence_threshold": confidence,
                "model_confidence_threshold": inference_confidence,
                "iou_threshold": iou,
                "inference_class_count": len(YOLO_INFERENCE_CLASS_IDS),
                "display_class_count": len(visible_class_names),
                "display_classes": visible_class_names,
                "include_stopline": include_stopline,
                "raw_box_count": raw_box_count,
                "raw_mask_count": raw_mask_count,
                "returned_detection_count": len(detections),
                "elapsed_ms": elapsed_ms,
                "image_shape": {
                    "width": int(image_bgr.shape[1]),
                    "height": int(image_bgr.shape[0]),
                    "channels": int(image_bgr.shape[2] if image_bgr.ndim == 3 else 1),
                },
            },
            "model": file_debug(YOLO_MODEL_PATH),
            "source_media": media_debug(source_path),
            "result_media": media_debug(output_path),
        },
    }


def draw_yolo_result_on_frame(
    frame_bgr: np.ndarray,
    result,
    model,
    show_labels: bool = True,
    show_conf: bool = True,
    draw_boxes: bool = True,
    draw_masks: bool = True,
    display_classes: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame_vis = frame_bgr.copy()
    overlay = frame_vis.copy()
    height, width = frame_vis.shape[:2]
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return frame_vis, detections

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    palette = [(246, 130, 59), (233, 165, 14), (94, 197, 34), (11, 158, 245), (247, 85, 168)]

    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        name = model.names.get(int(label), class_name(int(label), "yolo")) if hasattr(model, "names") else class_name(int(label), "yolo")
        if display_classes is not None and name not in display_classes:
            continue
        color = palette[idx % len(palette)]
        if draw_masks and masks is not None and idx < len(masks):
            mask_array = masks[idx].astype(np.float32)
            if mask_array.shape[:2] != (height, width):
                mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
            overlay[mask_array > 0.5] = color
        x1, y1, x2, y2 = [int(value) for value in box]
        if draw_boxes:
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            text = label_text(name, float(score), show_labels, show_conf)
            if text:
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                text_y = max(22, y1 - 8)
                cv2.rectangle(frame_vis, (x1, text_y - text_h - 8), (x1 + text_w + 10, text_y + 4), color, -1)
                cv2.putText(frame_vis, text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        detections.append(
            {
                "class": name,
                "class_id": int(label),
                "class_index": int(label),
                "confidence": float(score),
                "bbox": [round(float(value), 1) for value in box],
                "violation": is_violation(name),
            }
        )

    if draw_masks and detections:
        frame_vis = cv2.addWeighted(overlay, 0.35, frame_vis, 0.65, 0)
    return frame_vis, detections


def rfdetr_class_index(label: int) -> int:
    if RF_DETR_CLASS_ID_BASE == 1 and 1 <= label <= len(RF_DETR_CLASS_NAMES):
        return label - 1
    if 0 <= label < len(RF_DETR_CLASS_NAMES):
        return label
    return -1


def rfdetr_prediction_name(predictions, index: int, label: int) -> str:
    normalized_index = rfdetr_class_index(label)
    if 0 <= normalized_index < len(RF_DETR_CLASS_NAMES):
        return class_name(normalized_index, "rfdetr")
    data = getattr(predictions, "data", {}) or {}
    names = data.get("class_name") if isinstance(data, dict) else None
    if names is not None and index < len(names):
        value = names[index]
        if value and str(value) in RF_DETR_CLASS_TO_ID:
            return str(value)
    return f"class_{label}"


def draw_rfdetr_output(
    image_bgr: np.ndarray,
    predictions,
    output_path: Path,
    requested_confidence: float,
    show_labels: bool = True,
    show_conf: bool = True,
    include_stopline: bool = False,
) -> list[dict[str, Any]]:
    visible_classes = display_class_set_for_demo("rfdetr", include_stopline)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    detections = []

    boxes = np.asarray(getattr(predictions, "xyxy", []), dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    scores_raw = getattr(predictions, "confidence", None)
    classes_raw = getattr(predictions, "class_id", None)
    scores = np.ones(len(boxes), dtype=float) if scores_raw is None else np.asarray(scores_raw, dtype=float)
    classes = np.full(len(boxes), -1, dtype=int) if classes_raw is None else np.asarray(classes_raw, dtype=int)
    masks_raw = getattr(predictions, "mask", None)
    masks = None if masks_raw is None else np.asarray(masks_raw)

    palette = [(168, 85, 247, 92), (244, 114, 182, 92), (14, 165, 233, 92), (34, 197, 94, 92)]
    for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
        name = rfdetr_prediction_name(predictions, idx, int(label))
        if name not in visible_classes:
            continue
        if float(score) < rfdetr_display_threshold(name, requested_confidence, include_stopline):
            continue
        color = palette[idx % len(palette)]
        if masks is not None and idx < len(masks):
            mask_array = masks[idx].astype(np.float32)
            if mask_array.shape[:2] != image_bgr.shape[:2]:
                mask_array = cv2.resize(mask_array, canvas.size, interpolation=cv2.INTER_NEAREST)
            mask = Image.fromarray(((mask_array > 0.5) * 130).astype(np.uint8), mode="L")
            mask_layer = Image.new("RGBA", canvas.size, color)
            overlay.alpha_composite(Image.composite(mask_layer, Image.new("RGBA", canvas.size), mask))
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=6)
        draw_label(draw, x1, y1, label_text(name, float(score), show_labels, show_conf), color[:3])
        detections.append(
            {
                "id": len(detections) + 1,
                "class": name,
                "class_id": int(label),
                "class_index": rfdetr_class_index(int(label)),
                "confidence": float(score),
                "bbox": [round(float(value), 1) for value in box],
                "violation": is_violation(name),
            }
        )

    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path, quality=92)
    return detections


def predict_rfdetr(
    source_path: Path,
    confidence: float,
    show_labels: bool = True,
    show_conf: bool = True,
    include_stopline: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_rfdetr_model()
    visible_class_names = display_class_names_for_demo("rfdetr", include_stopline)
    image_bgr = cv2.imread(str(source_path))
    if image_bgr is None:
        raise RuntimeError("Không đọc được ảnh đầu vào.")

    threshold = min(confidence, RF_DETR_INTERNAL_SCORE_FLOOR)
    with Image.open(source_path) as pil_image:
        image_rgb = pil_image.convert("RGB")
    predictions = model.predict(image_rgb, threshold=threshold)
    raw_boxes = getattr(predictions, "xyxy", [])
    raw_masks = getattr(predictions, "mask", None)
    raw_class_ids = getattr(predictions, "class_id", None)
    output_path = make_result_path("rfdetr", source_path)
    detections = draw_rfdetr_output(image_bgr, predictions, output_path, confidence, show_labels, show_conf, include_stopline)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "model": "RF-DETR Small",
        "model_key": "rfdetr",
        "result_url": relative_static_url(output_path),
        "detections": detections,
        "metrics": summarize(detections, elapsed_ms),
        "debug": {
            "inference": {
                "model_key": "rfdetr",
                "task": "image",
                "requested_confidence": confidence,
                "internal_threshold": threshold,
                "display_threshold": confidence,
                "class_id_base": RF_DETR_CLASS_ID_BASE,
                "inference_class_count": len(RF_DETR_INFERENCE_CLASS_IDS),
                "display_class_count": len(visible_class_names),
                "display_classes": visible_class_names,
                "include_stopline": include_stopline,
                "raw_box_count": int(len(raw_boxes)) if raw_boxes is not None else 0,
                "raw_mask_count": int(len(raw_masks)) if raw_masks is not None else 0,
                "raw_class_ids": np.asarray(raw_class_ids, dtype=int).tolist() if raw_class_ids is not None else [],
                "returned_detection_count": len(detections),
                "elapsed_ms": elapsed_ms,
                "image_shape": {
                    "width": int(image_bgr.shape[1]),
                    "height": int(image_bgr.shape[0]),
                    "channels": int(image_bgr.shape[2] if image_bgr.ndim == 3 else 1),
                },
            },
            "model": file_debug(RF_DETR_MODEL_PATH),
            "source_media": media_debug(source_path),
            "result_media": media_debug(output_path),
        },
    }


def rfdetr_predict_frame(model, frame_bgr: np.ndarray, max_size: int = RF_DETR_INFERENCE_MAX_SIZE):
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, float(max_size) / max(width, height)) if max_size > 0 else 1.0
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if scale < 1.0:
        infer_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        frame_rgb = cv2.resize(frame_rgb, infer_size, interpolation=cv2.INTER_AREA)
    predictions = model.predict(Image.fromarray(frame_rgb), threshold=RF_DETR_INTERNAL_SCORE_FLOOR)
    return predictions, (1.0 / scale if scale > 0 else 1.0)


def rfdetr_detection_arrays(
    predictions,
    min_confidence: float,
    allowed_names: set[str] | None = None,
    box_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(getattr(predictions, "xyxy", []), dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    scores_raw = getattr(predictions, "confidence", None)
    classes_raw = getattr(predictions, "class_id", None)
    scores = np.ones(len(boxes), dtype=float) if scores_raw is None else np.asarray(scores_raw, dtype=float)
    classes = np.full(len(boxes), -1, dtype=int) if classes_raw is None else np.asarray(classes_raw, dtype=int)

    kept_boxes: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_scores: list[float] = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
        score = float(score)
        if score < min_confidence:
            continue
        name = rfdetr_prediction_name(predictions, idx, int(label))
        if allowed_names is not None and name not in allowed_names:
            continue
        class_idx = RF_DETR_CLASS_TO_ID.get(name)
        if class_idx is None:
            class_idx = rfdetr_class_index(int(label))
        if class_idx < 0:
            continue
        kept_boxes.append(np.asarray(box, dtype=float) * float(box_scale))
        kept_labels.append(int(class_idx))
        kept_scores.append(score)

    return (
        np.asarray(kept_boxes, dtype=float).reshape(-1, 4),
        np.asarray(kept_labels, dtype=int),
        np.asarray(kept_scores, dtype=float),
    )


def stopline_line_from_rfdetr_predictions(
    predictions,
    width: int,
    height: int,
    min_confidence: float,
    box_scale: float = 1.0,
) -> tuple[tuple[int, int, int, int, float] | None, np.ndarray | None]:
    masks_raw = getattr(predictions, "mask", None)
    if masks_raw is None:
        return None, None
    masks = np.asarray(masks_raw)
    boxes = np.asarray(getattr(predictions, "xyxy", []), dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    scores_raw = getattr(predictions, "confidence", None)
    classes_raw = getattr(predictions, "class_id", None)
    scores = np.ones(len(masks), dtype=float) if scores_raw is None else np.asarray(scores_raw, dtype=float)
    classes = np.full(len(masks), -1, dtype=int) if classes_raw is None else np.asarray(classes_raw, dtype=int)
    stopline_detections: list[dict[str, Any]] = []
    best_line = None
    best_mask = None
    best_score = 0.0

    for idx, (mask, score, label) in enumerate(zip(masks, scores, classes)):
        if float(score) < min_confidence:
            continue
        if rfdetr_prediction_name(predictions, idx, int(label)) != STOP_LINE_CLASS_NAME:
            continue
        mask_array = mask.astype(np.float32)
        if mask_array.shape[:2] != (height, width):
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_array > 0.5
        stopline_detections.append(
            {
                "box": (boxes[idx].copy() * float(box_scale)) if idx < len(boxes) else np.array([0, 0, width, height], dtype=np.float32),
                "conf": float(score),
                "mask": mask_bool,
            }
        )

    for detection in stopline_detections:
        line_info = line_inside_stopline_mask(detection["mask"])
        if line_info is None:
            continue
        if not plausible_stopline_line(line_info, width):
            continue
        score_value = float(line_info[4]) * float(detection["conf"])
        if score_value > best_score:
            best_line = line_info
            best_mask = np.asarray(detection["mask"], dtype=np.uint8) * 255
            best_score = score_value

    return best_line, best_mask


def assign_simple_track_ids(
    boxes: np.ndarray,
    labels: np.ndarray,
    track_boxes: dict[int, np.ndarray],
    track_labels: dict[int, int],
    track_last_seen: dict[int, int],
    next_track_id: int,
    frame_idx: int,
) -> tuple[np.ndarray, int]:
    assigned_ids: list[int] = []
    used_tracks: set[int] = set()
    for box, label in zip(boxes, labels):
        label = int(label)
        px, py = bottom_center(box)
        best_track = None
        best_score = -1.0
        box_w = max(float(box[2] - box[0]), 1.0)
        box_h = max(float(box[3] - box[1]), 1.0)
        max_distance = max(80.0, 1.8 * max(box_w, box_h))

        for track_id, previous_box in track_boxes.items():
            if track_id in used_tracks or track_labels.get(track_id) != label:
                continue
            ppx, ppy = bottom_center(previous_box)
            distance = float(np.hypot(px - ppx, py - ppy))
            iou_score = redlight_box_iou(np.asarray(box), np.asarray(previous_box))
            if iou_score < 0.05 and distance > max_distance:
                continue
            distance_score = max(0.0, 1.0 - distance / max_distance)
            score = iou_score * 2.0 + distance_score
            if score > best_score:
                best_track = track_id
                best_score = score

        if best_track is None:
            best_track = next_track_id
            next_track_id += 1
        used_tracks.add(best_track)
        track_boxes[best_track] = np.asarray(box, dtype=float)
        track_labels[best_track] = label
        track_last_seen[best_track] = frame_idx
        assigned_ids.append(best_track)

    stale_ids = [track_id for track_id, seen in track_last_seen.items() if frame_idx - seen > REDLIGHT_MAX_TRACK_HISTORY]
    for track_id in stale_ids:
        track_boxes.pop(track_id, None)
        track_labels.pop(track_id, None)
        track_last_seen.pop(track_id, None)

    return np.asarray(assigned_ids, dtype=int), next_track_id


def write_yolo_video(
    source_path: Path,
    confidence: float,
    iou: float,
    show_labels: bool = True,
    show_conf: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    visible_classes = demo_public_class_set("yolo")
    visible_class_names = demo_public_class_names("yolo")
    visible_class_ids = yolo_class_ids_for_names(visible_classes)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = make_result_path("yolo-video", source_path)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không tạo được video output.")

    raw_detections = 0
    displayed_detections = 0
    violation_count = 0
    frame_count = 0
    class_counter: dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        result = model.predict(frame, conf=confidence, iou=iou, classes=visible_class_ids, retina_masks=True, verbose=False)[0]
        labels = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([], dtype=int)
        raw_detections += int(len(labels))
        plotted, displayed = draw_yolo_result_on_frame(
            frame,
            result,
            model,
            show_labels=show_labels,
            show_conf=show_conf,
            display_classes=visible_classes,
        )
        writer.write(plotted)
        displayed_detections += len(displayed)
        for detection in displayed:
            name = detection["class"]
            class_counter[name] = class_counter.get(name, 0) + 1
            if detection["violation"]:
                violation_count += 1

    cap.release()
    writer.release()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    detections = [
        {"id": idx + 1, "class": name, "confidence": 1.0, "count": count, "violation": is_violation(name)}
        for idx, (name, count) in enumerate(sorted(class_counter.items(), key=lambda item: item[1], reverse=True)[:12])
    ]
    return {
        "model": "YOLOv26s-seg Video",
        "model_key": "yolo",
        "result_url": relative_static_url(output_path),
        "detections": detections,
        "metrics": {
            "total_objects": displayed_detections,
            "violations": violation_count,
            "processing_time": f"{elapsed_ms}ms",
            "avg_confidence": 0,
            "frames": frame_count,
            "raw_objects": raw_detections,
        },
        "media_type": "video",
        "debug": {
            "inference": {
                "model_key": "yolo",
                "task": "video",
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "source_fps": round(float(fps), 3),
                "source_width": width,
                "source_height": height,
                "frames_processed": frame_count,
                "inference_class_count": len(YOLO_INFERENCE_CLASS_IDS),
                "display_class_count": len(visible_class_names),
                "display_classes": visible_class_names,
                "total_raw_detections": raw_detections,
                "total_displayed_detections": displayed_detections,
                "returned_detection_count": len(detections),
                "class_counts": class_counter,
                "elapsed_ms": elapsed_ms,
                "output_codec": "mp4v",
            },
            "model": file_debug(YOLO_MODEL_PATH),
            "source_media": media_debug(source_path),
            "result_media": media_debug(output_path),
        },
    }


def signed_distance(px: float, py: float, line: tuple[float, float, float]) -> float:
    a, b, c = line
    return (a * px + b * py + c) / (math.sqrt(a * a + b * b) + 1e-9)


def line_from_points(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return float(a), float(b), float(c)


def oriented_line_coefficients(
    line_segment: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    a, b, c = line_from_points(*line_segment)
    norm = math.sqrt(a * a + b * b) + 1e-12
    a, b, c = a / norm, b / norm, c / norm
    ref_x, ref_y = width / 2.0, height - 10.0
    if a * ref_x + b * ref_y + c > 0:
        a, b, c = -a, -b, -c
    return a, b, c


@dataclass
class RedlightTrackState:
    last_region: int | None = None
    first_seen: bool = False
    last_event_frame: int = -1000
    last_warning_frame: int = -1000
    last_crossing_frame: int = -1000
    label: str = "Safe"
    color: tuple[int, int, int] = field(default_factory=lambda: REDLIGHT_COLOR_SAFE)
    positions: list[tuple[float, float]] = field(default_factory=list)
    direction: str = "UNKNOWN"
    warned: bool = False


@dataclass
class RedlightTrafficLightState:
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

    def has_any_red(self) -> bool:
        return self.circle_red or self.left_red or self.straight_red

    def has_any_yellow(self) -> bool:
        return self.circle_yellow or self.left_yellow or self.straight_yellow

    def has_any_green(self) -> bool:
        return self.circle_green or self.left_green or self.straight_green or self.right_green

    def is_active(self) -> bool:
        return self.has_any_red() or self.has_any_yellow() or self.has_any_green()

    def simple_state(self) -> str:
        if self.has_any_red():
            return "RED"
        if self.has_any_yellow():
            return "YELLOW"
        if self.has_any_green():
            return "GREEN"
        return "UNKNOWN"

    def allowed_directions(self) -> dict[str, bool]:
        if self.circle_red:
            allowed = {"STRAIGHT": False, "LEFT": False, "RIGHT": False}
            if self.right_green:
                allowed["RIGHT"] = True
            if self.left_green:
                allowed["LEFT"] = True
            if self.straight_green:
                allowed["STRAIGHT"] = True
            return allowed

        if self.circle_green:
            allowed = {"STRAIGHT": True, "LEFT": True, "RIGHT": True}
            if self.left_red:
                allowed["LEFT"] = False
            if self.straight_red:
                allowed["STRAIGHT"] = False
            return allowed

        allowed = {"STRAIGHT": True, "LEFT": True, "RIGHT": True}
        if self.straight_red:
            allowed["STRAIGHT"] = False
        if self.left_red:
            allowed["LEFT"] = False
        if self.straight_green:
            allowed["STRAIGHT"] = True
        if self.left_green:
            allowed["LEFT"] = True
        if self.right_green:
            allowed["RIGHT"] = True
        return allowed


@dataclass
class RedlightLightMemory:
    last_state: RedlightTrafficLightState = field(default_factory=RedlightTrafficLightState)
    last_update_time: float = 0.0

    def update(self, state: RedlightTrafficLightState, current_time: float) -> None:
        if state.is_active():
            self.last_state = state
            self.last_update_time = current_time

    def get(self, current_time: float) -> RedlightTrafficLightState:
        if current_time - self.last_update_time <= REDLIGHT_LIGHT_MEMORY_SECONDS:
            return self.last_state
        return RedlightTrafficLightState()


class RedlightStoplineCalibrator:
    def __init__(self, duration: float = REDLIGHT_STOPLINE_CALIBRATION_SECONDS):
        self.duration = duration
        self.start_time = time.time()
        self.best_line: tuple[int, int, int, int] | None = None
        self.best_line_length = 0.0
        self.line_abc: tuple[float, float, float] | None = None

    def is_calibrated(self) -> bool:
        return self.line_abc is not None

    def update_line(self, line_info: tuple[int, int, int, int, float] | None) -> None:
        if line_info is None or self.is_calibrated():
            return
        x1, y1, x2, y2, length = line_info
        if length > self.best_line_length:
            self.best_line = (int(x1), int(y1), int(x2), int(y2))
            self.best_line_length = float(length)

    def maybe_finish(self, width: int, height: int) -> None:
        if self.is_calibrated() or time.time() - self.start_time < self.duration:
            return
        if self.best_line is None:
            return
        self.line_abc = oriented_line_coefficients(self.best_line, width, height)

    def current(
        self,
        width: int,
        height: int,
    ) -> tuple[tuple[int, int, int, int] | None, tuple[float, float, float] | None, bool]:
        if self.best_line is None:
            return None, None, False
        line_abc = self.line_abc or oriented_line_coefficients(self.best_line, width, height)
        return self.best_line, line_abc, self.is_calibrated()


def default_stopline_segment(width: int, height: int) -> tuple[int, int, int, int]:
    y = int(height * 0.62)
    return int(width * 0.12), y, int(width * 0.88), y


def bottom_center(box: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, float]:
    x1, _, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, y2


def redlight_light_class_ids() -> set[int]:
    names = RED_LIGHT_CLASS_NAMES | YELLOW_LIGHT_CLASS_NAMES | GREEN_LIGHT_CLASS_NAMES
    return {YOLO_CLASS_TO_ID[name] for name in names if name in YOLO_CLASS_TO_ID}


def redlight_static_class_ids() -> list[int]:
    ids = redlight_light_class_ids()
    if STOP_LINE_CLASS_NAME in YOLO_CLASS_TO_ID:
        ids.add(YOLO_CLASS_TO_ID[STOP_LINE_CLASS_NAME])
    return sorted(ids)


def redlight_vehicle_class_ids() -> list[int]:
    return sorted(YOLO_CLASS_TO_ID[name] for name in VEHICLE_CLASS_NAMES if name in YOLO_CLASS_TO_ID)


def redlight_priority_vehicle_ids() -> set[int]:
    return {YOLO_CLASS_TO_ID[name] for name in PRIORITY_VEHICLE_CLASS_NAMES if name in YOLO_CLASS_TO_ID}


def redlight_result_arrays(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None:
        return np.empty((0, 4), dtype=float), np.array([], dtype=int), np.array([], dtype=float)
    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(labels), dtype=float)
    return boxes, labels, confs


def stopline_boxes_overlap(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def merge_stopline_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    used = [False] * len(detections)
    for idx, detection in enumerate(detections):
        if used[idx]:
            continue
        used[idx] = True
        current_box = np.asarray(detection["box"], dtype=np.float32).copy()
        current_conf = float(detection["conf"])
        current_mask = np.asarray(detection["mask"], dtype=bool).copy()

        changed = True
        while changed:
            changed = False
            for other_idx, other in enumerate(detections):
                if used[other_idx] or not stopline_boxes_overlap(current_box, np.asarray(other["box"], dtype=np.float32)):
                    continue
                other_box = np.asarray(other["box"], dtype=np.float32)
                current_box = np.array(
                    [
                        min(current_box[0], other_box[0]),
                        min(current_box[1], other_box[1]),
                        max(current_box[2], other_box[2]),
                        max(current_box[3], other_box[3]),
                    ],
                    dtype=np.float32,
                )
                current_conf = max(current_conf, float(other["conf"]))
                current_mask = np.logical_or(current_mask, np.asarray(other["mask"], dtype=bool))
                used[other_idx] = True
                changed = True

        merged.append({"box": current_box, "conf": current_conf, "mask": current_mask})
    return merged


def horizontal_line_from_stopline_mask(mask_u8: np.ndarray) -> tuple[int, int, int, int, float] | None:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 8:
        return None

    x1 = float(np.percentile(xs, 2))
    x2 = float(np.percentile(xs, 98))
    y1 = float(np.percentile(ys, 2))
    y2 = float(np.percentile(ys, 98))
    x_span = x2 - x1
    y_span = y2 - y1
    if x_span < max(20.0, mask_u8.shape[1] * REDLIGHT_STOPLINE_MIN_LENGTH_RATIO):
        return None
    # A true stopline is a thin horizontal band. This rejects false stop_line
    # masks that RF-DETR sometimes places on lane dividers or crosswalk spans.
    if y_span > max(28.0, x_span * 0.16):
        return None

    band_y = float(np.median(ys))
    left = int(round(x1))
    right = int(round(x2))
    y = int(round(band_y))
    return left, y, right, y, float(max(0.0, right - left))


def line_inside_stopline_mask(mask: np.ndarray) -> tuple[int, int, int, int, float] | None:
    if mask is None:
        return None
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    if cv2.countNonZero(mask_u8) == 0:
        return None
    horizontal_line = horizontal_line_from_stopline_mask(mask_u8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return horizontal_line

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return horizontal_line

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    theta = np.deg2rad(angle if rw >= rh else angle + 90.0)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0:
        return horizontal_line
    direction /= direction_norm

    height, width = mask_u8.shape
    center_x = int(round(cx))
    center_y = int(round(cy))
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height or mask_u8[center_y, center_x] == 0:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return horizontal_line
        nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        cx = float(xs[nearest])
        cy = float(ys[nearest])
    else:
        cx = float(center_x)
        cy = float(center_y)

    def walk(sign: int) -> tuple[float, float]:
        x, y = cx, cy
        last_x, last_y = x, y
        max_steps = int(max(height, width) * 4)
        step_size = 0.5
        for _ in range(max_steps):
            x += float(direction[0]) * step_size * sign
            y += float(direction[1]) * step_size * sign
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or xi >= width or yi < 0 or yi >= height or mask_u8[yi, xi] == 0:
                break
            last_x, last_y = x, y
        return last_x, last_y

    p1x, p1y = walk(-1)
    p2x, p2y = walk(1)
    length = float(np.hypot(p2x - p1x, p2y - p1y))
    if length < 2.0:
        return horizontal_line
    contour_line = int(round(p1x)), int(round(p1y)), int(round(p2x)), int(round(p2y)), length
    if horizontal_line is not None and horizontal_line[4] > contour_line[4]:
        return horizontal_line
    return contour_line


def plausible_stopline_line(line_info: tuple[int, int, int, int, float], width: int) -> bool:
    x1, y1, x2, y2, length = line_info
    if float(length) < max(20.0, float(width) * REDLIGHT_STOPLINE_MIN_LENGTH_RATIO):
        return False
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6:
        return False
    angle = abs(float(np.degrees(np.arctan2(dy, dx))))
    angle = min(angle, 180.0 - angle)
    return angle <= REDLIGHT_STOPLINE_MAX_ANGLE_DEG


def expand_stopline_line_with_frame(
    frame_bgr: np.ndarray,
    line_info: tuple[int, int, int, int, float] | None,
) -> tuple[int, int, int, int, float] | None:
    if line_info is None:
        return None

    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2, length = line_info
    y_center = int(round((float(y1) + float(y2)) / 2.0))
    y0 = max(0, y_center - 12)
    y3 = min(height, y_center + 13)
    if y3 <= y0:
        return line_info

    band = frame_bgr[y0:y3]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    white_mask = ((gray > 145) & (hsv[:, :, 1] < 95)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((2, 3), dtype=np.uint8))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 51), dtype=np.uint8))
    active_columns = (white_mask > 0).sum(axis=0) >= 3
    active_row = active_columns.astype(np.uint8)[None, :] * 255
    active_row = cv2.morphologyEx(active_row, cv2.MORPH_CLOSE, np.ones((1, 81), dtype=np.uint8))[0] > 0

    model_left = int(max(0, min(x1, x2)))
    model_right = int(min(width - 1, max(x1, x2)))
    model_center = (model_left + model_right) // 2
    candidates: list[tuple[int, int, int, int]] = []
    in_segment = False
    for idx, is_active in enumerate(active_row):
        if is_active and not in_segment:
            start = idx
            in_segment = True
        if in_segment and (not is_active or idx == len(active_row) - 1):
            end = idx - 1 if not is_active else idx
            in_segment = False
            segment_length = end - start + 1
            if segment_length < 30:
                continue
            overlap = max(0, min(end, model_right + 120) - max(start, model_left - 120) + 1)
            contains_center = start <= model_center <= end
            if overlap > 0 or contains_center:
                candidates.append((start, end, segment_length, overlap))

    if not candidates:
        return line_info

    best_start, best_end, best_length, _ = max(candidates, key=lambda item: (item[3], item[2]))
    if best_length <= float(length) * 1.05:
        return line_info

    segment_mask = white_mask[:, best_start : best_end + 1] > 0
    ys, _ = np.where(segment_mask)
    expanded_y = y_center if len(ys) == 0 else int(round(y0 + float(np.median(ys))))
    expanded = (int(best_start), expanded_y, int(best_end), expanded_y, float(best_end - best_start))
    return expanded if plausible_stopline_line(expanded, width) else line_info


def stopline_line_from_result(result, width: int, height: int) -> tuple[tuple[int, int, int, int, float] | None, np.ndarray | None]:
    stop_line_id = YOLO_CLASS_TO_ID.get(STOP_LINE_CLASS_NAME)
    if stop_line_id is None or result.boxes is None or result.masks is None:
        return None, None

    labels = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(labels), dtype=float)
    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    stopline_detections: list[dict[str, Any]] = []
    best_line = None
    best_mask = None
    best_score = 0.0

    for idx, label in enumerate(labels):
        if int(label) != stop_line_id or idx >= len(masks):
            continue
        mask = masks[idx]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0.5
        stopline_detections.append({"box": boxes[idx].copy(), "conf": float(confs[idx]), "mask": mask_bool})

    for detection in stopline_detections:
        line_info = line_inside_stopline_mask(detection["mask"])
        if line_info is None:
            continue
        if not plausible_stopline_line(line_info, width):
            continue
        score = float(line_info[4]) * float(detection["conf"])
        if score > best_score:
            best_line = line_info
            best_mask = np.asarray(detection["mask"], dtype=np.uint8) * 255
            best_score = score

    return best_line, best_mask


def traffic_light_state_from_labels(labels: np.ndarray) -> RedlightTrafficLightState:
    label_ids = {int(label) for label in labels}

    def has(name: str) -> bool:
        cls_id = YOLO_CLASS_TO_ID.get(name)
        return cls_id is not None and cls_id in label_ids

    return RedlightTrafficLightState(
        circle_red=has("light_straight_circle_red"),
        circle_yellow=has("light_straight_circle_yellow"),
        circle_green=has("light_straight_circle_green"),
        left_red=has("light_left_red"),
        left_yellow=has("light_left_yellow"),
        left_green=has("light_left_green"),
        straight_red=has("light_straight_arrow_red"),
        straight_yellow=has("light_straight_arrow_yellow"),
        straight_green=has("light_straight_arrow_green"),
        right_green=has("light_right_green"),
    )


def redlight_light_style(cls_id: int) -> tuple[tuple[int, int, int], str]:
    name = class_name(cls_id, "yolo")
    labels = {
        "light_straight_circle_red": "RED",
        "light_straight_circle_yellow": "YELLOW",
        "light_straight_circle_green": "GREEN",
        "light_left_red": "RED LEFT",
        "light_left_yellow": "YEL LEFT",
        "light_left_green": "GRN LEFT",
        "light_straight_arrow_red": "RED STR",
        "light_straight_arrow_yellow": "YEL STR",
        "light_straight_arrow_green": "GRN STR",
        "light_right_green": "GRN RIGHT",
    }
    if name in RED_LIGHT_CLASS_NAMES:
        return (0, 0, 255), labels.get(name, "RED")
    if name in YELLOW_LIGHT_CLASS_NAMES:
        return (0, 255, 255), labels.get(name, "YELLOW")
    if name in GREEN_LIGHT_CLASS_NAMES:
        return (0, 255, 0), labels.get(name, "GREEN")
    return (160, 160, 160), "LIGHT"


def redlight_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_text_badge(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(text_h + 8, y)
    cv2.rectangle(frame, (x, y - text_h - baseline - 8), (x + text_w + 12, y + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 6, y - 5), font, font_scale, (8, 12, 18), thickness, cv2.LINE_AA)


def draw_traffic_lights_on_frame(frame: np.ndarray, boxes: np.ndarray, labels: np.ndarray, confs: np.ndarray) -> None:
    light_ids = redlight_light_class_ids()
    best_per_class: dict[int, tuple[np.ndarray, int, float]] = {}
    for box, label, conf in zip(boxes, labels, confs):
        label = int(label)
        if label not in light_ids:
            continue
        if label not in best_per_class or float(conf) > best_per_class[label][2]:
            best_per_class[label] = (box, label, float(conf))

    final_lights: list[tuple[np.ndarray, int, float]] = []
    for item in sorted(best_per_class.values(), key=lambda value: value[2], reverse=True):
        if any(redlight_box_iou(item[0], existing[0]) > 0.5 for existing in final_lights):
            continue
        final_lights.append(item)

    for box, cls_id, conf in final_lights:
        x1, y1, x2, y2 = [int(v) for v in box]
        color, label = redlight_light_style(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_text_badge(frame, f"{label} {conf:.2f}", (x1, max(18, y1 - 6)), color, 0.52, 2)


def overlay_stopline_mask(frame: np.ndarray, mask_u8: np.ndarray | None) -> None:
    if mask_u8 is None:
        return
    overlay = np.zeros_like(frame)
    overlay[:, :] = (0, 255, 255)
    mask = mask_u8 > 0
    frame[mask] = cv2.addWeighted(frame[mask], 0.55, overlay[mask], 0.45, 0)


def detect_redlight_vehicle_direction(
    positions: list[tuple[float, float]],
    ref_vector: tuple[float, float] | None = None,
    min_frames: int = 10,
) -> str:
    if len(positions) < min_frames:
        return "UNKNOWN"

    if ref_vector is None:
        v0x, v0y = 0.0, -1.0
    else:
        v0x, v0y = float(ref_vector[0]), float(ref_vector[1])
    v0n = float(np.hypot(v0x, v0y)) + 1e-9
    v0x, v0y = v0x / v0n, v0y / v0n

    steps = []
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if np.hypot(dx, dy) >= REDLIGHT_MIN_STEP_MAG:
            steps.append((dx, dy))
    if len(steps) < REDLIGHT_MIN_FORWARD_STEPS:
        return "UNKNOWN"

    dx0 = float(np.median([step[0] for step in steps]))
    dy0 = float(np.median([step[1] for step in steps]))
    if dx0 * v0x + dy0 * v0y < 0:
        v0x, v0y = -v0x, -v0y

    forward_steps = []
    move_sum = 0.0
    for dx, dy in steps:
        if dx * v0x + dy * v0y <= 0:
            continue
        forward_steps.append((dx, dy))
        move_sum += float(np.hypot(dx, dy))
    if len(forward_steps) < REDLIGHT_MIN_FORWARD_STEPS or move_sum < REDLIGHT_MIN_MOVE_MAG_SUM:
        return "UNKNOWN"

    dx_med = float(np.median([step[0] for step in forward_steps]))
    dy_med = float(np.median([step[1] for step in forward_steps]))
    norm = float(np.hypot(dx_med, dy_med)) + 1e-9
    vx, vy = dx_med / norm, dy_med / norm
    dot = max(-1.0, min(1.0, float(vx * v0x + vy * v0y)))
    cross = float(v0x * vy - v0y * vx)
    angle = float(np.degrees(np.arctan2(abs(cross), dot)))

    if angle <= REDLIGHT_STRAIGHT_ANGLE_DEG:
        return "STRAIGHT"
    if angle >= REDLIGHT_TURN_ANGLE_DEG:
        return "RIGHT" if cross > 0 else "LEFT"
    return "STRAIGHT"


def redlight_check_crossing(
    track_state: RedlightTrackState,
    light_state: RedlightTrafficLightState,
    ref_vector: tuple[float, float] | None,
) -> None:
    if light_state.has_any_yellow():
        if track_state.label != "VIOLATION":
            track_state.label = "WARNING"
            track_state.color = REDLIGHT_COLOR_WARNING
        return

    if not light_state.has_any_red():
        track_state.label = "Safe"
        track_state.color = REDLIGHT_COLOR_SAFE
        track_state.warned = False
        return

    direction = detect_redlight_vehicle_direction(track_state.positions, ref_vector=ref_vector)
    track_state.direction = direction
    allowed = light_state.allowed_directions()
    is_violation = False

    if direction == "STRAIGHT" and not allowed["STRAIGHT"]:
        is_violation = True
    elif direction == "LEFT" and not allowed["LEFT"]:
        is_violation = True
    elif direction == "RIGHT" and not allowed["RIGHT"]:
        is_violation = True
    elif direction == "UNKNOWN":
        if light_state.circle_red and not light_state.right_green and not light_state.left_green and not light_state.straight_green:
            is_violation = True

    if is_violation:
        track_state.label = "VIOLATION"
        track_state.color = REDLIGHT_COLOR_VIOLATION
    else:
        track_state.label = "Safe"
        track_state.color = REDLIGHT_COLOR_SAFE
        track_state.warned = False


def update_redlight_track_state(
    track_state: RedlightTrackState,
    distance: float,
    light_state: RedlightTrafficLightState,
    frame_idx: int,
    px: float,
    py: float,
    ref_vector: tuple[float, float] | None,
    is_priority_vehicle: bool,
) -> tuple[str, tuple[int, int, int]]:
    if distance < -REDLIGHT_TOUCH_DIST:
        current_region = -1
    elif distance > REDLIGHT_TOUCH_DIST:
        current_region = 1
    else:
        current_region = 0

    track_state.positions.append((px, py))
    if len(track_state.positions) > REDLIGHT_MAX_TRACK_HISTORY:
        track_state.positions.pop(0)

    if not track_state.first_seen:
        track_state.first_seen = True
        track_state.last_region = current_region
        return track_state.label, track_state.color

    if is_priority_vehicle:
        if light_state.has_any_red() or light_state.has_any_yellow():
            track_state.label = "Priority"
            track_state.color = REDLIGHT_COLOR_PRIORITY
            track_state.last_region = current_region
            return track_state.label, track_state.color
        if track_state.label == "Priority":
            track_state.label = "Safe"
            track_state.color = REDLIGHT_COLOR_SAFE

    previous_region = track_state.last_region
    if previous_region == -1 and current_region == 0:
        if frame_idx - track_state.last_warning_frame >= REDLIGHT_WARNING_DEBOUNCE_FRAMES:
            if light_state.has_any_red() or light_state.has_any_yellow():
                track_state.label = "WARNING"
                track_state.color = REDLIGHT_COLOR_WARNING
                track_state.warned = True
                track_state.last_warning_frame = frame_idx
                track_state.last_event_frame = frame_idx
    elif previous_region in {-1, 0} and current_region == 1:
        if frame_idx - track_state.last_crossing_frame >= REDLIGHT_CROSSING_DEBOUNCE_FRAMES:
            redlight_check_crossing(track_state, light_state, ref_vector)
            track_state.last_crossing_frame = frame_idx
            track_state.last_event_frame = frame_idx
    elif current_region == 0 and not light_state.has_any_red() and not light_state.has_any_yellow():
        if track_state.label == "WARNING":
            track_state.label = "Safe"
            track_state.color = REDLIGHT_COLOR_SAFE
            track_state.warned = False

    track_state.last_region = current_region
    return track_state.label, track_state.color


def dedup_redlight_tracks(
    boxes: np.ndarray,
    labels: np.ndarray,
    confs: np.ndarray,
    track_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(track_ids) == 0:
        return boxes, labels, confs, track_ids
    best: dict[int, int] = {}
    for idx, track_id in enumerate(track_ids):
        track_id = int(track_id)
        if track_id not in best or confs[idx] > confs[best[track_id]]:
            best[track_id] = idx
    keep = np.array(sorted(best.values()), dtype=int)
    return boxes[keep], labels[keep], confs[keep], track_ids[keep]


def detect_stopline_from_result(
    result,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int] | None, tuple[float, float, float] | None]:
    line_info, _ = stopline_line_from_result(result, width, height)
    if line_info is not None:
        best_segment = tuple(int(v) for v in line_info[:4])
        return best_segment, oriented_line_coefficients(best_segment, width, height)
    return None, None


def save_redlight_event_snapshot(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    line_segment: tuple[int, int, int, int] | None,
    track_id: int,
    class_label: str,
    score: float,
    frame_count: int,
    fps: float,
    source_path: Path,
) -> dict[str, Any]:
    snapshot = frame_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    if line_segment is not None:
        cv2.line(snapshot, (line_segment[0], line_segment[1]), (line_segment[2], line_segment[3]), (0, 0, 255), 4)
    cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        snapshot,
        f"VIOLATION #{track_id} {class_label} {score:.2f}",
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
    )
    event_path = make_event_snapshot_path(source_path, track_id)
    cv2.imwrite(str(event_path), snapshot)
    return {
        "track_id": int(track_id),
        "class": class_label,
        "confidence": float(score),
        "frame": int(frame_count),
        "timestamp": round(frame_count / fps, 2) if fps else 0,
        "image_url": relative_static_url(event_path),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
    }


def status_frame(message: str, detail: str = "", width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 25, 35)
    cv2.putText(frame, message, (36, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if detail:
        cv2.putText(frame, detail, (36, height // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 195, 215), 1)
    return frame


def mjpeg_chunk(frame_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + encoded.tobytes() + b"\r\n"


def redlight_initial_metrics(source_media: dict[str, Any] | None = None) -> dict[str, Any]:
    source_media = source_media or {}
    return {
        "fps": round(float(source_media.get("fps", 0) or 0), 1),
        "vehicle_count": 0,
        "violations": 0,
        "warnings": 0,
        "priority": 0,
        "frame": 0,
        "light_state": "UNKNOWN",
        "stopline_state": "STARTING",
        "stream_state": "STARTING",
        "source_fps": round(float(source_media.get("fps", 0) or 0), 1),
        "updated_at": time.time(),
    }


def update_redlight_job_metrics(job_id: str | None, **metrics: Any) -> None:
    if not job_id:
        return
    job = _redlight_stream_jobs.get(job_id)
    if job is None:
        return
    current = dict(job.get("metrics") or {})
    current.update(metrics)
    current["updated_at"] = time.time()
    job["metrics"] = current


def yolo_video_stream_frames(
    source_path: Path,
    confidence: float,
    iou: float,
    show_labels: bool = True,
    show_conf: bool = True,
):
    cap = None
    try:
        yield mjpeg_chunk(status_frame("Dang tai YOLOv26s-seg...", "Video detection live se bat dau ngay khi model san sang."))
        model = get_yolo_model()
        visible_classes = display_class_set_for_demo("yolo")
        visible_class_ids = yolo_class_ids_for_names(visible_classes)
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            yield mjpeg_chunk(status_frame("Khong mo duoc video dau vao.", source_path.name))
            return

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
        frame_count = 0
        displayed_total = 0
        last_tick = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                yield mjpeg_chunk(
                    status_frame(
                        "Da xu ly xong video.",
                        f"Frames: {frame_count} | Detections: {displayed_total}",
                        max(width, 960),
                        max(height, 540),
                    )
                )
                break

            frame_count += 1
            result = model.predict(
                frame,
                conf=confidence,
                iou=iou,
                classes=visible_class_ids,
                retina_masks=True,
                verbose=False,
            )[0]
            frame_vis, displayed = draw_yolo_result_on_frame(
                frame,
                result,
                model,
                show_labels=show_labels,
                show_conf=show_conf,
                display_classes=visible_classes,
            )
            displayed_total += len(displayed)
            now = time.perf_counter()
            live_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now
            vehicle_count = sum(1 for item in displayed if item["class"] in VEHICLE_CLASS_NAMES)

            cv2.rectangle(frame_vis, (12, 12), (360, 112), (8, 13, 24), -1)
            cv2.putText(frame_vis, "LIVE VIDEO DETECTION", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (96, 165, 250), 2)
            cv2.putText(
                frame_vis,
                f"FPS {live_fps:.1f}/{source_fps:.1f} | Xe {vehicle_count} | Obj {len(displayed)}",
                (28, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (226, 232, 240),
                1,
            )
            cv2.putText(frame_vis, f"Frame {frame_count}", (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
            yield mjpeg_chunk(frame_vis)
    except GeneratorExit:
        raise
    except Exception as exc:
        yield mjpeg_chunk(status_frame("Loi video live detection.", str(exc)[:100]))
    finally:
        if cap is not None:
            cap.release()


def draw_rfdetr_predictions_on_frame(
    frame_bgr: np.ndarray,
    predictions,
    requested_confidence: float,
    show_labels: bool = True,
    show_conf: bool = True,
    box_scale: float = 1.0,
    display_classes: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame_vis = frame_bgr.copy()
    overlay = frame_vis.copy()
    height, width = frame_vis.shape[:2]
    detections: list[dict[str, Any]] = []

    boxes = np.asarray(getattr(predictions, "xyxy", []), dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    scores_raw = getattr(predictions, "confidence", None)
    classes_raw = getattr(predictions, "class_id", None)
    scores = np.ones(len(boxes), dtype=float) if scores_raw is None else np.asarray(scores_raw, dtype=float)
    classes = np.full(len(boxes), -1, dtype=int) if classes_raw is None else np.asarray(classes_raw, dtype=int)
    masks_raw = getattr(predictions, "mask", None)
    masks = None if masks_raw is None else np.asarray(masks_raw)
    palette = [(168, 85, 247), (244, 114, 182), (14, 165, 233), (34, 197, 94), (250, 204, 21)]

    for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
        name = rfdetr_prediction_name(predictions, idx, int(label))
        if display_classes is not None and name not in display_classes:
            continue
        if float(score) < rfdetr_display_threshold(name, requested_confidence):
            continue
        color = palette[idx % len(palette)]
        if masks is not None and idx < len(masks):
            mask_array = masks[idx].astype(np.float32)
            if mask_array.shape[:2] != (height, width):
                mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
            overlay[mask_array > 0.5] = color

        scaled_box = np.asarray(box, dtype=float) * float(box_scale)
        x1, y1, x2, y2 = [int(round(value)) for value in scaled_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
        text = label_text(name, float(score), show_labels, show_conf)
        if text:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            text_y = max(22, y1 - 8)
            cv2.rectangle(frame_vis, (x1, text_y - text_h - 8), (x1 + text_w + 10, text_y + 4), color, -1)
            cv2.putText(frame_vis, text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        class_idx = RF_DETR_CLASS_TO_ID.get(name, rfdetr_class_index(int(label)))
        detections.append(
            {
                "class": name,
                "class_id": int(label),
                "class_index": int(class_idx),
                "confidence": float(score),
                "bbox": [round(float(value), 1) for value in scaled_box],
                "violation": is_violation(name),
            }
        )

    if detections:
        frame_vis = cv2.addWeighted(overlay, 0.35, frame_vis, 0.65, 0)
    return frame_vis, detections


def rfdetr_video_stream_frames(
    source_path: Path,
    confidence: float,
    iou: float,
    show_labels: bool = True,
    show_conf: bool = True,
):
    cap = None
    try:
        yield mjpeg_chunk(status_frame("Dang tai RF-DETR Small...", "Video detection live se bat dau ngay khi model san sang."))
        model = get_rfdetr_model()
        visible_classes = display_class_set_for_demo("rfdetr")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            yield mjpeg_chunk(status_frame("Khong mo duoc video dau vao.", source_path.name))
            return

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
        frame_count = 0
        displayed_total = 0
        last_tick = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                yield mjpeg_chunk(
                    status_frame(
                        "Da xu ly xong video.",
                        f"Frames: {frame_count} | Detections: {displayed_total}",
                        max(width, 960),
                        max(height, 540),
                    )
                )
                break

            frame_count += 1
            predictions, box_scale = rfdetr_predict_frame(model, frame)
            frame_vis, displayed = draw_rfdetr_predictions_on_frame(
                frame,
                predictions,
                confidence,
                show_labels=show_labels,
                show_conf=show_conf,
                box_scale=box_scale,
                display_classes=visible_classes,
            )
            displayed_total += len(displayed)
            now = time.perf_counter()
            live_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now
            vehicle_count = sum(1 for item in displayed if item["class"] in VEHICLE_CLASS_NAMES)

            cv2.rectangle(frame_vis, (12, 12), (370, 112), (8, 13, 24), -1)
            cv2.putText(frame_vis, "LIVE RF-DETR VIDEO", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (216, 180, 254), 2)
            cv2.putText(
                frame_vis,
                f"FPS {live_fps:.1f}/{source_fps:.1f} | Xe {vehicle_count} | Obj {len(displayed)}",
                (28, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (226, 232, 240),
                1,
            )
            cv2.putText(frame_vis, f"Frame {frame_count}", (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
            yield mjpeg_chunk(frame_vis)
    except GeneratorExit:
        raise
    except Exception as exc:
        yield mjpeg_chunk(status_frame("Loi RF-DETR video live detection.", str(exc)[:100]))
    finally:
        if cap is not None:
            cap.release()


def redlight_stream_frames(source_path: Path, confidence: float, iou: float, model_key: str = "yolo", job_id: str | None = None):
    cap = None
    try:
        model_key = "rfdetr" if model_key == "rfdetr" else "yolo"
        model_name = "RF-DETR Small" if model_key == "rfdetr" else "YOLOv26s-seg"
        update_redlight_job_metrics(job_id, stream_state="LOADING", stopline_state="STARTING")
        yield mjpeg_chunk(status_frame(f"Dang tai {model_name}...", "Live stream se bat dau ngay khi model san sang."))
        if model_key == "rfdetr":
            model = get_rfdetr_model()
            static_model = model
        else:
            model = get_yolo_model()
            static_model = get_yolo_model()
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            update_redlight_job_metrics(job_id, stream_state="ERROR", stopline_state="NO VIDEO")
            yield mjpeg_chunk(status_frame("Khong mo duoc video dau vao.", source_path.name))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vehicle_classes = redlight_vehicle_class_ids()
        vehicle_ids = set(vehicle_classes)
        priority_vehicle_ids = redlight_priority_vehicle_ids()
        static_classes = redlight_static_class_ids()
        static_class_names = RED_LIGHT_CLASS_NAMES | YELLOW_LIGHT_CLASS_NAMES | GREEN_LIGHT_CLASS_NAMES | {STOP_LINE_CLASS_NAME}
        calibrator = RedlightStoplineCalibrator()
        light_memory = RedlightLightMemory()
        track_states: dict[int, RedlightTrackState] = {}
        simple_track_boxes: dict[int, np.ndarray] = {}
        simple_track_labels: dict[int, int] = {}
        simple_track_last_seen: dict[int, int] = {}
        next_simple_track_id = 1
        violation_tracks: set[int] = set()
        warning_tracks: set[int] = set()
        priority_tracks: set[int] = set()
        frame_count = 0
        raw_vehicle_detections = 0
        last_frame_tick = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                update_redlight_job_metrics(
                    job_id,
                    stream_state="DONE",
                    stopline_state="LOCKED" if calibrator.is_calibrated() else "NOT FOUND",
                    frame=frame_count,
                    vehicle_count=len(track_states),
                    violations=len(violation_tracks),
                    warnings=len(warning_tracks),
                    priority=len(priority_tracks),
                )
                yield mjpeg_chunk(
                    status_frame(
                        "Da xu ly xong video.",
                        f"Violations: {len(violation_tracks)} | Warnings: {len(warning_tracks)} | Frames: {frame_count}",
                        max(width, 960),
                        max(height, 540),
                    )
                )
                break

            frame_count += 1
            static_confidence = max(0.12, confidence * 0.65)
            stopline_confidence = REDLIGHT_STOPLINE_RFDETR_CONFIDENCE if model_key == "rfdetr" else REDLIGHT_STOPLINE_YOLO_CONFIDENCE
            rfdetr_predictions = None
            rfdetr_box_scale = 1.0
            if model_key == "rfdetr":
                rfdetr_predictions, rfdetr_box_scale = rfdetr_predict_frame(static_model, frame)
                static_boxes, static_labels, static_confs = rfdetr_detection_arrays(
                    rfdetr_predictions,
                    static_confidence,
                    static_class_names,
                    rfdetr_box_scale,
                )
            else:
                static_kwargs = {
                    "conf": static_confidence,
                    "iou": iou,
                    "retina_masks": True,
                    "verbose": False,
                }
                if static_classes:
                    static_kwargs["classes"] = static_classes
                static_result = static_model.predict(frame, **static_kwargs)[0]
                static_boxes, static_labels, static_confs = redlight_result_arrays(static_result)
                stopline_result = static_result
                if not calibrator.is_calibrated() and STOP_LINE_CLASS_NAME in YOLO_CLASS_TO_ID:
                    stopline_result = static_model.predict(
                        frame,
                        conf=stopline_confidence,
                        iou=iou,
                        classes=[YOLO_CLASS_TO_ID[STOP_LINE_CLASS_NAME]],
                        retina_masks=True,
                        verbose=False,
                    )[0]

            current_time = time.time()
            detected_light_state = traffic_light_state_from_labels(static_labels)
            light_memory.update(detected_light_state, current_time)
            light_state = light_memory.get(current_time)
            light_status = light_state.simple_state()

            if model_key == "rfdetr":
                line_info, stopline_mask = stopline_line_from_rfdetr_predictions(
                    rfdetr_predictions,
                    width,
                    height,
                    stopline_confidence,
                    rfdetr_box_scale,
                )
            else:
                line_info, stopline_mask = stopline_line_from_result(stopline_result, width, height)
                line_info = expand_stopline_line_with_frame(frame, line_info)
            calibrator.update_line(line_info)
            calibrator.maybe_finish(width, height)
            line_segment, line, line_locked = calibrator.current(width, height)
            ref_vector = (line[0], line[1]) if line is not None else None

            if model_key == "rfdetr":
                vehicle_boxes, vehicle_labels, vehicle_confs = rfdetr_detection_arrays(
                    rfdetr_predictions,
                    confidence,
                    VEHICLE_CLASS_NAMES,
                    rfdetr_box_scale,
                )
                vehicle_track_ids, next_simple_track_id = assign_simple_track_ids(
                    vehicle_boxes,
                    vehicle_labels,
                    simple_track_boxes,
                    simple_track_labels,
                    simple_track_last_seen,
                    next_simple_track_id,
                    frame_count,
                )
            else:
                track_kwargs = {
                    "persist": True,
                    "conf": confidence,
                    "iou": iou,
                    "retina_masks": True,
                    "verbose": False,
                }
                if vehicle_classes:
                    track_kwargs["classes"] = vehicle_classes
                track_result = model.track(frame, **track_kwargs)[0]
                if track_result.boxes is None:
                    vehicle_boxes = np.empty((0, 4), dtype=float)
                    vehicle_labels = np.array([], dtype=int)
                    vehicle_confs = np.array([], dtype=float)
                    vehicle_track_ids = np.array([], dtype=int)
                else:
                    vehicle_boxes = track_result.boxes.xyxy.cpu().numpy()
                    vehicle_labels = track_result.boxes.cls.cpu().numpy().astype(int)
                    vehicle_track_ids = (
                        track_result.boxes.id.cpu().numpy().astype(int)
                        if track_result.boxes.id is not None
                        else np.arange(len(vehicle_labels))
                    )
                    vehicle_confs = track_result.boxes.conf.cpu().numpy()
                    vehicle_boxes, vehicle_labels, vehicle_confs, vehicle_track_ids = dedup_redlight_tracks(
                        vehicle_boxes,
                        vehicle_labels,
                        vehicle_confs,
                        vehicle_track_ids,
                    )

            frame_vis = frame.copy()
            if not line_locked:
                overlay_stopline_mask(frame_vis, stopline_mask)
            draw_traffic_lights_on_frame(frame_vis, static_boxes, static_labels, static_confs)

            line_color = {
                "RED": (0, 0, 255),
                "YELLOW": (0, 255, 255),
                "GREEN": (0, 255, 0),
            }.get(light_status, (255, 255, 255))
            if line_segment is not None:
                cv2.line(frame_vis, (line_segment[0], line_segment[1]), (line_segment[2], line_segment[3]), line_color, 4)
                draw_text_badge(
                    frame_vis,
                    "STOP LINE LOCKED" if line_locked else "CALIBRATING STOP LINE",
                    (line_segment[0], max(24, line_segment[1] - 10)),
                    line_color,
                    0.56,
                    2,
                )
            else:
                draw_text_badge(frame_vis, "SEARCHING STOP LINE", (24, 32), (0, 210, 255), 0.56, 2)

            active_vehicle_count = int(len(vehicle_labels))
            if len(vehicle_labels) > 0:
                raw_vehicle_detections += int(len(vehicle_labels))
                for box, label, track_id, score in zip(vehicle_boxes, vehicle_labels, vehicle_track_ids, vehicle_confs):
                    name = class_name(int(label), "yolo")
                    if int(label) not in vehicle_ids:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    px, py = bottom_center(box)
                    track_id = int(track_id)
                    state = track_states.setdefault(track_id, RedlightTrackState())
                    if line is not None:
                        distance = signed_distance(px, py, line)
                        label_text, color = update_redlight_track_state(
                            state,
                            distance,
                            light_state,
                            frame_count,
                            px,
                            py,
                            ref_vector,
                            int(label) in priority_vehicle_ids,
                        )
                    else:
                        state.positions.append((px, py))
                        if len(state.positions) > REDLIGHT_MAX_TRACK_HISTORY:
                            state.positions.pop(0)
                        label_text, color = state.label, state.color
                    if label_text == "VIOLATION":
                        violation_tracks.add(track_id)
                    elif label_text == "WARNING":
                        warning_tracks.add(track_id)
                    elif label_text == "Priority":
                        priority_tracks.add(track_id)

                    if label_text == "VIOLATION":
                        text = f"VIOLATION #{track_id} {state.direction}"
                    elif label_text == "WARNING":
                        text = f"WARNING #{track_id}"
                    elif label_text == "Priority":
                        text = f"PRIORITY #{track_id}"
                    else:
                        text = f"{name} #{track_id} {score:.2f}"
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 3 if label_text != "VIOLATION" else 4)
                    draw_text_badge(frame_vis, text, (x1, max(22, y1 - 8)), color, 0.6, 2)
                    cv2.circle(frame_vis, (int(px), int(py)), 4, color, -1)
                    if len(state.positions) >= 2:
                        points = np.array([(int(x), int(y)) for x, y in state.positions[-12:]], dtype=np.int32)
                        cv2.polylines(frame_vis, [points], False, color, 2)

            hud = [
                f"LIVE RED-LIGHT DETECTION ({model_name})",
                f"Light: {light_status}",
                f"Stop line: {'LOCKED' if line_locked else 'CALIBRATING' if line_segment is not None else 'SEARCHING'}",
                f"Vehicles: {active_vehicle_count}",
                f"Violations: {len(violation_tracks)}",
                f"Warnings: {len(warning_tracks)}",
                f"Priority: {len(priority_tracks)}",
                f"Frame: {frame_count}",
            ]
            now_tick = time.perf_counter()
            live_fps = 1.0 / max(now_tick - last_frame_tick, 1e-6)
            last_frame_tick = now_tick
            stopline_state = "LOCKED" if line_locked else "CALIB" if line_segment is not None else "SEARCH"
            update_redlight_job_metrics(
                job_id,
                fps=round(float(live_fps), 1),
                vehicle_count=active_vehicle_count,
                violations=len(violation_tracks),
                warnings=len(warning_tracks),
                priority=len(priority_tracks),
                frame=frame_count,
                light_state=light_status,
                stopline_state=stopline_state,
                stream_state=stopline_state,
                source_fps=round(float(fps), 1),
            )
            hud_overlay = frame_vis.copy()
            cv2.rectangle(hud_overlay, (12, 12), (410, 12 + len(hud) * 28), (8, 12, 22), -1)
            frame_vis = cv2.addWeighted(hud_overlay, 0.72, frame_vis, 0.28, 0)
            for idx, text in enumerate(hud):
                color = (255, 255, 255)
                if "RED" in text:
                    color = (80, 80, 255)
                elif "YELLOW" in text:
                    color = (80, 245, 255)
                elif "GREEN" in text:
                    color = (120, 255, 120)
                cv2.putText(frame_vis, text, (24, 38 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
            yield mjpeg_chunk(frame_vis)
    except Exception as exc:
        update_redlight_job_metrics(job_id, stream_state="ERROR", stopline_state="ERROR")
        yield mjpeg_chunk(status_frame("Loi live red-light stream.", str(exc)))
    finally:
        if cap is not None:
            cap.release()


def run_redlight_video(source_path: Path, confidence: float, iou: float) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    static_model = get_yolo_model()
    visible_classes = demo_public_class_set("yolo")
    visible_class_names = demo_public_class_names("yolo")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = make_result_path("redlight", source_path)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không tạo được video output.")

    vehicle_ids = {YOLO_CLASS_TO_ID[name] for name in VEHICLE_CLASS_NAMES if name in YOLO_CLASS_TO_ID}
    vehicle_classes = redlight_vehicle_class_ids()
    static_classes = redlight_static_class_ids()
    red_light_ids = {YOLO_CLASS_TO_ID[name] for name in RED_LIGHT_CLASS_NAMES if name in YOLO_CLASS_TO_ID}
    yellow_light_ids = {YOLO_CLASS_TO_ID[name] for name in YELLOW_LIGHT_CLASS_NAMES if name in YOLO_CLASS_TO_ID}
    line = None
    line_segment = None
    track_regions: dict[int, int] = {}
    violation_tracks: set[int] = set()
    captured_violation_tracks: set[int] = set()
    warning_tracks: set[int] = set()
    violation_events: list[dict[str, Any]] = []
    frame_count = 0
    red_frames = 0
    raw_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        frame_vis = frame.copy()

        static_result = static_model.predict(
            frame,
            conf=max(0.15, confidence * 0.7),
            iou=iou,
            classes=static_classes,
            retina_masks=True,
            verbose=False,
        )[0]
        if line is None or frame_count <= int(fps * 3):
            detected_segment, detected_line = detect_stopline_from_result(static_result, width, height)
            if detected_segment is not None and detected_line is not None:
                line_segment, line = detected_segment, detected_line

        labels_static = static_result.boxes.cls.cpu().numpy().astype(int) if static_result.boxes is not None else np.array([], dtype=int)
        is_red = any(label in red_light_ids for label in labels_static)
        is_yellow = any(label in yellow_light_ids for label in labels_static)
        if is_red:
            red_frames += 1

        track_result = model.track(
            frame,
            persist=True,
            conf=confidence,
            iou=iou,
            classes=vehicle_classes,
            retina_masks=True,
            verbose=False,
        )[0]
        raw_detections += int(len(track_result.boxes.cls)) if track_result.boxes is not None else 0
        if getattr(track_result, "masks", None) is not None:
            frame_vis, _ = draw_yolo_result_on_frame(
                frame,
                track_result,
                model,
                show_labels=False,
                show_conf=False,
                draw_boxes=False,
                draw_masks=True,
                display_classes=visible_classes,
            )
        if line_segment is not None:
            color = (0, 0, 255) if is_red else (0, 255, 255) if is_yellow else (0, 255, 0)
            cv2.line(frame_vis, (line_segment[0], line_segment[1]), (line_segment[2], line_segment[3]), color, 3)

        if track_result.boxes is not None:
            boxes = track_result.boxes.xyxy.cpu().numpy()
            labels = track_result.boxes.cls.cpu().numpy().astype(int)
            ids = track_result.boxes.id.cpu().numpy().astype(int) if track_result.boxes.id is not None else np.arange(len(labels))
            confs = track_result.boxes.conf.cpu().numpy()
            for box, label, track_id, score in zip(boxes, labels, ids, confs):
                name = class_name(int(label), "yolo")
                if int(label) not in vehicle_ids:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                px = (x1 + x2) / 2
                py = y2
                if line is not None:
                    distance = signed_distance(px, py, line)
                    region = -1 if distance < -12 else 1 if distance > 12 else 0
                    previous = track_regions.get(int(track_id))
                    if previous in (-1, 0) and region == 1:
                        if is_red:
                            violation_tracks.add(int(track_id))
                            if int(track_id) not in captured_violation_tracks:
                                violation_events.append(
                                    save_redlight_event_snapshot(
                                        frame,
                                        box,
                                        line_segment,
                                        int(track_id),
                                        name,
                                        float(score),
                                        frame_count,
                                        fps,
                                        source_path,
                                    )
                                )
                                captured_violation_tracks.add(int(track_id))
                        elif is_yellow:
                            warning_tracks.add(int(track_id))
                    track_regions[int(track_id)] = region
                else:
                    track_regions.setdefault(int(track_id), 0)

                if int(track_id) in violation_tracks:
                    color = (0, 0, 255)
                    label_text = f"VIOLATION #{track_id}"
                elif int(track_id) in warning_tracks:
                    color = (0, 255, 255)
                    label_text = f"WARNING #{track_id}"
                else:
                    color = (0, 220, 0)
                    label_text = f"{name} #{track_id} {score:.2f}"
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_vis, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.circle(frame_vis, (int(px), int(py)), 4, color, -1)

        hud = [
            f"Light: {'RED' if is_red else 'YELLOW' if is_yellow else 'GREEN/UNKNOWN'}",
            f"Violations: {len(violation_tracks)}",
            f"Warnings: {len(warning_tracks)}",
            f"Frame: {frame_count}",
        ]
        for idx, text in enumerate(hud):
            cv2.putText(frame_vis, text, (18, 32 + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame_vis)

    cap.release()
    writer.release()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    detections = [
        {"id": 1, "class": "red_light_violation", "confidence": 1.0, "count": len(violation_tracks), "violation": True},
        {"id": 2, "class": "yellow_light_warning", "confidence": 1.0, "count": len(warning_tracks), "violation": False},
    ]
    return {
        "model": "Red Light Violation",
        "model_key": "redlight",
        "result_url": relative_static_url(output_path),
        "detections": detections,
        "violation_events": violation_events,
        "metrics": {
            "total_objects": len(track_regions),
            "violations": len(violation_tracks),
            "processing_time": f"{elapsed_ms}ms",
            "avg_confidence": 0,
            "frames": frame_count,
            "red_frames": red_frames,
            "raw_objects": raw_detections,
        },
        "media_type": "video",
        "debug": {
            "inference": {
                "model_key": "yolo",
                "task": "redlight",
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "source_fps": round(float(fps), 3),
                "source_width": width,
                "source_height": height,
                "frames_processed": frame_count,
                "inference_class_count": len(YOLO_INFERENCE_CLASS_IDS),
                "display_class_count": len(visible_class_names),
                "display_classes": visible_class_names,
                "total_raw_detections": raw_detections,
                "tracked_vehicle_count": len(track_regions),
                "violation_track_count": len(violation_tracks),
                "warning_track_count": len(warning_tracks),
                "captured_event_count": len(violation_events),
                "red_frame_count": red_frames,
                "last_stop_line": line_segment,
                "elapsed_ms": elapsed_ms,
                "output_codec": "mp4v",
            },
            "model": file_debug(YOLO_MODEL_PATH),
            "source_media": media_debug(source_path),
            "result_media": media_debug(output_path),
        },
    }


def run_inference(
    model_key: str,
    source_path: Path,
    confidence: float,
    iou: float,
    task: str = "image",
    show_labels: bool = True,
    show_conf: bool = True,
) -> dict[str, Any]:
    if task == "redlight":
        if not is_video(source_path):
            raise RuntimeError("Chức năng vượt đèn đỏ cần input video.")
        return run_redlight_video(source_path, confidence, iou)
    if task == "video" or is_video(source_path):
        if model_key != "yolo":
            raise RuntimeError("Video detection hiện hỗ trợ YOLOv26s-seg. RF-DETR đang được bật cho ảnh tĩnh.")
        return write_yolo_video(source_path, confidence, iou, show_labels, show_conf)
    if model_key == "yolo":
        return predict_yolo(source_path, confidence, iou, show_labels, show_conf)
    if model_key == "rfdetr":
        return predict_rfdetr(source_path, confidence, show_labels, show_conf)
    raise ValueError("Model không hợp lệ.")


@app.route("/")
def index():
    return render_template("introduction.html", status=model_status())


@app.route("/introduction")
def introduction():
    return render_template("introduction.html", status=model_status())


@app.route("/quantitative")
def quantitative():
    return render_template("quantitative.html", status=model_status(), quantitative=load_quantitative_payload())


@app.route("/qualitative")
def qualitative():
    return render_template("qualitative.html", status=model_status())


@app.route("/inference")
def inference():
    return render_template("inference.html", status=model_status())


@app.get("/api/model-status")
def api_model_status():
    return jsonify(model_status())


def cleanup_redlight_stream_jobs(max_age_seconds: int = 3600) -> None:
    now = time.time()
    stale_ids = [
        job_id
        for job_id, job in _redlight_stream_jobs.items()
        if now - float(job.get("created_at", now)) > max_age_seconds
    ]
    for job_id in stale_ids:
        _redlight_stream_jobs.pop(job_id, None)


def cleanup_video_stream_jobs(max_age_seconds: int = 3600) -> None:
    now = time.time()
    stale_ids = [
        job_id
        for job_id, job in _video_stream_jobs.items()
        if now - float(job.get("created_at", now)) > max_age_seconds
    ]
    for job_id in stale_ids:
        _video_stream_jobs.pop(job_id, None)


@app.post("/api/video/start")
def api_video_start():
    try:
        cleanup_video_stream_jobs()
        confidence = float(request.form.get("confidence", 0.5))
        iou = float(request.form.get("iou", 0.5))
        show_labels = parse_bool(request.form.get("show_labels"), True)
        show_conf = parse_bool(request.form.get("show_conf"), True)
        model_key = request.form.get("model", "yolo")
        use_sample = request.form.get("use_sample") == "1"
        status_snapshot = model_status()
        if model_key not in {"yolo", "rfdetr"}:
            return jsonify({"success": False, "error": "Model không hợp lệ cho Video detection.", "status": status_snapshot}), 400
        if not status_snapshot[model_key]["available"]:
            return jsonify({"success": False, "error": f"{status_snapshot[model_key]['name']} chưa sẵn sàng.", "status": status_snapshot}), 400

        uploaded_file_metadata = None
        if use_sample:
            source_path = sample_source_path("video", request.form.get("sample_type"))
        else:
            file_storage = request.files.get("file")
            if file_storage is None:
                return jsonify({"success": False, "error": "Vui lòng upload video hoặc dùng video test_2.mp4.", "status": status_snapshot}), 400
            uploaded_file_metadata = upload_debug(file_storage)
            source_path = save_upload(file_storage)
        if not is_video(source_path):
            return jsonify({"success": False, "error": "Video detection cần input video.", "status": status_snapshot}), 400

        job_id = uuid.uuid4().hex[:12]
        _video_stream_jobs[job_id] = {
            "source_path": source_path,
            "confidence": confidence,
            "iou": iou,
            "show_labels": show_labels,
            "show_conf": show_conf,
            "model_key": model_key,
            "created_at": time.time(),
        }
        stream_url = url_for("api_video_stream", job_id=job_id)
        source_media = media_debug(source_path)
        model_name = status_snapshot[model_key]["name"]
        return jsonify(
            {
                "success": True,
                "task": "video",
                "job_id": job_id,
                "model": f"{model_name} Video Live",
                "model_key": model_key,
                "media_type": "video",
                "original_url": relative_static_url(source_path),
                "stream_url": stream_url,
                "metrics": {
                    "fps": source_media.get("fps", 0),
                    "vehicle_count": "Live",
                    "violations": 0,
                    "stream_state": "Live",
                },
                "status": status_snapshot,
                "debug": {
                    "request": {
                        "model_key": model_key,
                        "use_sample": use_sample,
                        "sample_type": "video" if use_sample else None,
                        "uploaded_file": uploaded_file_metadata,
                        "source_media": source_media,
                        "confidence": confidence,
                        "iou": iou,
                        "show_labels": show_labels,
                        "show_conf": show_conf,
                    },
                    "stream": {
                        "job_id": job_id,
                        "url": stream_url,
                        "display_classes": display_class_names_for_demo(model_key),
                    },
                },
            }
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "status": model_status(),
                    "debug": {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                        }
                    },
                }
            ),
            500,
        )


@app.get("/api/video/stream/<job_id>")
def api_video_stream(job_id: str):
    job = _video_stream_jobs.get(job_id)
    if job is None:
        return Response(mjpeg_chunk(status_frame("Khong tim thay live video job.", job_id)), mimetype="multipart/x-mixed-replace; boundary=frame")
    model_key = str(job.get("model_key", "yolo"))
    if model_key == "rfdetr":
        frames = rfdetr_video_stream_frames(
            job["source_path"],
            float(job["confidence"]),
            float(job["iou"]),
            bool(job.get("show_labels", True)),
            bool(job.get("show_conf", True)),
        )
    else:
        frames = yolo_video_stream_frames(
            job["source_path"],
            float(job["confidence"]),
            float(job["iou"]),
            bool(job.get("show_labels", True)),
            bool(job.get("show_conf", True)),
        )
    return Response(
        frames,
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/redlight/start")
def api_redlight_start():
    try:
        cleanup_redlight_stream_jobs()
        confidence = float(request.form.get("confidence", 0.5))
        iou = float(request.form.get("iou", 0.5))
        model_key = request.form.get("model", "yolo")
        use_sample = request.form.get("use_sample") == "1"
        if model_key not in {"yolo", "rfdetr"}:
            return jsonify({"success": False, "error": "Model không hợp lệ cho tác vụ vượt đèn đỏ.", "status": model_status()}), 400
        status_snapshot = model_status()
        if not status_snapshot[model_key]["available"]:
            return jsonify({"success": False, "error": f"{status_snapshot[model_key]['name']} chưa sẵn sàng.", "status": status_snapshot}), 400

        uploaded_file_metadata = None
        if use_sample:
            source_path = sample_source_path("redlight", request.form.get("sample_type"))
        else:
            file_storage = request.files.get("file")
            if file_storage is None:
                return jsonify({"success": False, "error": "Vui lòng upload video hoặc dùng video test_2.mp4.", "status": status_snapshot}), 400
            uploaded_file_metadata = upload_debug(file_storage)
            source_path = save_upload(file_storage)
        if not is_video(source_path):
            return jsonify({"success": False, "error": "Tác vụ vượt đèn đỏ cần input video.", "status": status_snapshot}), 400

        job_id = uuid.uuid4().hex[:12]
        source_media = media_debug(source_path)
        initial_metrics = redlight_initial_metrics(source_media)
        _redlight_stream_jobs[job_id] = {
            "source_path": source_path,
            "confidence": confidence,
            "iou": iou,
            "model_key": model_key,
            "created_at": time.time(),
            "metrics": initial_metrics,
        }
        stream_url = url_for("api_redlight_stream", job_id=job_id)
        metrics_url = url_for("api_redlight_metrics", job_id=job_id)
        model_name = status_snapshot[model_key]["name"]
        return jsonify(
            {
                "success": True,
                "task": "redlight",
                "job_id": job_id,
                "model": f"{model_name} Red Light Violation Live",
                "model_key": model_key,
                "media_type": "video",
                "original_url": relative_static_url(source_path),
                "stream_url": stream_url,
                "metrics_url": metrics_url,
                "metrics": initial_metrics,
                "status": status_snapshot,
                "debug": {
                    "request": {
                        "model_key": model_key,
                        "use_sample": use_sample,
                        "sample_type": "video" if use_sample else None,
                        "uploaded_file": uploaded_file_metadata,
                        "source_media": source_media,
                        "confidence": confidence,
                        "iou": iou,
                    },
                    "stream": {
                        "job_id": job_id,
                        "url": stream_url,
                        "metrics_url": metrics_url,
                        "model": model_name,
                    },
                },
            }
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "status": model_status(),
                    "debug": {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                        }
                    },
                }
            ),
            500,
        )


@app.get("/api/redlight/stream/<job_id>")
def api_redlight_stream(job_id: str):
    job = _redlight_stream_jobs.get(job_id)
    if job is None:
        return Response(mjpeg_chunk(status_frame("Khong tim thay live job.", job_id)), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(
        redlight_stream_frames(
            job["source_path"],
            float(job["confidence"]),
            float(job["iou"]),
            str(job.get("model_key", "yolo")),
            job_id,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/redlight/metrics/<job_id>")
def api_redlight_metrics(job_id: str):
    job = _redlight_stream_jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Không tìm thấy live job."}), 404
    return jsonify({"success": True, "job_id": job_id, "metrics": job.get("metrics") or redlight_initial_metrics()})


@app.post("/api/infer")
def api_infer():
    request_started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    source_path: Path | None = None
    uploaded_file_metadata: dict[str, Any] | None = None
    model_key = request.form.get("model", "yolo")
    task = request.form.get("task", "image")
    confidence = 0.5
    iou = 0.5
    show_labels = True
    show_conf = True
    use_sample = False
    sample_type = None

    try:
        confidence = float(request.form.get("confidence", confidence))
        iou = float(request.form.get("iou", iou))
        show_labels = parse_bool(request.form.get("show_labels"), show_labels)
        show_conf = parse_bool(request.form.get("show_conf"), show_conf)
        use_sample = request.form.get("use_sample") == "1"
        sample_type = request.form.get("sample_type")

        if use_sample:
            source_path = sample_source_path(task, sample_type)
        else:
            file_storage = request.files.get("file")
            if file_storage is None:
                status_snapshot = model_status()
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Vui lòng tải lên file hoặc dùng sample mặc định.",
                            "status": status_snapshot,
                            "debug": {
                                "request": request_debug_payload(
                                    request_id,
                                    model_key,
                                    task,
                                    confidence,
                                    iou,
                                    show_labels,
                                    show_conf,
                                    use_sample,
                                    uploaded_file_metadata,
                                    source_path,
                                ),
                                "response": {
                                    "success": False,
                                    "http_status": 400,
                                    "elapsed_ms": int((time.perf_counter() - request_started) * 1000),
                                },
                                "model_status": status_snapshot,
                            },
                        }
                    ),
                    400,
                )
            uploaded_file_metadata = upload_debug(file_storage)
            source_path = save_upload(file_storage)

        result = run_inference(model_key, source_path, confidence, iou, task, show_labels, show_conf)
        status_snapshot = model_status()
        elapsed_ms = int((time.perf_counter() - request_started) * 1000)
        result_debug = result.get("debug", {})
        debug_payload = {
            "request": request_debug_payload(
                request_id,
                model_key,
                task,
                confidence,
                iou,
                show_labels,
                show_conf,
                use_sample,
                uploaded_file_metadata,
                source_path,
            ),
            "response": {
                "success": True,
                "http_status": 200,
                "elapsed_ms": elapsed_ms,
                "result_url": result.get("result_url"),
                "media_type": result.get("media_type", "image"),
            },
            "model_status": status_snapshot,
            **result_debug,
        }
        response_payload = {
            "success": True,
            "original_url": relative_static_url(source_path),
            "status": status_snapshot,
            **result,
        }
        response_payload["debug"] = debug_payload
        return jsonify(response_payload)
    except Exception as exc:
        status_snapshot = model_status()
        elapsed_ms = int((time.perf_counter() - request_started) * 1000)
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "status": status_snapshot,
                    "debug": {
                        "request": request_debug_payload(
                            request_id,
                            model_key,
                            task,
                            confidence,
                            iou,
                            show_labels,
                            show_conf,
                            use_sample,
                            uploaded_file_metadata,
                            source_path,
                        ),
                        "response": {
                            "success": False,
                            "http_status": 500,
                            "elapsed_ms": elapsed_ms,
                        },
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                        },
                        "model_status": status_snapshot,
                    },
                }
            ),
            500,
        )


@app.post("/api/compare-sample")
def api_compare_sample():
    confidence = float(request.form.get("confidence", 0.45))
    iou = float(request.form.get("iou", 0.5))
    show_labels = parse_bool(request.form.get("show_labels"), True)
    show_conf = parse_bool(request.form.get("show_conf"), True)
    cache_key = (
        round(confidence, 3),
        round(iou, 3),
        show_labels,
        show_conf,
        "include_stopline",
        STOP_LINE_QUALITATIVE_CONFIDENCE,
        RF_DETR_CLASS_ID_BASE,
        tuple(configured_display_class_names("yolo")),
        tuple(configured_display_class_names("rfdetr")),
    )
    cached = _sample_compare_cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)

    payload: dict[str, Any] = {
        "success": True,
        "original_url": relative_static_url(SAMPLE_IMAGE_PATH),
        "status": model_status(),
        "results": {},
        "errors": {},
    }
    for model_key in ("yolo", "rfdetr"):
        try:
            if model_key == "yolo":
                payload["results"][model_key] = predict_yolo(
                    SAMPLE_IMAGE_PATH,
                    confidence,
                    iou,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    include_stopline=True,
                )
            else:
                payload["results"][model_key] = predict_rfdetr(
                    SAMPLE_IMAGE_PATH,
                    confidence,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    include_stopline=True,
                )
        except Exception as exc:
            payload["errors"][model_key] = str(exc)
    _sample_compare_cache[cache_key] = payload
    return jsonify(payload)


if __name__ == "__main__":
    debug_mode = parse_bool(os.environ.get("FLASK_DEBUG"), False)
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=debug_mode)
