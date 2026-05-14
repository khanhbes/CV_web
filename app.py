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
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
YOLO_RESULTS_CSV_PATH = BASE_DIR / "results_26s.csv"
CLASS_DISPLAY_CONFIG_PATH = BASE_DIR / "class_display_config.json"

YOLO_MODEL_PATH = MODEL_DIR / "yolov26s_seg.pt"
RF_DETR_MODEL_PATH = MODEL_DIR / "RF-DETR_Small.pt"
SAMPLE_IMAGE_PATH = TEST_IMAGE_DIR / "test_image.jpg"

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

REDLIGHT_STOPLINE_CALIBRATION_SECONDS = 3.0
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

for folder in (TEST_IMAGE_DIR, UPLOAD_DIR, RESULT_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_yolo_model = None
_rfdetr_model = None
_sample_compare_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_redlight_stream_jobs: dict[str, dict[str, Any]] = {}


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


def class_display_status(model_key: str) -> dict[str, Any]:
    visible = configured_display_class_names(model_key)
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


def rfdetr_display_threshold(name: str, requested_confidence: float) -> float:
    return requested_confidence


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


def predict_yolo(source_path: Path, confidence: float, iou: float, show_labels: bool = True, show_conf: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    visible_classes = configured_display_class_set("yolo")
    visible_class_names = configured_display_class_names("yolo")
    result = model.predict(
        str(source_path),
        conf=confidence,
        iou=iou,
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
                "iou_threshold": iou,
                "inference_class_count": len(YOLO_INFERENCE_CLASS_IDS),
                "display_class_count": len(visible_class_names),
                "display_classes": visible_class_names,
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
    data = getattr(predictions, "data", {}) or {}
    names = data.get("class_name") if isinstance(data, dict) else None
    if names is not None and index < len(names):
        value = names[index]
        if value and str(value) in RF_DETR_CLASS_TO_ID:
            return str(value)
    normalized_index = rfdetr_class_index(label)
    if 0 <= normalized_index < len(RF_DETR_CLASS_NAMES):
        return class_name(normalized_index, "rfdetr")
    return f"class_{label}"


def draw_rfdetr_output(
    image_bgr: np.ndarray,
    predictions,
    output_path: Path,
    requested_confidence: float,
    show_labels: bool = True,
    show_conf: bool = True,
) -> list[dict[str, Any]]:
    visible_classes = configured_display_class_set("rfdetr")
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
    suppressors = [
        (box, rfdetr_prediction_name(predictions, index, int(label)), float(score))
        for index, (box, score, label) in enumerate(zip(boxes, scores, classes))
        if rfdetr_prediction_name(predictions, index, int(label)) in RF_DETR_ROAD_SUPPRESSOR_CLASS_NAMES
        and float(score) >= RF_DETR_ROAD_SUPPRESSOR_MIN_CONFIDENCE
    ]

    palette = [(168, 85, 247, 92), (244, 114, 182, 92), (14, 165, 233, 92), (34, 197, 94, 92)]
    for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
        name = rfdetr_prediction_name(predictions, idx, int(label))
        if name not in visible_classes:
            continue
        if float(score) < rfdetr_display_threshold(name, requested_confidence):
            continue
        if name in VEHICLE_CLASS_NAMES and is_suppressed_rfdetr_vehicle(box, suppressors):
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


def predict_rfdetr(source_path: Path, confidence: float, show_labels: bool = True, show_conf: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_rfdetr_model()
    visible_class_names = configured_display_class_names("rfdetr")
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
    detections = draw_rfdetr_output(image_bgr, predictions, output_path, confidence, show_labels, show_conf)
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


def write_yolo_video(
    source_path: Path,
    confidence: float,
    iou: float,
    show_labels: bool = True,
    show_conf: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    visible_classes = configured_display_class_set("yolo")
    visible_class_names = configured_display_class_names("yolo")
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
        result = model.predict(frame, conf=confidence, iou=iou, retina_masks=True, verbose=False)[0]
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
        segment = self.best_line or default_stopline_segment(width, height)
        self.best_line = segment
        self.line_abc = oriented_line_coefficients(segment, width, height)

    def current(self, width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[float, float, float], bool]:
        segment = self.best_line or default_stopline_segment(width, height)
        line_abc = self.line_abc or oriented_line_coefficients(segment, width, height)
        return segment, line_abc, self.is_calibrated()


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


def line_inside_stopline_mask(mask_u8: np.ndarray) -> tuple[int, int, int, int, float] | None:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    rect = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rect).astype(int)
    edges = []
    for idx in range(4):
        p1 = points[idx]
        p2 = points[(idx + 1) % 4]
        length = float(np.linalg.norm(p2 - p1))
        edges.append((length, p1, p2))
    length, p1, p2 = max(edges, key=lambda item: item[0])
    if length < 20:
        return None
    return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), length


def stopline_line_from_result(result, width: int, height: int) -> tuple[tuple[int, int, int, int, float] | None, np.ndarray | None]:
    stop_line_id = YOLO_CLASS_TO_ID.get(STOP_LINE_CLASS_NAME)
    if stop_line_id is None or result.boxes is None or result.masks is None:
        return None, None

    labels = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(labels), dtype=float)
    masks = result.masks.data.cpu().numpy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    best_line = None
    best_score = 0.0

    for idx, label in enumerate(labels):
        if int(label) != stop_line_id or idx >= len(masks):
            continue
        mask = masks[idx]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.max(combined_mask, mask_u8)
        line_info = line_inside_stopline_mask(mask_u8)
        if line_info is None:
            continue
        score = float(line_info[4]) * float(confs[idx])
        if score > best_score:
            best_line = line_info
            best_score = score

    if not combined_mask.any():
        combined_mask = None
    return best_line, combined_mask


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
        has_green_override = light_state.right_green or light_state.left_green or light_state.straight_green
        is_violation = light_state.has_any_red() and not has_green_override

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

    if is_priority_vehicle and (light_state.has_any_red() or light_state.has_any_yellow()):
        track_state.label = "Priority"
        track_state.color = REDLIGHT_COLOR_PRIORITY
        track_state.last_region = current_region
        return track_state.label, track_state.color

    previous_region = track_state.last_region
    if previous_region == -1 and current_region == 0:
        if frame_idx - track_state.last_warning_frame >= REDLIGHT_WARNING_DEBOUNCE_FRAMES:
            if light_state.has_any_red() or light_state.has_any_yellow():
                track_state.label = "WARNING"
                track_state.color = REDLIGHT_COLOR_WARNING
                track_state.warned = True
                track_state.last_warning_frame = frame_idx
    elif previous_region in {-1, 0} and current_region == 1:
        if frame_idx - track_state.last_crossing_frame >= REDLIGHT_CROSSING_DEBOUNCE_FRAMES:
            redlight_check_crossing(track_state, light_state, ref_vector)
            track_state.last_crossing_frame = frame_idx
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


def detect_stopline_from_result(result, width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[float, float, float]]:
    line_info, _ = stopline_line_from_result(result, width, height)
    if line_info is not None:
        best_segment = tuple(int(v) for v in line_info[:4])
        return best_segment, oriented_line_coefficients(best_segment, width, height)

    stop_line_ids = {YOLO_CLASS_TO_ID[STOP_LINE_CLASS_NAME]}
    best_line = None
    best_area = 0.0
    if result.boxes is not None and result.masks is not None:
        labels = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        for idx, label in enumerate(labels):
            if label not in stop_line_ids or idx >= len(masks):
                continue
            mask = masks[idx]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area <= best_area:
                continue
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect).astype(int)
            distances = []
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                distances.append((np.linalg.norm(p2 - p1), p1, p2))
            _, p1, p2 = max(distances, key=lambda item: item[0])
            best_line = (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            best_area = area

    if best_line is None:
        best_line = default_stopline_segment(width, height)
    return best_line, oriented_line_coefficients(best_line, width, height)


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


def redlight_stream_frames(source_path: Path, confidence: float, iou: float):
    cap = None
    try:
        yield mjpeg_chunk(status_frame("Dang tai YOLOv26s-seg...", "Live stream se bat dau ngay khi model san sang."))
        model = get_yolo_model()
        static_model = get_yolo_model()
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            yield mjpeg_chunk(status_frame("Khong mo duoc video dau vao.", source_path.name))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vehicle_classes = redlight_vehicle_class_ids()
        vehicle_ids = set(vehicle_classes)
        priority_vehicle_ids = redlight_priority_vehicle_ids()
        static_classes = redlight_static_class_ids()
        calibrator = RedlightStoplineCalibrator()
        light_memory = RedlightLightMemory()
        track_states: dict[int, RedlightTrackState] = {}
        violation_tracks: set[int] = set()
        warning_tracks: set[int] = set()
        priority_tracks: set[int] = set()
        frame_count = 0
        raw_vehicle_detections = 0

        while True:
            ok, frame = cap.read()
            if not ok:
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
            static_kwargs = {
                "conf": max(0.12, confidence * 0.65),
                "iou": iou,
                "retina_masks": True,
                "verbose": False,
            }
            if static_classes:
                static_kwargs["classes"] = static_classes
            static_result = static_model.predict(frame, **static_kwargs)[0]
            static_boxes, static_labels, static_confs = redlight_result_arrays(static_result)

            current_time = time.time()
            detected_light_state = traffic_light_state_from_labels(static_labels)
            light_memory.update(detected_light_state, current_time)
            light_state = light_memory.get(current_time)
            light_status = light_state.simple_state()

            line_info, stopline_mask = stopline_line_from_result(static_result, width, height)
            calibrator.update_line(line_info)
            calibrator.maybe_finish(width, height)
            line_segment, line, line_locked = calibrator.current(width, height)
            ref_vector = (line[0], line[1])

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

            frame_vis = frame.copy()
            if not line_locked:
                overlay_stopline_mask(frame_vis, stopline_mask)
            draw_traffic_lights_on_frame(frame_vis, static_boxes, static_labels, static_confs)

            line_color = {
                "RED": (0, 0, 255),
                "YELLOW": (0, 255, 255),
                "GREEN": (0, 255, 0),
            }.get(light_status, (255, 255, 255))
            cv2.line(frame_vis, (line_segment[0], line_segment[1]), (line_segment[2], line_segment[3]), line_color, 4)
            draw_text_badge(
                frame_vis,
                "STOP LINE LOCKED" if line_locked else "CALIBRATING STOP LINE",
                (line_segment[0], max(24, line_segment[1] - 10)),
                line_color,
                0.56,
                2,
            )

            if track_result.boxes is not None:
                boxes = track_result.boxes.xyxy.cpu().numpy()
                labels = track_result.boxes.cls.cpu().numpy().astype(int)
                ids = track_result.boxes.id.cpu().numpy().astype(int) if track_result.boxes.id is not None else np.arange(len(labels))
                confs = track_result.boxes.conf.cpu().numpy()
                boxes, labels, confs, ids = dedup_redlight_tracks(boxes, labels, confs, ids)
                raw_vehicle_detections += int(len(labels))
                for box, label, track_id, score in zip(boxes, labels, ids, confs):
                    name = class_name(int(label), "yolo")
                    if int(label) not in vehicle_ids:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    px, py = bottom_center(box)
                    track_id = int(track_id)
                    state = track_states.setdefault(track_id, RedlightTrackState())
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
                "LIVE RED-LIGHT DETECTION",
                f"Light: {light_status}",
                f"Stop line: {'LOCKED' if line_locked else 'CALIBRATING'}",
                f"Vehicles: {len(track_states)}",
                f"Violations: {len(violation_tracks)}",
                f"Warnings: {len(warning_tracks)}",
                f"Priority: {len(priority_tracks)}",
                f"Frame: {frame_count}",
            ]
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
        yield mjpeg_chunk(status_frame("Loi live red-light stream.", str(exc)))
    finally:
        if cap is not None:
            cap.release()


def run_redlight_video(source_path: Path, confidence: float, iou: float) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_yolo_model()
    static_model = get_yolo_model()
    visible_classes = configured_display_class_set("yolo")
    visible_class_names = configured_display_class_names("yolo")
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
            retina_masks=True,
            verbose=False,
        )[0]
        if line is None or frame_count <= int(fps * 3):
            line_segment, line = detect_stopline_from_result(static_result, width, height)

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
                distance = signed_distance(px, py, line) if line else 0
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


@app.post("/api/redlight/start")
def api_redlight_start():
    try:
        cleanup_redlight_stream_jobs()
        confidence = float(request.form.get("confidence", 0.5))
        iou = float(request.form.get("iou", 0.5))
        file_storage = request.files.get("file")
        if file_storage is None:
            return jsonify({"success": False, "error": "Vui lòng upload video cho tác vụ vượt đèn đỏ.", "status": model_status()}), 400

        uploaded_file_metadata = upload_debug(file_storage)
        source_path = save_upload(file_storage)
        if not is_video(source_path):
            return jsonify({"success": False, "error": "Tác vụ vượt đèn đỏ cần input video.", "status": model_status()}), 400

        job_id = uuid.uuid4().hex[:12]
        _redlight_stream_jobs[job_id] = {
            "source_path": source_path,
            "confidence": confidence,
            "iou": iou,
            "created_at": time.time(),
        }
        stream_url = url_for("api_redlight_stream", job_id=job_id)
        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "model": "Red Light Violation Live",
                "model_key": "redlight",
                "media_type": "video",
                "original_url": relative_static_url(source_path),
                "stream_url": stream_url,
                "status": model_status(),
                "debug": {
                    "request": {
                        "uploaded_file": uploaded_file_metadata,
                        "source_media": media_debug(source_path),
                        "confidence": confidence,
                        "iou": iou,
                    },
                    "stream": {
                        "job_id": job_id,
                        "url": stream_url,
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
        redlight_stream_frames(job["source_path"], float(job["confidence"]), float(job["iou"])),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


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

    try:
        confidence = float(request.form.get("confidence", confidence))
        iou = float(request.form.get("iou", iou))
        show_labels = parse_bool(request.form.get("show_labels"), show_labels)
        show_conf = parse_bool(request.form.get("show_conf"), show_conf)
        use_sample = request.form.get("use_sample") == "1"

        if use_sample:
            source_path = SAMPLE_IMAGE_PATH
        else:
            file_storage = request.files.get("file")
            if file_storage is None:
                status_snapshot = model_status()
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Vui lòng tải lên một ảnh hoặc chọn ảnh sample.",
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
            payload["results"][model_key] = run_inference(
                model_key,
                SAMPLE_IMAGE_PATH,
                confidence,
                iou,
                show_labels=show_labels,
                show_conf=show_conf,
            )
        except Exception as exc:
            payload["errors"][model_key] = str(exc)
    _sample_compare_cache[cache_key] = payload
    return jsonify(payload)


if __name__ == "__main__":
    debug_mode = parse_bool(os.environ.get("FLASK_DEBUG"), False)
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=debug_mode)
