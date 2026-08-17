from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


# ============================================================
# 使用者設定：平常只需要修改這一區
# ============================================================

# 要處理的原始 Task。
# 裡面必須有 rgb、depth。
TASK_DIR = r"data\dataset\weldboard_l\task_0021"

# 留空字串 "" 時，自動尋找最新修改的 best.pt。
# 也可以直接指定：
# MODEL_PATH = r"runs\smart_teaching_tip\模型名稱\weights\best.pt"
MODEL_PATH = ""

# 輸出根目錄。
OUTPUT_ROOT = r"runs\infer_yolo_depth_fusion"

# ------------------------------------------------------------
# YOLO 設定
# ------------------------------------------------------------

DEVICE = "0"
IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 1

# 信心高於此數值時，直接相信當前 YOLO。
YOLO_HIGH_CONFIDENCE = 0.70

# Depth 最多只能覆蓋低於此信心的 YOLO 結果。
DEPTH_OVERRIDE_MAX_CONFIDENCE = 0.55

# 狀態多數決視窗，必須是奇數。
YOLO_STATE_WINDOW = 5

# 尖端像素位置中位數視窗。
POINT_FILTER_WINDOW = 5

# YOLO 漏檢時保持上一狀態的幀數。
MAX_HOLD_FRAMES = 3

# ------------------------------------------------------------
# Depth 設定
# ------------------------------------------------------------

MIN_DEPTH_MM = 150
MAX_DEPTH_MM = 3000

# 尖端附近深度區域。
TIP_DEPTH_WINDOW = 11

# 尖端區域至少需要多少個有效 Depth 像素。
MIN_TIP_DEPTH_PIXELS = 8

# 使用有效深度中較靠近相機的一部分。
# 20 代表取第 20 百分位。
TIP_FRONT_PERCENTILE = 20.0

# 尖端周圍的環形區域，用來估算板面。
BOARD_OUTER_WINDOW = 81
BOARD_INNER_WINDOW = 25

# 環形區域至少需要的有效板面深度點。
MIN_LOCAL_BOARD_PIXELS = 120

# 局部板面失敗時，使用全畫面 ROI 備援。
# 順序為：左、上、右、下，數值介於 0～1。
BOARD_ROI = (0.10, 0.15, 0.90, 0.95)

MIN_GLOBAL_BOARD_PIXELS = 500

# 找主要深度平面的直方圖設定。
DEPTH_BIN_MM = 5.0
DEPTH_CLUSTER_BAND_MM = 15.0

# 尖端相對板面合理的高度範圍。
MAX_REASONABLE_TIP_HEIGHT_MM = 80.0
MAX_NEGATIVE_HEIGHT_MM = 8.0

# 高度判斷門檻。
# 高度很小只能用來「確認」DOWN，不會強制把 UP 改成 DOWN。
DOWN_MAX_HEIGHT_MM = 4.0

# 明確高於板面才可用 Depth 輔助判斷 UP。
UP_MIN_HEIGHT_MM = 12.0

# Depth 時間濾波。
DEPTH_TEMPORAL_WINDOW = 5
MIN_DEPTH_TEMPORAL_SAMPLES = 3

# 高度在最近幾幀波動超過此值，Depth 就不參與決策。
MAX_HEIGHT_MAD_MM = 5.0

# RGB 和 Depth 尺寸不同時，是否暫時用比例換算像素。
# 這不是正式 D2C 幾何對齊，只適合目前測試。
ALLOW_SCALED_DEPTH = True

# ------------------------------------------------------------
# 評估與影片設定
# ------------------------------------------------------------

# 有人工標註時，尖端誤差低於此值才顯示 PASS。
PASS_PIXEL_ERROR = 12.0

OUTPUT_FPS = 6.0  # 舊資料預設；有 metadata.json 時會自動判斷
SHOW_WINDOW = True
SAVE_VIDEO = True

# 畫出 Depth 掃描區域，方便觀察。
DRAW_DEPTH_REGIONS = True


# ============================================================
# 固定欄位，一般不需要修改
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

CSV_FIELDS = [
    "frame_id",
    "timestamp_s",
    "image_name",
    "rgb_path",
    "depth_path",

    "raw_state",
    "raw_confidence",
    "temporal_state",
    "depth_state",
    "final_state",
    "state_source",

    "x_px_raw",
    "y_px_raw",
    "x_px_filtered",
    "y_px_filtered",

    "depth_mapping",
    "board_depth_mm",
    "board_depth_source",
    "board_valid_pixels",
    "tip_depth_mm",
    "tip_valid_pixels",
    "height_raw_mm",
    "height_filtered_mm",
    "height_mad_mm",
    "depth_reliable",

    "gt_state",
    "gt_x",
    "gt_y",
    "state_correct",
    "point_error_px",
    "result",

    "inference_ms",
]


# ============================================================
# 路徑與資料工具
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="對單一 data/dataset/<category>/task_XXXX 執行 YOLO 與深度融合。"
    )
    parser.add_argument("--task-dir", type=Path, default=Path(TASK_DIR))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(MODEL_PATH) if MODEL_PATH.strip() else None,
        help="未指定時才自動選擇最新修改的 best.pt。",
    )
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--fps", type=float, default=None, help="覆寫輸出 FPS；未指定時由 Task metadata.json 自動判斷")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_WINDOW)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=SAVE_VIDEO)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def find_latest_model() -> Path:
    model_root = (
        PROJECT_ROOT
        / "runs"
        / "smart_teaching_tip"
    )

    candidates = [
        path
        for path in model_root.glob(
            "**/weights/best.pt"
        )
        if "evaluation" not in str(path).lower()
    ]

    if not candidates:
        raise SystemExit(
            "❌ 找不到 best.pt。\n"
            "請在程式最上方設定 MODEL_PATH。"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def find_rgb_images(rgb_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in rgb_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=natural_key,
    )


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def load_ground_truth(
    task_dir: Path,
) -> dict[str, dict[str, Any]]:
    label_path = (
        task_dir
        / "annotation"
        / "tip_state_labels.csv"
    )

    if not label_path.is_file():
        print(
            "⚠️ 找不到人工標註，"
            "影片只顯示推論結果，不顯示誤差。"
        )
        return {}

    labels: dict[str, dict[str, Any]] = {}

    with label_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = Path(
                str(row.get("image", ""))
            ).name

            if not image_name:
                continue

            labels[image_name] = {
                "state": str(
                    row.get("state", "NONE")
                ).upper(),
                "x": to_float(row.get("x")),
                "y": to_float(row.get("y")),
            }

    print(
        f"✅ 已讀取人工標註：{len(labels)} 張"
    )

    return labels


def find_depth_path(
    depth_dir: Path,
    rgb_path: Path,
) -> Path | None:
    candidates = [
        depth_dir / f"{rgb_path.stem}.png",
        depth_dir / f"{rgb_path.stem}.tiff",
        depth_dir / f"{rgb_path.stem}.tif",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_depth(
    path: Path | None,
) -> np.ndarray | None:
    if path is None:
        return None

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        return None

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


# ============================================================
# YOLO 與時間濾波
# ============================================================

def weighted_state_vote(
    history: deque[tuple[str, float]],
) -> str:
    score = {
        "UP": 0.0,
        "DOWN": 0.0,
    }

    for state, confidence in history:
        if state in score:
            score[state] += max(
                float(confidence),
                0.01,
            )

    if (
        score["UP"] == 0
        and score["DOWN"] == 0
    ):
        return "NONE"

    return max(
        score,
        key=score.get,
    )


def filtered_point(
    history: deque[
        tuple[float, float] | None
    ],
) -> tuple[float | None, float | None]:
    points = [
        point
        for point in history
        if point is not None
    ]

    if not points:
        return None, None

    x_values = [
        point[0]
        for point in points
    ]

    y_values = [
        point[1]
        for point in points
    ]

    return (
        float(np.median(x_values)),
        float(np.median(y_values)),
    )


# ============================================================
# Depth 處理
# ============================================================

def valid_depth_values(
    values: np.ndarray,
) -> np.ndarray:
    values = values.astype(
        np.float64,
        copy=False,
    )

    return values[
        (values >= MIN_DEPTH_MM)
        & (values <= MAX_DEPTH_MM)
    ]


def map_rgb_to_depth(
    x: float,
    y: float,
    rgb_width: int,
    rgb_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[int, int, str]:
    if (
        rgb_width == depth_width
        and rgb_height == depth_height
    ):
        depth_x = int(round(x))
        depth_y = int(round(y))
        mapping = "same_size"

    else:
        depth_x = int(
            round(
                x
                * max(depth_width - 1, 1)
                / max(rgb_width - 1, 1)
            )
        )

        depth_y = int(
            round(
                y
                * max(depth_height - 1, 1)
                / max(rgb_height - 1, 1)
            )
        )

        mapping = "scaled_pixel_approx"

    depth_x = int(
        np.clip(
            depth_x,
            0,
            depth_width - 1,
        )
    )

    depth_y = int(
        np.clip(
            depth_y,
            0,
            depth_height - 1,
        )
    )

    return depth_x, depth_y, mapping


def dominant_depth_cluster(
    values: np.ndarray,
    minimum_pixels: int,
) -> tuple[float | None, int]:
    values = valid_depth_values(values)

    if values.size < minimum_pixels:
        return None, int(values.size)

    bins = np.arange(
        MIN_DEPTH_MM,
        MAX_DEPTH_MM + DEPTH_BIN_MM,
        DEPTH_BIN_MM,
    )

    histogram, edges = np.histogram(
        values,
        bins=bins,
    )

    peak_index = int(
        np.argmax(histogram)
    )

    peak_count = int(
        histogram[peak_index]
    )

    if peak_count <= 0:
        return None, 0

    peak_center = (
        edges[peak_index]
        + edges[peak_index + 1]
    ) / 2.0

    cluster = values[
        np.abs(
            values - peak_center
        )
        <= DEPTH_CLUSTER_BAND_MM
    ]

    if cluster.size < minimum_pixels:
        return None, int(cluster.size)

    return (
        float(np.median(cluster)),
        int(cluster.size),
    )


def estimate_board_depth(
    depth: np.ndarray,
    depth_x: int,
    depth_y: int,
) -> tuple[
    float | None,
    str,
    int,
]:
    depth_height, depth_width = (
        depth.shape[:2]
    )

    outer_radius = (
        BOARD_OUTER_WINDOW // 2
    )

    inner_radius = (
        BOARD_INNER_WINDOW // 2
    )

    x0 = max(
        0,
        depth_x - outer_radius,
    )

    x1 = min(
        depth_width,
        depth_x + outer_radius + 1,
    )

    y0 = max(
        0,
        depth_y - outer_radius,
    )

    y1 = min(
        depth_height,
        depth_y + outer_radius + 1,
    )

    patch = depth[y0:y1, x0:x1]

    yy, xx = np.ogrid[
        :patch.shape[0],
        :patch.shape[1],
    ]

    local_x = depth_x - x0
    local_y = depth_y - y0

    distance_squared = (
        (xx - local_x) ** 2
        + (yy - local_y) ** 2
    )

    annulus_mask = (
        (
            distance_squared
            <= outer_radius ** 2
        )
        & (
            distance_squared
            >= inner_radius ** 2
        )
    )

    local_values = patch[
        annulus_mask
    ]

    local_depth, local_count = (
        dominant_depth_cluster(
            local_values,
            MIN_LOCAL_BOARD_PIXELS,
        )
    )

    if local_depth is not None:
        return (
            local_depth,
            "local_annulus",
            local_count,
        )

    left, top, right, bottom = (
        BOARD_ROI
    )

    roi_x0 = int(
        depth_width * left
    )

    roi_y0 = int(
        depth_height * top
    )

    roi_x1 = int(
        depth_width * right
    )

    roi_y1 = int(
        depth_height * bottom
    )

    global_values = depth[
        roi_y0:roi_y1,
        roi_x0:roi_x1,
    ]

    global_depth, global_count = (
        dominant_depth_cluster(
            global_values.reshape(-1),
            MIN_GLOBAL_BOARD_PIXELS,
        )
    )

    if global_depth is not None:
        return (
            global_depth,
            "global_roi",
            global_count,
        )

    return None, "invalid", 0


def estimate_tip_depth(
    depth: np.ndarray,
    depth_x: int,
    depth_y: int,
    board_depth: float | None,
) -> tuple[float | None, int]:
    radius = TIP_DEPTH_WINDOW // 2

    depth_height, depth_width = (
        depth.shape[:2]
    )

    x0 = max(
        0,
        depth_x - radius,
    )

    x1 = min(
        depth_width,
        depth_x + radius + 1,
    )

    y0 = max(
        0,
        depth_y - radius,
    )

    y1 = min(
        depth_height,
        depth_y + radius + 1,
    )

    values = valid_depth_values(
        depth[y0:y1, x0:x1]
    )

    if board_depth is not None:
        values = values[
            (
                values
                >= board_depth
                - MAX_REASONABLE_TIP_HEIGHT_MM
            )
            & (
                values
                <= board_depth + 20.0
            )
        ]

    if (
        values.size
        < MIN_TIP_DEPTH_PIXELS
    ):
        return None, int(values.size)

    tip_depth = float(
        np.percentile(
            values,
            TIP_FRONT_PERCENTILE,
        )
    )

    return tip_depth, int(values.size)


def measure_depth(
    depth: np.ndarray | None,
    x_px: float | None,
    y_px: float | None,
    rgb_width: int,
    rgb_height: int,
) -> dict[str, Any]:
    empty = {
        "mapping": "unavailable",
        "depth_x": None,
        "depth_y": None,
        "board_depth": None,
        "board_source": "invalid",
        "board_pixels": 0,
        "tip_depth": None,
        "tip_pixels": 0,
        "height": None,
        "raw_reliable": False,
    }

    if (
        depth is None
        or x_px is None
        or y_px is None
    ):
        return empty

    depth_height, depth_width = (
        depth.shape[:2]
    )

    depth_x, depth_y, mapping = (
        map_rgb_to_depth(
            x=x_px,
            y=y_px,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            depth_width=depth_width,
            depth_height=depth_height,
        )
    )

    (
        board_depth,
        board_source,
        board_pixels,
    ) = estimate_board_depth(
        depth,
        depth_x,
        depth_y,
    )

    tip_depth, tip_pixels = (
        estimate_tip_depth(
            depth,
            depth_x,
            depth_y,
            board_depth,
        )
    )

    height = None

    if (
        board_depth is not None
        and tip_depth is not None
    ):
        height = (
            board_depth - tip_depth
        )

    mapping_allowed = (
        mapping == "same_size"
        or ALLOW_SCALED_DEPTH
    )

    raw_reliable = (
        mapping_allowed
        and height is not None
        and height
        >= -MAX_NEGATIVE_HEIGHT_MM
        and height
        <= MAX_REASONABLE_TIP_HEIGHT_MM
    )

    return {
        "mapping": mapping,
        "depth_x": depth_x,
        "depth_y": depth_y,
        "board_depth": board_depth,
        "board_source": board_source,
        "board_pixels": board_pixels,
        "tip_depth": tip_depth,
        "tip_pixels": tip_pixels,
        "height": height,
        "raw_reliable": raw_reliable,
    }


def filtered_height(
    history: deque[float | None],
) -> tuple[
    float | None,
    float | None,
    bool,
]:
    values = np.array(
        [
            value
            for value in history
            if value is not None
        ],
        dtype=np.float64,
    )

    if (
        values.size
        < MIN_DEPTH_TEMPORAL_SAMPLES
    ):
        return None, None, False

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    reliable = (
        mad <= MAX_HEIGHT_MAD_MM
    )

    return median, mad, reliable


def height_to_state(
    height_mm: float | None,
    reliable: bool,
) -> str:
    if (
        height_mm is None
        or not reliable
    ):
        return "UNKNOWN"

    if height_mm >= UP_MIN_HEIGHT_MM:
        return "UP"

    if height_mm <= DOWN_MAX_HEIGHT_MM:
        return "DOWN"

    return "UNKNOWN"


# ============================================================
# 狀態融合
# ============================================================

def fuse_state(
    raw_state: str,
    raw_confidence: float,
    temporal_state: str,
    depth_state: str,
    depth_reliable: bool,
    last_state: str,
    missing_frames: int,
) -> tuple[str, str]:
    if (
        raw_state in {"UP", "DOWN"}
        and raw_confidence
        >= YOLO_HIGH_CONFIDENCE
    ):
        return (
            raw_state,
            "YOLO_HIGH",
        )

    if raw_state in {"UP", "DOWN"}:
        base_state = (
            temporal_state
            if temporal_state
            in {"UP", "DOWN"}
            else raw_state
        )

        if (
            depth_reliable
            and depth_state == base_state
        ):
            return (
                base_state,
                "YOLO_DEPTH_AGREE",
            )

        # Depth 明確看到尖端高於板面時，
        # 才允許它修正低信心 YOLO 為 UP。
        if (
            depth_reliable
            and depth_state == "UP"
            and base_state != "UP"
            and raw_confidence
            <= DEPTH_OVERRIDE_MAX_CONFIDENCE
        ):
            return (
                "UP",
                "DEPTH_ASSIST_UP",
            )

        # Depth 判斷 DOWN 時，不允許它單獨將 UP 改成 DOWN。
        if (
            depth_reliable
            and depth_state == "DOWN"
            and base_state == "UP"
        ):
            return (
                base_state,
                "YOLO_KEEP_DEPTH_DOWN_UNSAFE",
            )

        return (
            base_state,
            "YOLO_TEMPORAL",
        )

    if (
        last_state in {"UP", "DOWN"}
        and missing_frames
        <= MAX_HOLD_FRAMES
    ):
        return (
            last_state,
            "HOLD_PREVIOUS",
        )

    return "NONE", "NONE"


# ============================================================
# 顯示工具
# ============================================================

def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def state_color(
    state: str,
) -> tuple[int, int, int]:
    if state == "UP":
        return (0, 255, 255)

    if state == "DOWN":
        return (0, 0, 255)

    return (180, 180, 180)


def point_error(
    predicted_x: float | None,
    predicted_y: float | None,
    gt_x: float | None,
    gt_y: float | None,
) -> float | None:
    if (
        predicted_x is None
        or predicted_y is None
        or gt_x is None
        or gt_y is None
    ):
        return None

    return math.hypot(
        predicted_x - gt_x,
        predicted_y - gt_y,
    )


# ============================================================
# 主程式
# ============================================================

def main() -> int:
    global SHOW_WINDOW, SAVE_VIDEO, OUTPUT_FPS

    args = parse_args()
    SHOW_WINDOW = bool(args.show)
    SAVE_VIDEO = bool(args.save_video)
    task_dir = resolve_path(args.task_dir)
    if args.fps is not None:
        if args.fps <= 0:
            raise SystemExit("--fps 必須大於 0")
        OUTPUT_FPS = float(args.fps)
    else:
        metadata_path = task_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                recording_fps = float(metadata.get("recording_fps", 30.0))
                sample_interval = max(1, int(metadata.get("sample_interval", 1)))
                OUTPUT_FPS = recording_fps / sample_interval
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                OUTPUT_FPS = 6.0

    rgb_dir = task_dir / "rgb"
    depth_dir = task_dir / "depth"

    if not rgb_dir.is_dir():
        raise SystemExit(
            f"❌ 找不到 RGB 資料夾：\n{rgb_dir}"
        )

    if not depth_dir.is_dir():
        raise SystemExit(
            f"❌ 找不到 Depth 資料夾：\n{depth_dir}"
        )

    if args.model is not None:
        model_path = resolve_path(args.model)
    else:
        model_path = find_latest_model()

    if not model_path.is_file():
        raise SystemExit(
            f"❌ 找不到模型：\n{model_path}"
        )

    images = find_rgb_images(
        rgb_dir
    )

    if not images:
        raise SystemExit(
            f"❌ RGB 資料夾內沒有圖片：\n"
            f"{rgb_dir}"
        )

    ground_truth = load_ground_truth(
        task_dir
    )

    output_dir = (
        resolve_path(args.output_root)
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=============================="
    )
    print("YOLO + Depth 融合設定")
    print("==============================")
    print(f"Task：{task_dir}")
    print(f"Model：{model_path}")
    print(f"影像數量：{len(images)}")
    print(
        f"Depth 只輔助 UP："
        f"{UP_MIN_HEIGHT_MM:.1f} mm"
    )
    print(
        "Depth 不會單獨強制改成 DOWN"
    )
    print("==============================\n")

    try:
        from ultralytics import YOLO
        from infer_yolo import (
            extract_best_prediction,
            run_model,
        )
    except ImportError as exc:
        raise SystemExit(
            "❌ 缺少 ultralytics，"
            "或找不到同資料夾內的 "
            "infer_yolo.py。"
        ) from exc

    model = YOLO(
        str(model_path),
        task="detect",
    )

    yolo_args = SimpleNamespace(
        imgsz=IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        max_det=MAX_DETECTIONS,
    )

    state_history: deque[
        tuple[str, float]
    ] = deque(
        maxlen=YOLO_STATE_WINDOW
    )

    point_history: deque[
        tuple[float, float] | None
    ] = deque(
        maxlen=POINT_FILTER_WINDOW
    )

    height_history: deque[
        float | None
    ] = deque(
        maxlen=DEPTH_TEMPORAL_WINDOW
    )

    last_final_state = "NONE"
    missing_frames = 0

    rows: list[dict[str, Any]] = []

    video_writer = None
    video_path = (
        output_dir
        / "fusion_evaluation.mp4"
    )

    for frame_id, rgb_path in enumerate(
        images
    ):
        frame = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            print(
                f"⚠️ RGB 讀取失敗：{rgb_path}"
            )
            continue

        height, width = (
            frame.shape[:2]
        )

        result = run_model(
            model,
            frame,
            yolo_args,
        )

        prediction = extract_best_prediction(
            result=result,
            source=str(rgb_path),
            frame_id=frame_id,
            sample_id=frame_id,
            timestamp_s=(
                frame_id / OUTPUT_FPS
            ),
            width=width,
            height=height,
        )

        raw_state = str(
            prediction["state"]
        )

        raw_confidence = float(
            prediction["confidence"]
        )

        raw_x = prediction["x_px"]
        raw_y = prediction["y_px_top"]

        state_history.append(
            (
                raw_state,
                raw_confidence,
            )
        )

        temporal_state = (
            weighted_state_vote(
                state_history
            )
        )

        if (
            raw_x is not None
            and raw_y is not None
        ):
            point_history.append(
                (
                    float(raw_x),
                    float(raw_y),
                )
            )
            missing_frames = 0
        else:
            point_history.append(None)
            missing_frames += 1

        filtered_x, filtered_y = (
            filtered_point(
                point_history
            )
        )

        depth_path = find_depth_path(
            depth_dir,
            rgb_path,
        )

        depth = load_depth(
            depth_path
        )

        depth_result = measure_depth(
            depth=depth,
            x_px=filtered_x,
            y_px=filtered_y,
            rgb_width=width,
            rgb_height=height,
        )

        if depth_result["raw_reliable"]:
            height_history.append(
                float(
                    depth_result["height"]
                )
            )
        else:
            height_history.append(None)

        (
            filtered_height_mm,
            height_mad_mm,
            temporal_depth_reliable,
        ) = filtered_height(
            height_history
        )

        depth_reliable = (
            bool(
                depth_result[
                    "raw_reliable"
                ]
            )
            and temporal_depth_reliable
        )

        depth_state = height_to_state(
            filtered_height_mm,
            depth_reliable,
        )

        final_state, state_source = (
            fuse_state(
                raw_state=raw_state,
                raw_confidence=(
                    raw_confidence
                ),
                temporal_state=(
                    temporal_state
                ),
                depth_state=depth_state,
                depth_reliable=(
                    depth_reliable
                ),
                last_state=(
                    last_final_state
                ),
                missing_frames=(
                    missing_frames
                ),
            )
        )

        if final_state in {
            "UP",
            "DOWN",
        }:
            last_final_state = (
                final_state
            )

        gt = ground_truth.get(
            rgb_path.name
        )

        gt_state = (
            gt["state"]
            if gt is not None
            else None
        )

        gt_x = (
            gt["x"]
            if gt is not None
            else None
        )

        gt_y = (
            gt["y"]
            if gt is not None
            else None
        )

        state_correct = (
            final_state == gt_state
            if gt_state is not None
            else None
        )

        error_px = point_error(
            predicted_x=filtered_x,
            predicted_y=filtered_y,
            gt_x=gt_x,
            gt_y=gt_y,
        )

        if gt_state is None:
            frame_result = "N/A"

        elif gt_state == "NONE":
            frame_result = (
                "PASS"
                if final_state == "NONE"
                else "FAIL"
            )

        elif (
            state_correct
            and error_px is not None
            and error_px
            <= PASS_PIXEL_ERROR
        ):
            frame_result = "PASS"

        else:
            frame_result = "FAIL"

        row = {
            "frame_id": frame_id,
            "timestamp_s": round(frame_id / OUTPUT_FPS, 6),
            "image_name": rgb_path.name,
            "rgb_path": str(rgb_path),
            "depth_path": (
                str(depth_path)
                if depth_path is not None
                else ""
            ),

            "raw_state": raw_state,
            "raw_confidence": (
                raw_confidence
            ),
            "temporal_state": (
                temporal_state
            ),
            "depth_state": depth_state,
            "final_state": final_state,
            "state_source": state_source,

            "x_px_raw": raw_x,
            "y_px_raw": raw_y,
            "x_px_filtered": filtered_x,
            "y_px_filtered": filtered_y,

            "depth_mapping": (
                depth_result["mapping"]
            ),
            "board_depth_mm": (
                depth_result[
                    "board_depth"
                ]
            ),
            "board_depth_source": (
                depth_result[
                    "board_source"
                ]
            ),
            "board_valid_pixels": (
                depth_result[
                    "board_pixels"
                ]
            ),
            "tip_depth_mm": (
                depth_result["tip_depth"]
            ),
            "tip_valid_pixels": (
                depth_result["tip_pixels"]
            ),
            "height_raw_mm": (
                depth_result["height"]
            ),
            "height_filtered_mm": (
                filtered_height_mm
            ),
            "height_mad_mm": (
                height_mad_mm
            ),
            "depth_reliable": (
                depth_reliable
            ),

            "gt_state": gt_state,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "state_correct": (
                state_correct
            ),
            "point_error_px": error_px,
            "result": frame_result,

            "inference_ms": (
                prediction[
                    "inference_ms"
                ]
            ),
        }

        rows.append(row)

        annotated = frame.copy()

        if (
            prediction["box_x1"]
            is not None
        ):
            cv2.rectangle(
                annotated,
                (
                    int(
                        prediction[
                            "box_x1"
                        ]
                    ),
                    int(
                        prediction[
                            "box_y1"
                        ]
                    ),
                ),
                (
                    int(
                        prediction[
                            "box_x2"
                        ]
                    ),
                    int(
                        prediction[
                            "box_y2"
                        ]
                    ),
                ),
                state_color(raw_state),
                2,
                cv2.LINE_AA,
            )

        if (
            filtered_x is not None
            and filtered_y is not None
        ):
            predicted_point = (
                int(round(filtered_x)),
                int(round(filtered_y)),
            )

            cv2.drawMarker(
                annotated,
                predicted_point,
                state_color(final_state),
                cv2.MARKER_CROSS,
                24,
                2,
            )

        if (
            gt_x is not None
            and gt_y is not None
        ):
            gt_point = (
                int(round(gt_x)),
                int(round(gt_y)),
            )

            cv2.drawMarker(
                annotated,
                gt_point,
                (0, 255, 0),
                cv2.MARKER_TILTED_CROSS,
                24,
                2,
            )

            if (
                filtered_x is not None
                and filtered_y is not None
            ):
                cv2.line(
                    annotated,
                    predicted_point,
                    gt_point,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if (
            DRAW_DEPTH_REGIONS
            and depth_result["depth_x"]
            is not None
            and filtered_x is not None
            and filtered_y is not None
        ):
            # RGB 畫面上只顯示大致區域。
            cv2.circle(
                annotated,
                (
                    int(round(filtered_x)),
                    int(round(filtered_y)),
                ),
                TIP_DEPTH_WINDOW // 2,
                (255, 0, 255),
                1,
            )

            cv2.circle(
                annotated,
                (
                    int(round(filtered_x)),
                    int(round(filtered_y)),
                ),
                BOARD_OUTER_WINDOW // 2,
                (255, 0, 0),
                1,
            )

        draw_text(
            annotated,
            (
                f"RAW={raw_state} "
                f"{raw_confidence:.3f}"
            ),
            15,
            28,
        )

        draw_text(
            annotated,
            (
                f"FINAL={final_state} "
                f"[{state_source}]"
            ),
            15,
            54,
            state_color(final_state),
        )

        if filtered_height_mm is None:
            height_text = "N/A"
        else:
            height_text = (
                f"{filtered_height_mm:.1f} mm"
            )

        draw_text(
            annotated,
            (
                f"DEPTH={depth_state} "
                f"height={height_text} "
                f"reliable={depth_reliable}"
            ),
            15,
            80,
        )

        if gt_state is not None:
            if error_px is None:
                error_text = "N/A"
            else:
                error_text = (
                    f"{error_px:.1f} px"
                )

            result_color = (
                (0, 255, 0)
                if frame_result == "PASS"
                else (0, 0, 255)
            )

            draw_text(
                annotated,
                (
                    f"GT={gt_state} "
                    f"state_ok={state_correct} "
                    f"error={error_text} "
                    f"{frame_result}"
                ),
                15,
                106,
                result_color,
            )

        draw_text(
            annotated,
            (
                f"frame={frame_id} "
                f"mapping="
                f"{depth_result['mapping']}"
            ),
            15,
            height - 18,
        )

        if (
            SAVE_VIDEO
            and video_writer is None
        ):
            video_writer = (
                cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    ),
                    OUTPUT_FPS,
                    (width, height),
                )
            )

            if not video_writer.isOpened():
                raise SystemExit(
                    "❌ 無法建立輸出影片。"
                )

        if video_writer is not None:
            video_writer.write(
                annotated
            )

        if SHOW_WINDOW:
            cv2.imshow(
                "YOLO + Depth Fusion",
                annotated,
            )

            key = (
                cv2.waitKey(
                    max(
                        1,
                        int(
                            1000
                            / OUTPUT_FPS
                        ),
                    )
                )
                & 0xFF
            )

            if key in {
                ord("q"),
                27,
            }:
                break

        print(
            f"[{frame_id + 1:>4}/"
            f"{len(images)}] "
            f"raw={raw_state:<4} "
            f"final={final_state:<4} "
            f"source={state_source:<28} "
            f"height={height_text:<10} "
            f"result={frame_result}"
        )

    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()

    csv_path = (
        output_dir
        / "fusion_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    evaluated_rows = [
        row
        for row in rows
        if row["gt_state"] is not None
    ]

    correct_rows = [
        row
        for row in evaluated_rows
        if row["state_correct"] is True
    ]

    error_values = np.array(
        [
            row["point_error_px"]
            for row in evaluated_rows
            if row["point_error_px"]
            is not None
        ],
        dtype=np.float64,
    )

    per_class: dict[str, Any] = {}

    for state in (
        "UP",
        "DOWN",
        "NONE",
    ):
        state_rows = [
            row
            for row in evaluated_rows
            if row["gt_state"] == state
        ]

        state_correct_count = sum(
            row["state_correct"] is True
            for row in state_rows
        )

        per_class[state] = {
            "samples": len(state_rows),
            "correct": (
                state_correct_count
            ),
            "accuracy": (
                state_correct_count
                / len(state_rows)
                if state_rows
                else None
            ),
        }

    summary = {
        "task_dir": str(task_dir),
        "model": str(model_path),
        "total_frames": len(rows),
        "output_fps": OUTPUT_FPS,

        "evaluated_frames": (
            len(evaluated_rows)
        ),
        "state_correct_frames": (
            len(correct_rows)
        ),
        "state_accuracy": (
            len(correct_rows)
            / len(evaluated_rows)
            if evaluated_rows
            else None
        ),

        "per_class": per_class,

        "depth_reliable_frames": sum(
            bool(row["depth_reliable"])
            for row in rows
        ),

        "depth_assist_up_frames": sum(
            row["state_source"]
            == "DEPTH_ASSIST_UP"
            for row in rows
        ),

        "depth_agreement_frames": sum(
            row["state_source"]
            == "YOLO_DEPTH_AGREE"
            for row in rows
        ),

        "mean_point_error_px": (
            float(np.mean(error_values))
            if error_values.size
            else None
        ),

        "median_point_error_px": (
            float(np.median(error_values))
            if error_values.size
            else None
        ),

        "p95_point_error_px": (
            float(
                np.percentile(
                    error_values,
                    95,
                )
            )
            if error_values.size
            else None
        ),

        "pass_pixel_error": (
            PASS_PIXEL_ERROR
        ),

        "important_rule": (
            "Depth cannot force UP to DOWN. "
            "Depth only confirms DOWN or "
            "assists low-confidence UP."
        ),
    }

    summary_path = (
        output_dir
        / "fusion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ 融合推論完成")
    print(f"輸出資料夾：{output_dir}")
    print(f"CSV：{csv_path}")
    print(f"Summary：{summary_path}")

    if SAVE_VIDEO:
        print(f"影片：{video_path}")

    if summary["state_accuracy"] is not None:
        print(
            "狀態正確率："
            f"{summary['state_accuracy']:.2%}"
        )

    if (
        summary[
            "median_point_error_px"
        ]
        is not None
    ):
        print(
            "尖端中位誤差："
            f"{summary['median_point_error_px']:.2f} px"
        )

    print(
        "Depth 可靠幀數："
        f"{summary['depth_reliable_frames']}"
        f"/{summary['total_frames']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
