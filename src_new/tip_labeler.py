from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# 基本設定
# ============================================================

WINDOW_NAME = "Smart Teaching Tip Labeler"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

DEPTH_EXTENSIONS = {
    ".png",
    ".tif",
    ".tiff",
    ".npy",
    ".exr",
}

NUMBER_PATTERN = re.compile(r"(\d+)")

DEFAULT_TASK_DIR: str | None = None

VALID_STATES = {
    "UP",
    "DOWN",
    "NONE",
}

# 右側資訊欄寬度
INFO_PANEL_WIDTH = 330

# 局部放大鏡大小
MAGNIFIER_SIZE = 240

# 放大鏡擷取範圍半徑
MAGNIFIER_SOURCE_RADIUS = 25

# 深度中位數使用 5×5 範圍
DEPTH_MEDIAN_RADIUS = 2

# 初始顯示倍率
DEFAULT_ZOOM = 1.0

# 最小與最大顯示倍率
MIN_ZOOM = 1.0
MAX_ZOOM = 3.0

# 滑鼠滾輪每次縮放倍率
ZOOM_STEP = 0.25


# ============================================================
# 標註資料格式
# ============================================================

@dataclass
class Annotation:
    """
    Smart Teaching 專案使用的標註格式。

    座標原點：
        左上角

    x：
        往右增加

    y：
        往下增加
    """

    image: str
    state: str

    x: int | None
    y: int | None

    x_norm: float | None
    y_norm: float | None

    width: int
    height: int

    completed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# 通用工具
# ============================================================

def natural_key(value: str | Path) -> tuple[Any, ...]:
    """
    讓 frame_2 排在 frame_10 前面。
    """

    text = value.name if isinstance(value, Path) else value

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in NUMBER_PATTERN.split(text)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smart Teaching 焊槍尖端與 "
            "UP / DOWN / NONE 狀態標註工具"
        )
    )

    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help=(
            "task_XXXX 資料夾。"
            "未指定時會跳出資料夾選擇視窗。"
        ),
    )

    parser.add_argument(
        "--depth-dir-name",
        default="depth",
        help=(
            "task 資料夾內的深度資料夾名稱，"
            "預設為 depth。"
        ),
    )

    return parser.parse_args()


def choose_task_directory() -> Path:
    """
    使用 Tkinter 選擇 task_XXXX 資料夾。
    """

    try:
        import tkinter as tk
        from tkinter import filedialog

    except ImportError as exc:
        raise SystemExit(
            "無法載入 tkinter，"
            "請使用 --task-dir 指定資料夾。"
        ) from exc

    root = tk.Tk()

    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askdirectory(
        title="選擇要標註的 task_XXXX 資料夾"
    )

    root.destroy()

    if not selected:
        raise SystemExit(
            "尚未選擇資料夾，程式已結束。"
        )

    return Path(selected)


def resolve_task_directory(
    task_arg: Path | None,
) -> Path:
    """
    取得並驗證 task 資料夾。
    """

    task_dir = task_arg or (
        Path(DEFAULT_TASK_DIR)
        if DEFAULT_TASK_DIR
        else choose_task_directory()
    )

    task_dir = task_dir.expanduser().resolve()

    rgb_dir = task_dir / "rgb"

    if not rgb_dir.is_dir():
        raise SystemExit(
            f"所選資料夾內找不到 rgb：{rgb_dir}"
        )

    return task_dir


def find_images(
    rgb_dir: Path,
) -> list[Path]:
    """
    讀取 RGB 資料夾內所有圖片。
    """

    images = sorted(
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

    if not images:
        raise SystemExit(
            f"RGB 資料夾內沒有可標註圖片：{rgb_dir}"
        )

    return images


# ============================================================
# RGB 與 Depth 讀取
# ============================================================

def load_rgb(
    path: Path,
) -> np.ndarray:
    """
    讀取 RGB 圖片。
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"RGB 圖片讀取失敗：{path}"
        )

    return image


def find_depth_path(
    depth_dir: Path,
    rgb_path: Path,
) -> Path | None:
    """
    尋找與 RGB 圖片相對應的深度檔案。

    支援：
        frame_0001.png
        frame_0001.npy
        frame_0001.tiff

    也支援：
        rgb_0001.png
        depth_0001.png
    """

    if not depth_dir.is_dir():
        return None

    # 優先嘗試完全相同檔名
    exact_path = depth_dir / rgb_path.name

    if exact_path.is_file():
        return exact_path

    # 嘗試相同 stem、不同副檔名
    for suffix in DEPTH_EXTENSIONS:
        candidate = (
            depth_dir
            / f"{rgb_path.stem}{suffix}"
        )

        if candidate.is_file():
            return candidate

    # 嘗試使用最後一組數字配對
    rgb_numbers = NUMBER_PATTERN.findall(
        rgb_path.stem
    )

    if rgb_numbers:
        last_number = rgb_numbers[-1]

        matches = [
            path
            for path in depth_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in DEPTH_EXTENSIONS
                and last_number in path.stem
            )
        ]

        if len(matches) == 1:
            return matches[0]

    return None


def load_depth(
    path: Path | None,
    target_size: tuple[int, int],
) -> np.ndarray | None:
    """
    讀取原始深度資料。

    target_size：
        (width, height)

    深度資料會保留為 float32，
    不會為了顯示而修改原始值。
    """

    if path is None:
        return None

    try:
        if path.suffix.lower() == ".npy":
            depth = np.load(path)

        else:
            depth = cv2.imread(
                str(path),
                cv2.IMREAD_UNCHANGED,
            )

    except Exception as exc:
        print(
            f"⚠️ 深度檔案讀取失敗："
            f"{path}，原因：{exc}"
        )

        return None

    if depth is None:
        return None

    # 若為三通道，只取第一通道
    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = np.asarray(depth)

    if depth.dtype.kind not in {
        "u",
        "i",
        "f",
    }:
        print(
            f"⚠️ 不支援的深度資料型態："
            f"{depth.dtype}"
        )

        return None

    depth = depth.astype(
        np.float32,
        copy=False,
    )

    target_width, target_height = (
        target_size
    )

    # RGB 和 Depth 尺寸不同時，
    # 使用最近鄰插值調整。
    if (
        depth.shape[1] != target_width
        or depth.shape[0] != target_height
    ):
        depth = cv2.resize(
            depth,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return depth


# ============================================================
# Depth 顯示與讀值
# ============================================================

def depth_to_colormap(
    depth: np.ndarray | None,
    size: tuple[int, int],
) -> np.ndarray:
    """
    將原始深度資料轉成彩色圖。

    只影響畫面顯示，
    不修改原始深度值。
    """

    width, height = size

    if depth is None:
        view = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        draw_text(
            view,
            "Depth unavailable",
            (20, 42),
            scale=0.75,
            thickness=2,
        )

        return view

    valid_mask = (
        np.isfinite(depth)
        & (depth > 0)
    )

    if not np.any(valid_mask):
        view = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        draw_text(
            view,
            "No valid depth",
            (20, 42),
            scale=0.75,
            thickness=2,
        )

        return view

    valid_values = depth[valid_mask]

    # 排除極少數離群值
    near_value = float(
        np.percentile(
            valid_values,
            2,
        )
    )

    far_value = float(
        np.percentile(
            valid_values,
            98,
        )
    )

    if far_value <= near_value:
        far_value = near_value + 1.0

    normalized = np.zeros_like(
        depth,
        dtype=np.float32,
    )

    normalized[valid_mask] = np.clip(
        (
            depth[valid_mask]
            - near_value
        )
        / (
            far_value
            - near_value
        ),
        0.0,
        1.0,
    )

    # 近距離顯示為較暖色
    gray = (
        255.0
        * (
            1.0
            - normalized
        )
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )

    colored[~valid_mask] = 0

    return colored


def get_depth_value(
    depth: np.ndarray | None,
    x: int,
    y: int,
) -> float | None:
    """
    取得指定像素的單點深度。
    """

    if depth is None:
        return None

    if not (
        0 <= y < depth.shape[0]
        and 0 <= x < depth.shape[1]
    ):
        return None

    value = float(
        depth[y, x]
    )

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        return None

    return value


def get_median_depth(
    depth: np.ndarray | None,
    x: int,
    y: int,
    radius: int = DEPTH_MEDIAN_RADIUS,
) -> float | None:
    """
    取得游標周圍的有效深度中位數。

    radius=2 時：
        使用 5×5 範圍。
    """

    if depth is None:
        return None

    x_start = max(
        0,
        x - radius,
    )

    x_end = min(
        depth.shape[1],
        x + radius + 1,
    )

    y_start = max(
        0,
        y - radius,
    )

    y_end = min(
        depth.shape[0],
        y + radius + 1,
    )

    patch = depth[
        y_start:y_end,
        x_start:x_end,
    ]

    valid_values = patch[
        np.isfinite(patch)
        & (patch > 0)
    ]

    if valid_values.size == 0:
        return None

    return float(
        np.median(valid_values)
    )


def format_depth(
    value: float | None,
) -> str:
    """
    將深度轉成顯示文字。
    """

    if value is None:
        return "--"

    return f"{value:.1f} mm"


# ============================================================
# 標註檔讀寫
# ============================================================

def validate_annotation(
    item: dict[str, Any],
) -> tuple[bool, str]:
    """
    驗證標註格式。

    UP / DOWN：
        必須包含尖端座標。

    NONE：
        不可包含尖端座標。
    """

    state = item.get("state")

    if state not in VALID_STATES:
        return (
            False,
            "State 必須是 UP、DOWN 或 NONE",
        )

    x = item.get("x")
    y = item.get("y")

    has_point = (
        x is not None
        and y is not None
    )

    if (
        state in {"UP", "DOWN"}
        and not has_point
    ):
        return (
            False,
            f"{state} 必須標註尖端",
        )

    if (
        state == "NONE"
        and has_point
    ):
        return (
            False,
            "NONE 不可包含尖端座標",
        )

    width = item.get("width")
    height = item.get("height")

    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return (
            False,
            "影像寬高無效",
        )

    if has_point:
        point_x = int(x)
        point_y = int(y)

        if not (
            0 <= point_x < width
            and 0 <= point_y < height
        ):
            return (
                False,
                "尖端座標超出影像範圍",
            )

    return True, "OK"


def load_existing_annotations(
    jsonl_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    讀取既有 JSONL 標註。
    """

    annotations: dict[
        str,
        dict[str, Any],
    ] = {}

    if not jsonl_path.exists():
        return annotations

    lines = jsonl_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = json.loads(line)

        except json.JSONDecodeError as exc:
            print(
                f"⚠️ 忽略 JSONL "
                f"第 {line_number} 行：{exc}"
            )

            continue

        if (
            not isinstance(item, dict)
            or not isinstance(
                item.get("image"),
                str,
            )
        ):
            print(
                f"⚠️ 忽略 JSONL "
                f"第 {line_number} 行："
                f"缺少 image"
            )

            continue

        valid, reason = (
            validate_annotation(item)
        )

        if not valid:
            print(
                f"⚠️ 忽略 "
                f"{item.get('image')}："
                f"{reason}"
            )

            continue

        annotations[
            item["image"]
        ] = item

    return annotations


def save_annotations(
    annotations: dict[
        str,
        dict[str, Any],
    ],
    jsonl_path: Path,
    csv_path: Path,
) -> None:
    """
    同步儲存 JSONL 與 CSV。

    使用暫存檔後再取代，
    避免程式意外關閉造成檔案損壞。
    """

    ordered_names = sorted(
        annotations,
        key=natural_key,
    )

    fields = [
        "image",
        "state",
        "x",
        "y",
        "x_norm",
        "y_norm",
        "width",
        "height",
        "completed",
    ]

    temp_jsonl = jsonl_path.with_suffix(
        ".jsonl.tmp"
    )

    temp_csv = csv_path.with_suffix(
        ".csv.tmp"
    )

    with temp_jsonl.open(
        "w",
        encoding="utf-8",
    ) as file:
        for image_name in ordered_names:
            file.write(
                json.dumps(
                    annotations[image_name],
                    ensure_ascii=False,
                )
                + "\n"
            )

    with temp_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for image_name in ordered_names:
            writer.writerow(
                {
                    field: annotations[
                        image_name
                    ].get(field)
                    for field in fields
                }
            )

    temp_jsonl.replace(
        jsonl_path
    )

    temp_csv.replace(
        csv_path
    )


def make_point_annotation(
    image_name: str,
    state: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Annotation:
    """
    建立 UP 或 DOWN 尖端標註。
    """

    item = Annotation(
        image=image_name,
        state=state,
        x=x,
        y=y,
        x_norm=x / width,
        y_norm=y / height,
        width=width,
        height=height,
    )

    valid, reason = (
        validate_annotation(
            item.to_dict()
        )
    )

    if not valid:
        raise ValueError(reason)

    return item


def make_none_annotation(
    image_name: str,
    width: int,
    height: int,
) -> Annotation:
    """
    建立 NONE 標註。
    """

    item = Annotation(
        image=image_name,
        state="NONE",
        x=None,
        y=None,
        x_norm=None,
        y_norm=None,
        width=width,
        height=height,
    )

    valid, reason = (
        validate_annotation(
            item.to_dict()
        )
    )

    if not valid:
        raise ValueError(reason)

    return item


# ============================================================
# 圖片索引
# ============================================================

def first_unfinished_index(
    images: list[Path],
    annotations: dict[
        str,
        dict[str, Any],
    ],
) -> int:
    """
    找到第一張未完成圖片。
    """

    for index, image in enumerate(images):
        if image.name not in annotations:
            return index

    return 0


def next_unfinished_index(
    images: list[Path],
    annotations: dict[
        str,
        dict[str, Any],
    ],
    start: int,
) -> int | None:
    """
    從目前圖片開始搜尋下一張未完成圖片。
    """

    for offset in range(
        1,
        len(images) + 1,
    ):
        index = (
            start + offset
        ) % len(images)

        if (
            images[index].name
            not in annotations
        ):
            return index

    return None


# ============================================================
# 畫面繪製
# ============================================================

def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.58,
    thickness: int = 1,
    color: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
) -> None:
    """
    使用黑色描邊加白色文字。

    不使用大面積黑色矩形，
    避免遮住原始畫面。
    """

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_marker(
    image: np.ndarray,
    x: int,
    y: int,
    selected: bool,
) -> None:
    """
    畫滑鼠十字線或已選取的標註點。
    """

    radius = (
        12
        if selected
        else 7
    )

    color = (
        (0, 255, 255)
        if selected
        else (255, 255, 255)
    )

    cv2.line(
        image,
        (
            x - radius,
            y,
        ),
        (
            x + radius,
            y,
        ),
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        image,
        (
            x,
            y - radius,
        ),
        (
            x,
            y + radius,
        ),
        color,
        1,
        cv2.LINE_AA,
    )

    if selected:
        cv2.circle(
            image,
            (x, y),
            11,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            (x, y),
            3,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )


def make_magnifier(
    rgb: np.ndarray,
    x: int,
    y: int,
    output_size: int = MAGNIFIER_SIZE,
) -> np.ndarray:
    """
    建立 RGB 局部放大鏡。

    使用最近鄰插值，
    讓像素邊界更清楚，
    避免平滑後看不準尖端。
    """

    radius = MAGNIFIER_SOURCE_RADIUS

    x_start = max(
        0,
        x - radius,
    )

    x_end = min(
        rgb.shape[1],
        x + radius + 1,
    )

    y_start = max(
        0,
        y - radius,
    )

    y_end = min(
        rgb.shape[0],
        y + radius + 1,
    )

    crop = rgb[
        y_start:y_end,
        x_start:x_end,
    ]

    if crop.size == 0:
        return np.zeros(
            (
                output_size,
                output_size,
                3,
            ),
            dtype=np.uint8,
        )

    magnifier = cv2.resize(
        crop,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    center = output_size // 2

    cv2.line(
        magnifier,
        (
            center - 18,
            center,
        ),
        (
            center + 18,
            center,
        ),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        magnifier,
        (
            center,
            center - 18,
        ),
        (
            center,
            center + 18,
        ),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        magnifier,
        (0, 0),
        (
            output_size - 1,
            output_size - 1,
        ),
        (255, 255, 255),
        1,
    )

    return magnifier


# ============================================================
# 主程式
# ============================================================

def main() -> int:
    args = parse_args()

    task_dir = resolve_task_directory(
        args.task_dir
    )

    rgb_dir = task_dir / "rgb"

    depth_dir = (
        task_dir
        / args.depth_dir_name
    )

    annotation_dir = (
        task_dir
        / "annotation"
    )

    annotation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        annotation_dir
        / "tip_state_labels.jsonl"
    )

    csv_path = (
        annotation_dir
        / "tip_state_labels.csv"
    )

    images = find_images(
        rgb_dir
    )

    annotations = (
        load_existing_annotations(
            jsonl_path
        )
    )

    current_index = (
        first_unfinished_index(
            images,
            annotations,
        )
    )

    # 下一次點擊時使用的狀態
    pending_state = "UP"

    current_rgb: np.ndarray | None = None
    current_depth: np.ndarray | None = None
    current_depth_view: np.ndarray | None = None

    width = 0
    height = 0

    image_changed = True

    notice_text = ""
    notice_until = 0

    mouse_x = -1
    mouse_y = -1
    mouse_inside = False

    zoom = DEFAULT_ZOOM

    def save_all(
        message: str,
    ) -> None:
        nonlocal notice_text
        nonlocal notice_until

        save_annotations(
            annotations,
            jsonl_path,
            csv_path,
        )

        notice_text = message

        notice_until = (
            cv2.getTickCount()
            + int(
                cv2.getTickFrequency()
                * 1.0
            )
        )

        print(f"✅ {message}")

    def load_current() -> None:
        nonlocal current_rgb
        nonlocal current_depth
        nonlocal current_depth_view
        nonlocal width
        nonlocal height
        nonlocal image_changed
        nonlocal mouse_inside

        rgb_path = images[
            current_index
        ]

        current_rgb = load_rgb(
            rgb_path
        )

        height, width = (
            current_rgb.shape[:2]
        )

        depth_path = find_depth_path(
            depth_dir,
            rgb_path,
        )

        current_depth = load_depth(
            depth_path,
            (
                width,
                height,
            ),
        )

        current_depth_view = (
            depth_to_colormap(
                current_depth,
                (
                    width,
                    height,
                ),
            )
        )

        image_changed = False
        mouse_inside = False

    def canvas_to_image(
        canvas_x: int,
        canvas_y: int,
    ) -> tuple[int, int] | None:
        """
        將畫面座標換算回原始 RGB 座標。

        RGB 區與 Depth 區皆可點擊。
        """

        scaled_width = int(
            round(
                width * zoom
            )
        )

        scaled_height = int(
            round(
                height * zoom
            )
        )

        if not (
            0 <= canvas_y
            < scaled_height
        ):
            return None

        # RGB 區域
        if (
            0 <= canvas_x
            < scaled_width
        ):
            original_x = int(
                canvas_x / zoom
            )

            original_y = int(
                canvas_y / zoom
            )

        # Depth 區域
        elif (
            scaled_width
            <= canvas_x
            < scaled_width * 2
        ):
            original_x = int(
                (
                    canvas_x
                    - scaled_width
                )
                / zoom
            )

            original_y = int(
                canvas_y / zoom
            )

        else:
            return None

        original_x = min(
            width - 1,
            max(
                0,
                original_x,
            ),
        )

        original_y = min(
            height - 1,
            max(
                0,
                original_y,
            ),
        )

        return (
            original_x,
            original_y,
        )

    def on_mouse(
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        nonlocal mouse_x
        nonlocal mouse_y
        nonlocal mouse_inside
        nonlocal zoom

        mapped = canvas_to_image(
            x,
            y,
        )

        if mapped is not None:
            mouse_x, mouse_y = mapped
            mouse_inside = True

        else:
            mouse_inside = False

        # 滑鼠滾輪縮放
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom = min(
                    MAX_ZOOM,
                    zoom + ZOOM_STEP,
                )

            else:
                zoom = max(
                    MIN_ZOOM,
                    zoom - ZOOM_STEP,
                )

            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if mapped is None:
            return

        if current_rgb is None:
            return

        if pending_state == "NONE":
            print(
                "⚠️ 目前 State=NONE；"
                "請按 W 或 S 後再標註尖端。"
            )

            return

        point_x, point_y = mapped

        image_name = images[
            current_index
        ].name

        annotation = (
            make_point_annotation(
                image_name=image_name,
                state=pending_state,
                x=point_x,
                y=point_y,
                width=width,
                height=height,
            )
        )

        annotations[
            image_name
        ] = annotation.to_dict()

        save_all(
            f"{image_name}："
            f"{pending_state} "
            f"尖端標註完成"
        )

        # 不自動換張
        # 使用者確認後按 D 前往下一張

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        on_mouse,
    )

    print("=" * 72)

    print(
        f"Task：{task_dir}"
    )

    print(
        f"RGB：{rgb_dir}"
    )

    print(
        f"Depth：{depth_dir} "
        f"{'(存在)' if depth_dir.is_dir() else '(不存在，仍可標註)'}"
    )

    print(
        f"圖片：{len(images)} 張；"
        f"已完成：{len(annotations)} 張"
    )

    print(
        "A：上一張；D：下一張"
    )

    print(
        "W：焊槍抬起 UP"
    )

    print(
        "S：焊槍落下 DOWN"
    )

    print(
        "N：畫面沒有焊槍 NONE"
    )

    print(
        "左鍵：標註目前 UP/DOWN 尖端"
    )

    print(
        "點擊完成後不會自動換張"
    )

    print(
        "F：下一張未完成"
    )

    print(
        "X/Delete：刪除目前標註"
    )

    print(
        "1/2/3：顯示倍率"
    )

    print(
        "滑鼠滾輪：縮放"
    )

    print(
        "Q/ESC：離開"
    )

    print("=" * 72)

    while True:
        if image_changed:
            load_current()

        assert current_rgb is not None
        assert current_depth_view is not None

        image_path = images[
            current_index
        ]

        item = annotations.get(
            image_path.name
        )

        completed = item is not None

        display_state = (
            str(item["state"])
            if completed
            else pending_state
        )

        rgb_view = current_rgb.copy()

        depth_view = (
            current_depth_view.copy()
        )

        # 滑鼠十字線
        if mouse_inside:
            draw_marker(
                rgb_view,
                mouse_x,
                mouse_y,
                selected=False,
            )

            draw_marker(
                depth_view,
                mouse_x,
                mouse_y,
                selected=False,
            )

        selected_x: int | None = None
        selected_y: int | None = None

        # 已完成的標註點
        if (
            item
            and item["state"]
            in {"UP", "DOWN"}
        ):
            selected_x = int(
                item["x"]
            )

            selected_y = int(
                item["y"]
            )

            draw_marker(
                rgb_view,
                selected_x,
                selected_y,
                selected=True,
            )

            draw_marker(
                depth_view,
                selected_x,
                selected_y,
                selected=True,
            )

        scaled_width = int(
            round(
                width * zoom
            )
        )

        scaled_height = int(
            round(
                height * zoom
            )
        )

        rgb_scaled = cv2.resize(
            rgb_view,
            (
                scaled_width,
                scaled_height,
            ),
            interpolation=(
                cv2.INTER_NEAREST
                if zoom > 1.0
                else cv2.INTER_LINEAR
            ),
        )

        depth_scaled = cv2.resize(
            depth_view,
            (
                scaled_width,
                scaled_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        canvas_height = max(
            scaled_height,
            650,
        )

        canvas_width = (
            scaled_width * 2
            + INFO_PANEL_WIDTH
        )

        canvas = np.zeros(
            (
                canvas_height,
                canvas_width,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            :scaled_height,
            :scaled_width,
        ] = rgb_scaled

        canvas[
            :scaled_height,
            scaled_width:scaled_width * 2,
        ] = depth_scaled

        # 小型標題，不使用大型黑色遮罩
        draw_text(
            canvas,
            "RGB",
            (10, 24),
            scale=0.65,
            thickness=1,
        )

        draw_text(
            canvas,
            "DEPTH",
            (
                scaled_width + 10,
                24,
            ),
            scale=0.65,
            thickness=1,
        )

        panel_x = (
            scaled_width * 2
            + 14
        )

        start_y = 28
        line_height = 26

        unfinished_count = sum(
            image.name
            not in annotations
            for image in images
        )

        if mouse_inside:
            mouse_depth = (
                get_depth_value(
                    current_depth,
                    mouse_x,
                    mouse_y,
                )
            )

            mouse_median_depth = (
                get_median_depth(
                    current_depth,
                    mouse_x,
                    mouse_y,
                )
            )

        else:
            mouse_depth = None
            mouse_median_depth = None

        if (
            selected_x is not None
            and selected_y is not None
        ):
            selected_depth = (
                get_depth_value(
                    current_depth,
                    selected_x,
                    selected_y,
                )
            )

            selected_median_depth = (
                get_median_depth(
                    current_depth,
                    selected_x,
                    selected_y,
                )
            )

        else:
            selected_depth = None
            selected_median_depth = None

        information = [
            f"Image {current_index + 1}/{len(images)}",
            image_path.name[:42],
            "",
            f"State: {display_state}",
            (
                "Completed: YES"
                if completed
                else "Completed: NO"
            ),
            f"Unfinished: {unfinished_count}",
            f"Zoom: {zoom:.2f}x",
            "",
            "Mouse",
            (
                f"X: {mouse_x}"
                if mouse_inside
                else "X: --"
            ),
            (
                f"Y: {mouse_y}"
                if mouse_inside
                else "Y: --"
            ),
            (
                f"Depth: "
                f"{format_depth(mouse_depth)}"
            ),
            (
                f"Median: "
                f"{format_depth(mouse_median_depth)}"
            ),
            "",
            "Selected",
            (
                f"X: {selected_x}"
                if selected_x is not None
                else "X: --"
            ),
            (
                f"Y: {selected_y}"
                if selected_y is not None
                else "Y: --"
            ),
            (
                f"Depth: "
                f"{format_depth(selected_depth)}"
            ),
            (
                f"Median: "
                f"{format_depth(selected_median_depth)}"
            ),
            "",
            "A prev / D next",
            "W UP / S DOWN",
            "N NONE",
            "F unfinished",
            "X delete / Q quit",
            "1 2 3 zoom",
        ]

        for line_index, text in enumerate(
            information
        ):
            if not text:
                continue

            draw_text(
                canvas,
                text,
                (
                    panel_x,
                    start_y
                    + line_index
                    * line_height,
                ),
                scale=0.47,
                thickness=1,
            )

        # 放大鏡位置優先使用滑鼠
        if mouse_inside:
            focus_x = mouse_x
            focus_y = mouse_y

        elif (
            selected_x is not None
            and selected_y is not None
        ):
            focus_x = selected_x
            focus_y = selected_y

        else:
            focus_x = width // 2
            focus_y = height // 2

        magnifier = make_magnifier(
            current_rgb,
            int(focus_x),
            int(focus_y),
        )

        magnifier_y = (
            canvas_height
            - MAGNIFIER_SIZE
            - 10
        )

        magnifier_x = panel_x

        if (
            magnifier_y >= 0
            and magnifier_x
            + MAGNIFIER_SIZE
            <= canvas_width
        ):
            canvas[
                magnifier_y:
                magnifier_y
                + MAGNIFIER_SIZE,
                magnifier_x:
                magnifier_x
                + MAGNIFIER_SIZE,
            ] = magnifier

        # 顯示短暫通知
        if (
            cv2.getTickCount()
            < notice_until
        ):
            draw_text(
                canvas,
                notice_text,
                (
                    10,
                    canvas_height - 16,
                ),
                scale=0.52,
                thickness=1,
            )

        cv2.imshow(
            WINDOW_NAME,
            canvas,
        )

        key = cv2.waitKeyEx(20)

        # ----------------------------------------------------
        # 離開
        # ----------------------------------------------------

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            break

        # ----------------------------------------------------
        # 狀態切換
        # ----------------------------------------------------

        if key in (
            ord("w"),
            ord("W"),
        ):
            pending_state = "UP"

            notice_text = (
                "Selected State: UP"
            )

            notice_until = (
                cv2.getTickCount()
                + int(
                    cv2.getTickFrequency()
                    * 0.8
                )
            )

            continue

        if key in (
            ord("s"),
            ord("S"),
        ):
            pending_state = "DOWN"

            notice_text = (
                "Selected State: DOWN"
            )

            notice_until = (
                cv2.getTickCount()
                + int(
                    cv2.getTickFrequency()
                    * 0.8
                )
            )

            continue

        if key in (
            ord("n"),
            ord("N"),
        ):
            image_name = image_path.name

            annotation = (
                make_none_annotation(
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            )

            annotations[
                image_name
            ] = annotation.to_dict()

            save_all(
                f"{image_name}："
                f"NONE 標註完成"
            )

            # 不自動換張
            continue

        # ----------------------------------------------------
        # 圖片切換
        # ----------------------------------------------------

        if key in (
            ord("a"),
            ord("A"),
            2424832,
        ):
            if current_index > 0:
                current_index -= 1
                image_changed = True

            continue

        if key in (
            ord("d"),
            ord("D"),
            2555904,
        ):
            if (
                current_index
                < len(images) - 1
            ):
                current_index += 1
                image_changed = True

            continue

        # ----------------------------------------------------
        # 下一張未完成
        # ----------------------------------------------------

        if key in (
            ord("f"),
            ord("F"),
        ):
            target_index = (
                next_unfinished_index(
                    images,
                    annotations,
                    current_index,
                )
            )

            if target_index is None:
                notice_text = (
                    "All images completed"
                )

                notice_until = (
                    cv2.getTickCount()
                    + int(
                        cv2.getTickFrequency()
                        * 1.2
                    )
                )

                print(
                    "✔ 所有圖片皆已完成標註。"
                )

            else:
                current_index = (
                    target_index
                )

                image_changed = True

            continue

        # ----------------------------------------------------
        # 刪除目前標註
        # ----------------------------------------------------

        if key in (
            ord("x"),
            ord("X"),
            3014656,
            127,
        ):
            if (
                image_path.name
                in annotations
            ):
                del annotations[
                    image_path.name
                ]

                save_all(
                    f"{image_path.name}："
                    f"標註已刪除"
                )
            continue

        # ----------------------------------------------------
        # 固定顯示倍率
        # ----------------------------------------------------

        if key == ord("1"):
            zoom = 1.0
            continue

        if key == ord("2"):
            zoom = 2.0
            continue

        if key == ord("3"):
            zoom = 3.0
            continue

    cv2.destroyAllWindows()

    # 離開前再次同步
    save_annotations(
        annotations,
        jsonl_path,
        csv_path,
    )

    print(
        f"JSONL：{jsonl_path}"
    )

    print(
        f"CSV：{csv_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())