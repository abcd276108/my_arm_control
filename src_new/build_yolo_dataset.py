from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# 固定設定
# ============================================================

CLASS_TO_ID = {
    "UP": 0,
    "DOWN": 1,
}

CLASS_NAMES = [
    "tip_up",
    "tip_down",
]

STATES = (
    "UP",
    "DOWN",
    "NONE",
)

SPLITS = (
    "train",
    "val",
    "test",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


@dataclass(frozen=True)
class Sample:
    category: str
    task_name: str
    image_path: Path
    annotation: dict[str, Any]

    @property
    def task_key(self) -> tuple[str, str]:
        return self.category, self.task_name


@dataclass
class SourceScanStats:
    task_count: int = 0
    rgb_image_count: int = 0
    annotation_count: int = 0
    unannotated_image_count: int = 0
    missing_image_count: int = 0


# ============================================================
# 參數
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "將 Smart Teaching 標註轉換成 "
            "tip_up / tip_down 兩類 YOLO Dataset"
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "data"
            / "dataset"
        ),
        help="原始資料集根目錄，結構為 類別/task_XXXX/rgb。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "data"
            / "yolo_tip_state_dataset"
        ),
        help="YOLO Dataset 輸出資料夾。",
    )

    parser.add_argument(
        "--split-file",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "data"
            / "dataset_split.json"
        ),
        help=(
            "固定 Train/Val/Test task 的 JSON 設定檔。"
            "預設為 data/dataset_split.json。"
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--box-size",
        type=int,
        default=20,
        help="以尖端為中心產生的正方形框邊長，預設 20 px。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--copy-mode",
        choices=(
            "copy",
            "hardlink",
        ),
        default="copy",
        help="copy 最穩定；hardlink 可節省磁碟空間。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="輸出資料夾已存在時，刪除後重新建立。",
    )

    args = parser.parse_args()

    ratios = (
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    if any(value < 0 for value in ratios):
        parser.error(
            "Train、Val、Test 比例不可小於 0。"
        )

    if abs(sum(ratios) - 1.0) > 1e-6:
        parser.error(
            "Train、Val、Test 比例總和必須等於 1。"
        )

    if args.train_ratio <= 0:
        parser.error(
            "--train-ratio 必須大於 0。"
        )

    if args.box_size <= 0:
        parser.error(
            "--box-size 必須大於 0。"
        )

    return args


# ============================================================
# 通用工具
# ============================================================

def natural_key(
    value: str | Path,
) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part)
        if part.isdigit()
        else part.casefold()
        for part in re.split(
            r"(\d+)",
            str(value),
        )
    )


def image_files(
    directory: Path,
) -> list[Path]:
    if not directory.is_dir():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=natural_key,
    )


def validate_annotation(
    item: dict[str, Any],
) -> tuple[bool, str]:
    state = item.get("state")

    if state not in STATES:
        return (
            False,
            "state 必須是 UP、DOWN 或 NONE",
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
            f"{state} 缺少尖端座標",
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
    ):
        return (
            False,
            "width/height 必須是整數",
        )

    if width <= 0 or height <= 0:
        return (
            False,
            "width/height 必須大於 0",
        )

    if has_point:
        point_x = float(x)
        point_y = float(y)

        if not (
            0 <= point_x < width
            and 0 <= point_y < height
        ):
            return (
                False,
                "尖端座標超出影像範圍",
            )

    return True, "OK"


def load_jsonl(
    path: Path,
) -> dict[str, dict[str, Any]]:
    annotations: dict[
        str,
        dict[str, Any],
    ] = {}

    if not path.exists():
        return annotations

    lines = path.read_text(
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
                f"⚠️ {path} 第 {line_number} 行 "
                f"JSON 錯誤：{exc}"
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
                f"⚠️ {path} 第 {line_number} 行 "
                "缺少 image，已略過。"
            )
            continue

        valid, reason = validate_annotation(
            item
        )

        if not valid:
            print(
                f"⚠️ {path} 第 {line_number} 行 "
                f"無效：{reason}"
            )
            continue

        annotations[
            item["image"]
        ] = item

    return annotations


def find_image(
    rgb_dir: Path,
    image_name: str,
) -> Path | None:
    direct_path = (
        rgb_dir
        / image_name
    )

    if direct_path.is_file():
        return direct_path

    stem = Path(image_name).stem

    for suffix in sorted(
        IMAGE_EXTENSIONS
    ):
        candidate = (
            rgb_dir
            / f"{stem}{suffix}"
        )

        if candidate.is_file():
            return candidate

    return None


# ============================================================
# 掃描資料
# ============================================================

def scan_samples(
    dataset_root: Path,
) -> tuple[
    list[Sample],
    list[str],
    SourceScanStats,
    dict[str, Any],
]:
    samples: list[Sample] = []
    warnings: list[str] = []

    source_stats = SourceScanStats()

    per_task_source: dict[
        str,
        Any,
    ] = {}

    task_dirs = sorted(
        dataset_root.glob(
            "*/task_*"
        ),
        key=natural_key,
    )

    source_stats.task_count = len(
        task_dirs
    )

    for task_dir in task_dirs:
        category = (
            task_dir.parent.name
        )

        task_key_text = (
            f"{category}/{task_dir.name}"
        )

        rgb_dir = (
            task_dir
            / "rgb"
        )

        jsonl_path = (
            task_dir
            / "annotation"
            / "tip_state_labels.jsonl"
        )

        if not rgb_dir.is_dir():
            warnings.append(
                f"找不到 rgb 資料夾：{task_dir}"
            )
            continue

        rgb_paths = image_files(
            rgb_dir
        )

        source_stats.rgb_image_count += (
            len(rgb_paths)
        )

        annotations = load_jsonl(
            jsonl_path
        )

        source_stats.annotation_count += (
            len(annotations)
        )

        if not annotations:
            warnings.append(
                f"沒有有效標註：{task_dir}"
            )

        task_state_counts = Counter()

        matched_rgb_names: set[str] = set()
        missing_images = 0

        for image_name in sorted(
            annotations,
            key=natural_key,
        ):
            image_path = find_image(
                rgb_dir,
                image_name,
            )

            if image_path is None:
                warnings.append(
                    "標註找不到對應圖片："
                    f"{rgb_dir / image_name}"
                )

                source_stats.missing_image_count += 1
                missing_images += 1

                continue

            matched_rgb_names.add(
                image_path.name
            )

            annotation = annotations[
                image_name
            ]

            state = str(
                annotation["state"]
            )

            task_state_counts[
                state
            ] += 1

            samples.append(
                Sample(
                    category=category,
                    task_name=task_dir.name,
                    image_path=image_path,
                    annotation=annotation,
                )
            )

        rgb_names = {
            path.name
            for path in rgb_paths
        }

        unannotated = sorted(
            rgb_names
            - matched_rgb_names,
            key=natural_key,
        )

        source_stats.unannotated_image_count += (
            len(unannotated)
        )

        if unannotated:
            warnings.append(
                f"{task_key_text} 尚有 "
                f"{len(unannotated)} 張 RGB 未標註"
            )

        per_task_source[
            task_key_text
        ] = {
            "category": category,
            "task": task_dir.name,
            "rgb_images": len(
                rgb_paths
            ),
            "valid_annotations": sum(
                task_state_counts.values()
            ),
            "UP": task_state_counts[
                "UP"
            ],
            "DOWN": task_state_counts[
                "DOWN"
            ],
            "NONE": task_state_counts[
                "NONE"
            ],
            "unannotated_images": len(
                unannotated
            ),
            "annotations_missing_image": (
                missing_images
            ),
        }

    return (
        samples,
        warnings,
        source_stats,
        per_task_source,
    )


# ============================================================
# Task-based split
# ============================================================

def normalize_task_key(
    value: str,
) -> str:
    """
    將 JSON 中的 task 路徑統一成：

        category/task_XXXX

    同時支援：
        category/task_XXXX
        category\\task_XXXX
    """

    normalized = (
        value.strip()
        .replace("\\", "/")
    )

    parts = [
        part
        for part in normalized.split("/")
        if part
    ]

    if len(parts) != 2:
        raise ValueError(
            "Task 路徑必須是 "
            "category/task_XXXX 格式："
            f"{value}"
        )

    category = parts[0]
    task_name = parts[1]

    if not task_name.startswith("task_"):
        raise ValueError(
            "Task 資料夾名稱必須以 task_ 開頭："
            f"{value}"
        )

    return (
        f"{category}/{task_name}"
    )


def discover_task_keys(
    dataset_root: Path,
) -> set[str]:
    """
    讀取 data/dataset 下實際存在的所有 task。
    """

    return {
        (
            f"{task_dir.parent.name}/"
            f"{task_dir.name}"
        )
        for task_dir
        in dataset_root.glob("*/task_*")
        if task_dir.is_dir()
    }


def load_fixed_task_split(
    split_file: Path,
    dataset_root: Path,
) -> dict[tuple[str, str], str]:
    """
    讀取 dataset_split.json。

    執行以下檢查：
    1. JSON 格式是否正確
    2. Task 是否重複出現在不同集合
    3. JSON 中的 task 資料夾是否存在
    4. 未列入 JSON 的 task 保留在原始資料集，但不參與本次建立
    """

    if not split_file.is_file():
        raise SystemExit(
            "找不到固定切分檔："
            f"{split_file}\n"
            "請確認 data/dataset_split.json 是否存在。"
        )

    try:
        split_data = json.loads(
            split_file.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise SystemExit(
            "dataset_split.json 格式錯誤：\n"
            f"{exc}"
        ) from exc

    if not isinstance(split_data, dict):
        raise SystemExit(
            "dataset_split.json 最外層必須是物件，"
            "並包含 train、val、test。"
        )

    allowed_keys = set(
        SPLITS
    )

    unknown_keys = (
        set(split_data)
        - allowed_keys
    )

    if unknown_keys:
        raise SystemExit(
            "dataset_split.json 出現未知欄位："
            + ", ".join(
                sorted(unknown_keys)
            )
        )

    task_to_split: dict[
        str,
        str,
    ] = {}

    duplicate_messages: list[str] = []

    for split in SPLITS:
        task_list = split_data.get(
            split,
            [],
        )

        if not isinstance(
            task_list,
            list,
        ):
            raise SystemExit(
                f"dataset_split.json 的 {split} "
                "必須是陣列。"
            )

        for raw_task in task_list:
            if not isinstance(
                raw_task,
                str,
            ):
                raise SystemExit(
                    f"{split} 中包含非文字 task："
                    f"{raw_task}"
                )

            try:
                task_key = normalize_task_key(
                    raw_task
                )

            except ValueError as exc:
                raise SystemExit(
                    str(exc)
                ) from exc

            previous_split = (
                task_to_split.get(
                    task_key
                )
            )

            if previous_split is not None:
                duplicate_messages.append(
                    f"{task_key} 同時出現在 "
                    f"{previous_split} 與 {split}"
                )

                continue

            task_to_split[
                task_key
            ] = split

    if duplicate_messages:
        raise SystemExit(
            "dataset_split.json 有重複 task：\n"
            + "\n".join(
                duplicate_messages
            )
        )

    if not task_to_split:
        raise SystemExit(
            "dataset_split.json 沒有列出任何 task。"
        )

    actual_tasks = discover_task_keys(
        dataset_root
    )

    listed_tasks = set(
        task_to_split
    )

    # JSON 有寫，但資料夾不存在。
    missing_on_disk = sorted(
        listed_tasks
        - actual_tasks,
        key=natural_key,
    )

    # 資料夾存在，但 JSON 沒寫。
    unlisted_tasks = sorted(
        actual_tasks
        - listed_tasks,
        key=natural_key,
    )

    error_messages: list[str] = []

    if missing_on_disk:
        error_messages.append(
            "以下 task 已列在 JSON，"
            "但資料夾不存在：\n  "
            + "\n  ".join(
                missing_on_disk
            )
        )

    if error_messages:
        raise SystemExit(
            "固定切分檢查失敗：\n\n"
            + "\n\n".join(
                error_messages
            )
        )

    if unlisted_tasks:
        print(
            "\n[注意] 以下 task 未列入 dataset_split.json，"
            "本次將保留但不加入 YOLO Dataset：\n  "
            + "\n  ".join(unlisted_tasks)
        )

    mapping: dict[
        tuple[str, str],
        str,
    ] = {}

    for (
        task_key,
        split,
    ) in task_to_split.items():
        category, task_name = (
            task_key.split(
                "/",
                1,
            )
        )

        mapping[
            (
                category,
                task_name,
            )
        ] = split

    print(
        "\n✅ 已讀取固定 Dataset 切分"
    )

    print(
        f"設定檔：{split_file}"
    )

    for split in SPLITS:
        count = sum(
            1
            for assigned_split
            in mapping.values()
            if assigned_split == split
        )

        print(
            f"{split:>5}：{count} 個 task"
        )

    return mapping

def split_tasks(
    samples: Iterable[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[
    tuple[str, str],
    str,
]:
    """
    以 task 為單位切分。

    同一段連續影格不會同時出現在
    Train、Val 或 Test。
    """

    tasks = sorted(
        {
            sample.task_key
            for sample in samples
        },
        key=lambda item: natural_key(
            str(item)
        ),
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        tasks
    )

    task_count = len(
        tasks
    )

    if task_count == 0:
        return {}

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    counts = {
        split: int(
            task_count * ratio
        )
        for split, ratio
        in ratios.items()
    }

    remaining = (
        task_count
        - sum(
            counts.values()
        )
    )

    remainders = sorted(
        ratios,
        key=lambda split: (
            task_count
            * ratios[split]
            - counts[split]
        ),
        reverse=True,
    )

    for split in remainders[
        :remaining
    ]:
        counts[split] += 1

    if counts["train"] == 0:
        donor = max(
            counts,
            key=counts.get,
        )

        counts[donor] -= 1
        counts["train"] = 1

    # Task 數量足夠時，Val/Test 各至少一個 task。
    if task_count >= 3:
        for split in (
            "val",
            "test",
        ):
            if (
                ratios[split] > 0
                and counts[split] == 0
                and counts["train"] > 1
            ):
                counts["train"] -= 1
                counts[split] += 1

    mapping: dict[
        tuple[str, str],
        str,
    ] = {}

    start = 0

    for split in SPLITS:
        end = (
            start
            + counts[split]
        )

        for task in tasks[
            start:end
        ]:
            mapping[
                task
            ] = split

        start = end

    return mapping


# ============================================================
# YOLO 轉換
# ============================================================

def safe_output_name(
    sample: Sample,
) -> str:
    raw_name = (
        f"{sample.category}__"
        f"{sample.task_name}__"
        f"{sample.image_path.stem}"
    )

    cleaned_name = "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "_"
        for character in raw_name
    )

    digest = hashlib.sha1(
        str(
            sample.image_path
        ).encode("utf-8")
    ).hexdigest()[:8]

    return (
        f"{cleaned_name}__"
        f"{digest}"
        f"{sample.image_path.suffix.lower()}"
    )


def annotation_to_yolo(
    item: dict[str, Any],
    box_size: int,
) -> str:
    """
    NONE：
        空 txt

    UP：
        class 0

    DOWN：
        class 1
    """

    state = str(
        item["state"]
    )

    if state == "NONE":
        return ""

    width = int(
        item["width"]
    )

    height = int(
        item["height"]
    )

    x = float(
        item["x"]
    )

    y = float(
        item["y"]
    )

    half_size = (
        box_size
        / 2.0
    )

    x1 = max(
        0.0,
        x - half_size,
    )

    y1 = max(
        0.0,
        y - half_size,
    )

    x2 = min(
        float(width),
        x + half_size,
    )

    y2 = min(
        float(height),
        y + half_size,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            "尖端方框無效"
        )

    x_center = (
        (
            x1 + x2
        )
        / 2.0
        / width
    )

    y_center = (
        (
            y1 + y2
        )
        / 2.0
        / height
    )

    normalized_width = (
        (
            x2 - x1
        )
        / width
    )

    normalized_height = (
        (
            y2 - y1
        )
        / height
    )

    class_id = CLASS_TO_ID[
        state
    ]

    return (
        f"{class_id} "
        f"{x_center:.8f} "
        f"{y_center:.8f} "
        f"{normalized_width:.8f} "
        f"{normalized_height:.8f}\n"
    )


def transfer_file(
    source: Path,
    destination: Path,
    mode: str,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if mode == "hardlink":
        try:
            destination.hardlink_to(
                source
            )
            return

        except OSError:
            print(
                "⚠️ hardlink 失敗，"
                f"改用 copy：{source}"
            )

    shutil.copy2(
        source,
        destination,
    )


def write_dataset_yaml(
    output_dir: Path,
) -> Path:
    yaml_path = (
        output_dir
        / "dataset.yaml"
    )

    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tip_up\n"
        "  1: tip_down\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    return yaml_path


# ============================================================
# Dataset 完整性檢查
# ============================================================

def parse_yolo_label(
    label_path: Path,
) -> tuple[
    bool,
    str,
    int,
]:
    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return (
            True,
            "empty",
            0,
        )

    object_count = 0

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            return (
                False,
                (
                    f"第 {line_number} 行"
                    "欄位數不是 5"
                ),
                object_count,
            )

        try:
            class_id = int(
                parts[0]
            )

            values = [
                float(value)
                for value
                in parts[1:]
            ]

        except ValueError:
            return (
                False,
                (
                    f"第 {line_number} 行"
                    "包含非數字內容"
                ),
                object_count,
            )

        if (
            class_id
            not in CLASS_TO_ID.values()
        ):
            return (
                False,
                (
                    f"第 {line_number} 行"
                    f"類別 ID 無效：{class_id}"
                ),
                object_count,
            )

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            return (
                False,
                (
                    f"第 {line_number} 行"
                    "座標超出 0～1"
                ),
                object_count,
            )

        box_width = values[2]
        box_height = values[3]

        if (
            box_width <= 0
            or box_height <= 0
        ):
            return (
                False,
                (
                    f"第 {line_number} 行"
                    "框寬高必須大於 0"
                ),
                object_count,
            )

        object_count += 1

    return (
        True,
        "ok",
        object_count,
    )


def check_dataset_integrity(
    output_dir: Path,
    manifest_rows: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    issues: list[str] = []

    split_results: dict[
        str,
        Any,
    ] = {}

    manifest_by_output = {
        str(
            row["output_image"]
        ): row
        for row in manifest_rows
    }

    for split in SPLITS:
        image_dir = (
            output_dir
            / "images"
            / split
        )

        label_dir = (
            output_dir
            / "labels"
            / split
        )

        images = image_files(
            image_dir
        )

        labels = sorted(
            label_dir.glob(
                "*.txt"
            ),
            key=natural_key,
        )

        image_stems = {
            path.stem
            for path in images
        }

        label_stems = {
            path.stem
            for path in labels
        }

        missing_labels = sorted(
            image_stems
            - label_stems,
            key=natural_key,
        )

        orphan_labels = sorted(
            label_stems
            - image_stems,
            key=natural_key,
        )

        for stem in missing_labels:
            issues.append(
                f"{split} 圖片缺少 label：{stem}"
            )

        for stem in orphan_labels:
            issues.append(
                f"{split} label 缺少圖片：{stem}"
            )

        empty_labels = 0
        nonempty_labels = 0
        invalid_labels = 0
        object_count = 0

        for label_path in labels:
            (
                valid,
                reason,
                objects,
            ) = parse_yolo_label(
                label_path
            )

            if not valid:
                invalid_labels += 1

                issues.append(
                    f"{split} 無效 label "
                    f"{label_path.name}：{reason}"
                )

                continue

            object_count += objects

            if reason == "empty":
                empty_labels += 1
            else:
                nonempty_labels += 1

        state_mismatch = 0

        for image_path in images:
            relative_image = (
                image_path
                .relative_to(
                    output_dir
                )
                .as_posix()
            )

            row = manifest_by_output.get(
                relative_image
            )

            if row is None:
                state_mismatch += 1

                issues.append(
                    "manifest 缺少圖片："
                    f"{relative_image}"
                )

                continue

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            is_empty = not (
                label_path.read_text(
                    encoding="utf-8"
                ).strip()
            )

            expected_empty = (
                row["state"]
                == "NONE"
            )

            if is_empty != expected_empty:
                state_mismatch += 1

                issues.append(
                    f"{split} {image_path.name}："
                    f"state={row['state']} "
                    "與 label 空白狀態不一致"
                )

        split_ok = not (
            missing_labels
            or orphan_labels
            or invalid_labels
            or state_mismatch
        )

        split_results[
            split
        ] = {
            "images": len(
                images
            ),
            "labels": len(
                labels
            ),
            "missing_labels": len(
                missing_labels
            ),
            "orphan_labels": len(
                orphan_labels
            ),
            "empty_labels": (
                empty_labels
            ),
            "nonempty_labels": (
                nonempty_labels
            ),
            "invalid_labels": (
                invalid_labels
            ),
            "objects": object_count,
            "manifest_state_mismatch": (
                state_mismatch
            ),
            "ok": split_ok,
        }

    return {
        "ok": len(
            issues
        ) == 0,
        "splits": split_results,
        "issues": issues,
    }


# ============================================================
# 統計顯示
# ============================================================

def safe_ratio(
    value: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return (
        value
        / total
    )


def print_statistics(
    split_counts: dict[
        str,
        dict[str, int],
    ],
    task_stats: dict[
        str,
        dict[str, Any],
    ],
    integrity: dict[str, Any],
) -> None:
    print(
        "\n"
        + "=" * 72
    )

    print(
        "Dataset 統計"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Split':<8}"
        f"{'UP':>9}"
        f"{'DOWN':>9}"
        f"{'NONE':>9}"
        f"{'Total':>10}"
    )

    for split in SPLITS:
        counts = split_counts[
            split
        ]

        total = sum(
            counts.values()
        )

        print(
            f"{split:<8}"
            f"{counts['UP']:>9}"
            f"{counts['DOWN']:>9}"
            f"{counts['NONE']:>9}"
            f"{total:>10}"
        )

    total_counts = Counter()

    for counts in (
        split_counts.values()
    ):
        total_counts.update(
            counts
        )

    total = sum(
        total_counts.values()
    )

    print(
        "-" * 72
    )

    print(
        f"{'TOTAL':<8}"
        f"{total_counts['UP']:>9}"
        f"{total_counts['DOWN']:>9}"
        f"{total_counts['NONE']:>9}"
        f"{total:>10}"
    )

    print(
        "比例："
        f"UP {safe_ratio(total_counts['UP'], total):.2%} / "
        f"DOWN {safe_ratio(total_counts['DOWN'], total):.2%} / "
        f"NONE {safe_ratio(total_counts['NONE'], total):.2%}"
    )

    print(
        "\n每個 task："
    )

    for task_key in sorted(
        task_stats,
        key=natural_key,
    ):
        item = task_stats[
            task_key
        ]

        print(
            f"  {task_key:<34} "
            f"[{item['split']:<5}] "
            f"UP {item['UP']:>5} / "
            f"DOWN {item['DOWN']:>5} / "
            f"NONE {item['NONE']:>5} / "
            f"Total {item['total']:>5}"
        )

    print(
        "\nIntegrity Check"
    )

    for split in SPLITS:
        item = integrity[
            "splits"
        ][split]

        mark = (
            "✔"
            if item["ok"]
            else "✘"
        )

        print(
            f"{mark} {split:<5} "
            f"images={item['images']} "
            f"labels={item['labels']} "
            f"empty={item['empty_labels']} "
            f"invalid={item['invalid_labels']}"
        )

    if integrity["ok"]:
        print(
            "✔ Dataset 完整性檢查通過"
        )
    else:
        print(
            "✘ Dataset 發現完整性問題"
        )


# ============================================================
# 主程式
# ============================================================

def main() -> int:
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    split_file = (
        args.split_file
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise SystemExit(
            f"找不到原始資料集：{dataset_root}"
        )

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"輸出資料夾已存在：{output_dir}\n"
                "請加 --overwrite，"
                "或指定其他 --output-dir。"
            )

        shutil.rmtree(
            output_dir
        )

    (
        samples,
        warnings,
        source_stats,
        per_task_source,
    ) = scan_samples(
        dataset_root
    )

    if not samples:
        raise SystemExit(
            "沒有找到可轉換的新格式標註。"
        )

    task_split = load_fixed_task_split(
        split_file=split_file,
        dataset_root=dataset_root,
    )

    # dataset_split.json is the explicit allowlist for this build. Tasks that
    # remain on disk but are intentionally not listed must not reach the
    # conversion loop below.
    samples = [
        sample
        for sample in samples
        if sample.task_key in task_split
    ]

    sample_task_keys = {
        sample.task_key
        for sample in samples
    }

    tasks_without_valid_samples = sorted(
        (
            set(task_split)
            - sample_task_keys
        ),
        key=lambda item: natural_key(
            f"{item[0]}/{item[1]}"
        ),
    )

    if tasks_without_valid_samples:
        formatted_tasks = "\n  ".join(
            (
                f"{category}/{task_name}"
            )
            for (
                category,
                task_name,
            ) in tasks_without_valid_samples
        )

        raise SystemExit(
            "以下 task 雖然存在並已列入切分，"
            "但沒有可用的有效標註：\n  "
            + formatted_tasks
        )

    
    for split in SPLITS:
        (
            output_dir
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_dir
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    split_state_counts = {
        split: {
            state: 0
            for state in STATES
        }
        for split in SPLITS
    }

    task_stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(
        lambda: {
            "split": "",
            "UP": 0,
            "DOWN": 0,
            "NONE": 0,
            "total": 0,
        }
    )

    manifest_rows: list[
        dict[str, Any]
    ] = []

    skipped = 0

    for sample in samples:
        split = task_split[
            sample.task_key
        ]

        output_name = safe_output_name(
            sample
        )

        image_destination = (
            output_dir
            / "images"
            / split
            / output_name
        )

        label_destination = (
            output_dir
            / "labels"
            / split
            / f"{Path(output_name).stem}.txt"
        )

        try:
            label_text = annotation_to_yolo(
                sample.annotation,
                args.box_size,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            warnings.append(
                "略過無效標註 "
                f"{sample.image_path}：{exc}"
            )

            skipped += 1
            continue

        transfer_file(
            sample.image_path,
            image_destination,
            args.copy_mode,
        )

        label_destination.write_text(
            label_text,
            encoding="utf-8",
        )

        state = str(
            sample.annotation[
                "state"
            ]
        )

        split_state_counts[
            split
        ][state] += 1

        task_key_text = (
            f"{sample.category}/"
            f"{sample.task_name}"
        )

        task_stats[
            task_key_text
        ]["split"] = split

        task_stats[
            task_key_text
        ][state] += 1

        task_stats[
            task_key_text
        ]["total"] += 1

        manifest_rows.append(
            {
                "image": output_name,
                "state": state,
                "class_id": (
                    CLASS_TO_ID.get(
                        state,
                        "",
                    )
                ),
                "class_name": (
                    CLASS_NAMES[
                        CLASS_TO_ID[state]
                    ]
                    if state
                    in CLASS_TO_ID
                    else ""
                ),
                "x": sample.annotation.get(
                    "x"
                ),
                "y": sample.annotation.get(
                    "y"
                ),
                "x_norm": sample.annotation.get(
                    "x_norm"
                ),
                "y_norm": sample.annotation.get(
                    "y_norm"
                ),
                "width": sample.annotation.get(
                    "width"
                ),
                "height": sample.annotation.get(
                    "height"
                ),
                "box_size_px": args.box_size,
                "category": sample.category,
                "task": sample.task_name,
                "split": split,
                "source_image": str(
                    sample.image_path
                ),
                "output_image": (
                    image_destination
                    .relative_to(
                        output_dir
                    )
                    .as_posix()
                ),
                "output_label": (
                    label_destination
                    .relative_to(
                        output_dir
                    )
                    .as_posix()
                ),
            }
        )

    yaml_path = write_dataset_yaml(
        output_dir
    )

    manifest_path = (
        output_dir
        / "manifest.csv"
    )

    manifest_fields = [
        "image",
        "state",
        "class_id",
        "class_name",
        "x",
        "y",
        "x_norm",
        "y_norm",
        "width",
        "height",
        "box_size_px",
        "category",
        "task",
        "split",
        "source_image",
        "output_image",
        "output_label",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fields,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    integrity = check_dataset_integrity(
        output_dir,
        manifest_rows,
    )

    total_state_counts = Counter()

    for counts in (
        split_state_counts.values()
    ):
        total_state_counts.update(
            counts
        )

    total_samples = sum(
        total_state_counts.values()
    )

    task_split_summary = {
        f"{category}/{task}": split
        for (
            category,
            task,
        ), split in sorted(
            task_split.items(),
            key=lambda item: natural_key(
                str(item[0])
            ),
        )
    }

    summary = {
        "dataset_root": str(
            dataset_root
        ),
        "output_dir": str(
            output_dir
        ),
        "class_names": CLASS_NAMES,
        "class_mapping": {
            "tip_up": 0,
            "tip_down": 1,
        },
        "none_policy": (
            "empty_label_file"
        ),
        "box_size_px": (
            args.box_size
        ),
        "split_by": "fixed_task_file",

        "split_file": str(
            split_file
        ),
        "task_counts": {
            split: sum(
                1
                for assigned_split
                in task_split.values()
                if assigned_split == split
            )
            for split in SPLITS
        },
        "task_split": (
            task_split_summary
        ),
        "split_state_counts": (
            split_state_counts
        ),
        "total_state_counts": dict(
            total_state_counts
        ),
        "state_proportions": {
            state: safe_ratio(
                total_state_counts[state],
                total_samples,
            )
            for state in STATES
        },
        "total_samples": (
            total_samples
        ),
        "task_statistics": dict(
            task_stats
        ),
        "source_scan": {
            "task_count": (
                source_stats.task_count
            ),
            "rgb_image_count": (
                source_stats.rgb_image_count
            ),
            "annotation_count": (
                source_stats.annotation_count
            ),
            "unannotated_image_count": (
                source_stats
                .unannotated_image_count
            ),
            "missing_image_count": (
                source_stats
                .missing_image_count
            ),
            "per_task": (
                per_task_source
            ),
        },
        "integrity": integrity,
        "skipped": skipped,
        "warnings": warnings,
    }

    summary_path = (
        output_dir
        / "build_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n✅ 兩類 YOLO Dataset 建立完成"
    )

    print(
        f"輸出：{output_dir}"
    )

    print_statistics(
        split_state_counts,
        dict(task_stats),
        integrity,
    )

    print(
        f"\n略過：{skipped} 張"
    )

    print(
        f"dataset.yaml：{yaml_path}"
    )

    print(
        f"manifest.csv：{manifest_path}"
    )

    print(
        "build_summary.json："
        f"{summary_path}"
    )

    if warnings:
        print(
            f"警告：{len(warnings)} 項，"
            "詳見 build_summary.json"
        )

    if not integrity["ok"]:
        raise SystemExit(
            "Dataset 已建立，但完整性檢查未通過，"
            "請查看 build_summary.json。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
