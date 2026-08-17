from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable


# ============================================================
# 參數
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "訓練 tip_up / tip_down "
            "兩類 Ultralytics YOLO Detect 模型"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "data"
            / "yolo_tip_state_dataset"
            / "dataset.yaml"
        ),
        help="build_yolo_dataset.py 產生的 dataset.yaml。",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="預訓練權重，或 Resume 使用的 last.pt。",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="0",
        help="例如 0、cpu；空字串代表自動選擇。",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=(
            Path("runs")
            / "smart_teaching_tip"
        ),
    )

    parser.add_argument(
        "--name",
        default="tip_up_down_yolo11n",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--optimizer",
        default="AdamW",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help=(
            "初始學習率。未指定時，"
            "使用 Ultralytics 預設值。"
        ),
    )

    parser.add_argument(
        "--lrf",
        type=float,
        default=None,
        help=(
            "最終學習率比例。未指定時，"
            "使用 Ultralytics 預設值。"
        ),
    )

    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "快取資料集到 RAM；"
            "RAM 不足時使用 --no-cache。"
        ),
    )

    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="輸出 Ultralytics 原生圖表。",
    )

    parser.add_argument(
        "--extra-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "從 results.csv 額外產生 "
            "Loss、Precision、Recall、mAP 圖。"
        ),
    )

    parser.add_argument(
        "--metric-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "額外保存最佳 Precision、Recall、"
            "mAP50 與 mAP50-95 權重。"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "從 --model 指定的 last.pt "
            "繼續訓練。"
        ),
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="允許使用既有 project/name 資料夾。",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error(
            "--epochs 必須大於 0。"
        )

    if args.imgsz <= 0:
        parser.error(
            "--imgsz 必須大於 0。"
        )

    if args.batch == 0:
        parser.error(
            "--batch 不可為 0。"
        )

    if args.workers < 0:
        parser.error(
            "--workers 不可小於 0。"
        )

    if args.patience < 0:
        parser.error(
            "--patience 不可小於 0。"
        )

    if args.close_mosaic < 0:
        parser.error(
            "--close-mosaic 不可小於 0。"
        )

    if (
        args.save_period == 0
        or args.save_period < -1
    ):
        parser.error(
            "--save-period 必須為 -1 或正整數。"
        )

    if (
        args.lr0 is not None
        and args.lr0 <= 0
    ):
        parser.error(
            "--lr0 必須大於 0。"
        )

    if (
        args.lrf is not None
        and args.lrf <= 0
    ):
        parser.error(
            "--lrf 必須大於 0。"
        )

    return args


# ============================================================
# Metric checkpoint
# ============================================================

def to_float(
    value: Any,
) -> float | None:
    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def find_metric(
    metrics: dict[str, Any],
    candidates: tuple[str, ...],
) -> float | None:
    normalized = {
        str(key).strip(): value
        for key, value
        in metrics.items()
    }

    for candidate in candidates:
        if candidate in normalized:
            return to_float(
                normalized[candidate]
            )

    lowered = {
        key.lower(): value
        for key, value
        in normalized.items()
    }

    for candidate in candidates:
        candidate_lower = (
            candidate.lower()
        )

        for key, value in (
            lowered.items()
        ):
            if candidate_lower in key:
                result = to_float(
                    value
                )

                if result is not None:
                    return result

    return None


def make_metric_checkpoint_callback(
) -> Callable[[Any], None]:
    best_values = {
        "precision": float(
            "-inf"
        ),
        "recall": float(
            "-inf"
        ),
        "map50": float(
            "-inf"
        ),
        "map50_95": float(
            "-inf"
        ),
    }

    metric_keys = {
        "precision": (
            "metrics/precision(B)",
            "metrics/precision",
        ),
        "recall": (
            "metrics/recall(B)",
            "metrics/recall",
        ),
        "map50": (
            "metrics/mAP50(B)",
            "metrics/mAP50",
        ),
        "map50_95": (
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ),
    }

    output_names = {
        "precision": (
            "best_precision.pt"
        ),
        "recall": (
            "best_recall.pt"
        ),
        "map50": (
            "best_map50.pt"
        ),
        "map50_95": (
            "best_map50_95.pt"
        ),
    }

    initialized = False

    def save_metric_checkpoints(
        trainer: Any,
    ) -> None:
        nonlocal initialized

        metrics = getattr(
            trainer,
            "metrics",
            None,
        )

        if not isinstance(
            metrics,
            dict,
        ):
            return

        checkpoint = Path(
            getattr(
                trainer,
                "last",
                "",
            )
        )

        if not checkpoint.is_file():
            return

        save_dir = Path(
            getattr(
                trainer,
                "save_dir",
                checkpoint.parent.parent,
            )
        )

        weights_dir = (
            checkpoint.parent
        )

        record_path = (
            save_dir
            / "metric_best_checkpoints.json"
        )

        # Resume 時讀取先前最佳結果，避免被較差結果覆蓋。
        if (
            not initialized
            and record_path.is_file()
        ):
            try:
                existing = json.loads(
                    record_path.read_text(
                        encoding="utf-8"
                    )
                )

                for metric_name in (
                    best_values
                ):
                    item = existing.get(
                        metric_name
                    )

                    if isinstance(
                        item,
                        dict,
                    ):
                        previous_value = (
                            to_float(
                                item.get(
                                    "value"
                                )
                            )
                        )

                        if (
                            previous_value
                            is not None
                        ):
                            best_values[
                                metric_name
                            ] = previous_value

            except (
                OSError,
                json.JSONDecodeError,
            ):
                pass

        initialized = True

        epoch = (
            int(
                getattr(
                    trainer,
                    "epoch",
                    -1,
                )
            )
            + 1
        )

        changed = False

        for (
            metric_name,
            candidates,
        ) in metric_keys.items():
            value = find_metric(
                metrics,
                candidates,
            )

            if value is None:
                continue

            if (
                value
                <= best_values[
                    metric_name
                ]
            ):
                continue

            best_values[
                metric_name
            ] = value

            destination = (
                weights_dir
                / output_names[
                    metric_name
                ]
            )

            shutil.copy2(
                checkpoint,
                destination,
            )

            changed = True

            print(
                f"\n⭐ {metric_name} 新最佳值 "
                f"{value:.6f} "
                f"(epoch {epoch}) "
                f"→ {destination.name}"
            )

        if not changed:
            return

        record: dict[str, Any] = {
            metric_name: {
                "value": value,
                "checkpoint": str(
                    (
                        weights_dir
                        / output_names[
                            metric_name
                        ]
                    ).resolve()
                ),
            }
            for (
                metric_name,
                value,
            ) in best_values.items()
            if math.isfinite(
                value
            )
        }

        record[
            "last_updated_epoch"
        ] = epoch

        record_path.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return save_metric_checkpoints


# ============================================================
# Results CSV
# ============================================================

def read_results_csv(
    path: Path,
) -> tuple[
    list[int],
    dict[str, list[float]],
]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "results.csv 沒有欄位名稱。"
            )

        field_map = {
            name.strip(): name
            for name
            in reader.fieldnames
        }

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "results.csv 沒有資料。"
        )

    epoch_key = field_map.get(
        "epoch"
    )

    epochs: list[int] = []

    for index, row in enumerate(
        rows
    ):
        if epoch_key is None:
            epochs.append(
                index + 1
            )
            continue

        epoch_value = to_float(
            row.get(
                epoch_key
            )
        )

        if epoch_value is None:
            epochs.append(
                index + 1
            )
        else:
            epochs.append(
                int(epoch_value)
                + 1
            )

    columns: dict[
        str,
        list[float],
    ] = {}

    for (
        stripped_name,
        original_name,
    ) in field_map.items():
        if stripped_name == "epoch":
            continue

        values: list[float] = []
        valid_count = 0

        for row in rows:
            value = to_float(
                row.get(
                    original_name
                )
            )

            if value is None:
                values.append(
                    float("nan")
                )
            else:
                values.append(
                    value
                )
                valid_count += 1

        if valid_count > 0:
            columns[
                stripped_name
            ] = values

    return (
        epochs,
        columns,
    )


# ============================================================
# 額外圖表
# ============================================================

def plot_selected_columns(
    epochs: list[int],
    columns: dict[
        str,
        list[float],
    ],
    selected: list[
        tuple[str, str]
    ],
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "⚠️ 未安裝 matplotlib，"
            "略過額外訓練曲線。"
        )
        return False

    plotted_count = 0

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for (
        column_name,
        label,
    ) in selected:
        values = columns.get(
            column_name
        )

        if values is None:
            continue

        plt.plot(
            epochs,
            values,
            label=label,
        )

        plotted_count += 1

    if plotted_count == 0:
        plt.close()
        return False

    plt.title(
        title
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        ylabel
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    if plotted_count > 1:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    return True


def generate_extra_plots(
    results_csv: Path,
    save_dir: Path,
) -> list[Path]:
    epochs, columns = read_results_csv(
        results_csv
    )

    plots_dir = (
        save_dir
        / "custom_plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_specs = [
        {
            "columns": [
                (
                    "train/box_loss",
                    "Train Box Loss",
                ),
                (
                    "train/cls_loss",
                    "Train Class Loss",
                ),
                (
                    "train/dfl_loss",
                    "Train DFL Loss",
                ),
                (
                    "val/box_loss",
                    "Validation Box Loss",
                ),
                (
                    "val/cls_loss",
                    "Validation Class Loss",
                ),
                (
                    "val/dfl_loss",
                    "Validation DFL Loss",
                ),
            ],
            "title": (
                "Training and Validation Loss"
            ),
            "ylabel": "Loss",
            "filename": "loss.png",
        },
        {
            "columns": [
                (
                    "metrics/precision(B)",
                    "Precision",
                )
            ],
            "title": "Precision",
            "ylabel": "Precision",
            "filename": "precision.png",
        },
        {
            "columns": [
                (
                    "metrics/recall(B)",
                    "Recall",
                )
            ],
            "title": "Recall",
            "ylabel": "Recall",
            "filename": "recall.png",
        },
        {
            "columns": [
                (
                    "metrics/mAP50(B)",
                    "mAP50",
                )
            ],
            "title": "mAP50",
            "ylabel": "mAP50",
            "filename": "mAP50.png",
        },
        {
            "columns": [
                (
                    "metrics/mAP50-95(B)",
                    "mAP50-95",
                )
            ],
            "title": "mAP50-95",
            "ylabel": "mAP50-95",
            "filename": "mAP50-95.png",
        },
    ]

    created_paths: list[
        Path
    ] = []

    for spec in plot_specs:
        output_path = (
            plots_dir
            / str(
                spec["filename"]
            )
        )

        created = plot_selected_columns(
            epochs=epochs,
            columns=columns,
            selected=spec["columns"],
            title=str(
                spec["title"]
            ),
            ylabel=str(
                spec["ylabel"]
            ),
            output_path=output_path,
        )

        if created:
            created_paths.append(
                output_path
            )

    return created_paths


# ============================================================
# 主程式
# ============================================================

def main() -> int:
    args = parse_args()

    data_path = (
        args.data
        .expanduser()
        .resolve()
    )

    project_path = (
        args.project
        .expanduser()
        .resolve()
    )

    if not data_path.is_file():
        raise SystemExit(
            f"找不到 dataset.yaml：{data_path}"
        )

    if args.resume:
        resume_path = (
            Path(args.model)
            .expanduser()
            .resolve()
        )

        if not resume_path.is_file():
            raise SystemExit(
                "使用 --resume 時，"
                "--model 必須指向存在的 last.pt。\n"
                f"目前指定：{resume_path}"
            )

        model_source = str(
            resume_path
        )

    else:
        model_source = (
            args.model
        )

    try:
        from ultralytics import YOLO

    except ImportError as exc:
        raise SystemExit(
            "尚未安裝 ultralytics，"
            "請先執行：\n"
            "pip install ultralytics"
        ) from exc

    model = YOLO(
        model_source
    )

    if args.metric_checkpoints:
        model.add_callback(
            "on_model_save",
            make_metric_checkpoint_callback(),
        )

    train_options: dict[
        str,
        Any,
    ] = {
        "data": str(
            data_path
        ),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(
            project_path
        ),
        "name": args.name,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "cache": args.cache,
        "cos_lr": args.cos_lr,
        "close_mosaic": (
            args.close_mosaic
        ),
        "amp": args.amp,
        "plots": args.plots,
        "save_period": (
            args.save_period
        ),
        "seed": args.seed,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
    }

    if args.device != "":
        train_options[
            "device"
        ] = args.device

    if args.lr0 is not None:
        train_options[
            "lr0"
        ] = args.lr0

    if args.lrf is not None:
        train_options[
            "lrf"
        ] = args.lrf

    print(
        "=" * 72
    )

    print(
        "Smart Teaching YOLO Training"
    )

    print(
        "=" * 72
    )

    print(
        f"Data       : {data_path}"
    )

    print(
        f"Model      : {model_source}"
    )

    print(
        f"Epochs     : {args.epochs}"
    )

    print(
        f"Image size : {args.imgsz}"
    )

    print(
        f"Batch      : {args.batch}"
    )

    print(
        f"Optimizer  : {args.optimizer}"
    )

    print(
        "LR0        : "
        + (
            str(args.lr0)
            if args.lr0 is not None
            else "Ultralytics default"
        )
    )

    print(
        f"Cache      : {args.cache}"
    )

    print(
        f"AMP        : {args.amp}"
    )

    print(
        f"Cosine LR  : {args.cos_lr}"
    )

    print(
        f"Resume     : {args.resume}"
    )

    print(
        "=" * 72
    )

    model.train(
        **train_options
    )

    trainer = getattr(
        model,
        "trainer",
        None,
    )

    fallback_dir = (
        project_path
        / args.name
    )

    save_dir = Path(
        getattr(
            trainer,
            "save_dir",
            fallback_dir,
        )
    ).resolve()

    weights_dir = (
        save_dir
        / "weights"
    )

    best_path = (
        weights_dir
        / "best.pt"
    )

    last_path = (
        weights_dir
        / "last.pt"
    )

    results_csv = (
        save_dir
        / "results.csv"
    )

    extra_plot_paths: list[
        Path
    ] = []

    if (
        args.extra_plots
        and results_csv.is_file()
    ):
        try:
            extra_plot_paths = (
                generate_extra_plots(
                    results_csv,
                    save_dir,
                )
            )

        except (
            OSError,
            ValueError,
        ) as exc:
            print(
                "⚠️ 額外圖表建立失敗："
                f"{exc}"
            )

    print(
        "\n✅ tip_up / tip_down 模型訓練完成"
    )

    print(
        f"輸出資料夾：{save_dir}"
    )

    print(
        "best.pt："
        + (
            str(best_path)
            if best_path.exists()
            else "未找到"
        )
    )

    print(
        "last.pt："
        + (
            str(last_path)
            if last_path.exists()
            else "未找到"
        )
    )

    print(
        "results.csv："
        + (
            str(results_csv)
            if results_csv.exists()
            else "未找到"
        )
    )

    metric_files = [
        (
            weights_dir
            / "best_precision.pt"
        ),
        (
            weights_dir
            / "best_recall.pt"
        ),
        (
            weights_dir
            / "best_map50.pt"
        ),
        (
            weights_dir
            / "best_map50_95.pt"
        ),
    ]

    for path in metric_files:
        if path.exists():
            print(
                f"Metric checkpoint：{path}"
            )

    if extra_plot_paths:
        print(
            "額外圖表："
        )

        for path in extra_plot_paths:
            print(
                f"  {path}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )