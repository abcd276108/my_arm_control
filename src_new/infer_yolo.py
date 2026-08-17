from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# ============================================================
# 使用者設定
# ============================================================

MODEL_PATH = (
    r"runs\smart_teaching_tip"
    r"\tip_up_down_yolo11n"
    r"\weights\best.pt"
)

# 已經每 5 個原始影格擷取一張的 Test 圖片資料夾
SOURCE = (
    r"data\yolo_tip_state_dataset"
    r"\images\test"
)

OUTPUT_DIR = r"runs\infer_yolo"

DEVICE = "0"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7

# 原始 30 FPS 影片或攝影機：
# 每 5 個影格執行一次 YOLO
VIDEO_STRIDE = 1

# 已經抽過幀的圖片資料夾：
# 資料夾內每張都推論
IMAGE_FOLDER_STRIDE = 1

# 每 5 幀取一張，相當於 30 / 5 = 6 FPS
# 1000 / 6 ≈ 167 ms
IMAGE_DISPLAY_DELAY_MS = 33

# 畫面理論上只有一支焊槍，因此只保留最佳偵測
MAX_DETECTIONS = 1


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
}

CSV_FIELDS = [
    "source",
    "source_frame_id",
    "sample_id",
    "timestamp_s",
    "state",
    "class_id",
    "class_name",
    "confidence",
    "detection_count",
    "x_px",
    "y_px_top",
    "y_px_bottom",
    "x_norm",
    "y_norm_top",
    "y_norm_bottom",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "image_width",
    "image_height",
    "inference_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smart Teaching YOLO 焊槍尖端推論工具"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(MODEL_PATH),
    )

    parser.add_argument(
        "--source",
        default=SOURCE,
        help=(
            "圖片、圖片資料夾、影片路徑，"
            "或攝影機編號，例如 0。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "取樣間隔。未指定時，"
            "圖片資料夾自動使用 1，"
            "影片或攝影機自動使用 5。"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
    )

    parser.add_argument(
        "--device",
        default=DEVICE,
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=MAX_DETECTIONS,
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="即時顯示推論畫面。",
    )

    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="儲存標記後的圖片或影片。",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="圖片資料夾使用遞迴搜尋。",
    )

    args = parser.parse_args()

    source_text = str(args.source)
    source_path = Path(source_text).expanduser()

    is_camera = source_text.isdigit()

    is_video = (
        source_path.is_file()
        and source_path.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    if args.stride is None:
        if is_camera or is_video:
            args.stride = VIDEO_STRIDE
        else:
            args.stride = IMAGE_FOLDER_STRIDE

    if args.stride <= 0:
        parser.error(
            "--stride 必須大於 0。"
        )

    if args.imgsz <= 0:
        parser.error(
            "--imgsz 必須大於 0。"
        )

    if not 0 <= args.conf <= 1:
        parser.error(
            "--conf 必須介於 0～1。"
        )

    if not 0 <= args.iou <= 1:
        parser.error(
            "--iou 必須介於 0～1。"
        )

    return args


def resolve_class_name(
    names: dict[int, str] | list[str],
    class_id: int,
) -> str:
    if isinstance(names, dict):
        return str(
            names.get(
                class_id,
                class_id,
            )
        )

    if 0 <= class_id < len(names):
        return str(
            names[class_id]
        )

    return str(
        class_id
    )


def class_to_state(
    class_id: int,
    class_name: str,
) -> str:
    # 目前 Dataset 固定：
    # class 0 = tip_up
    # class 1 = tip_down
    if class_id == 0:
        return "UP"

    if class_id == 1:
        return "DOWN"

    lowered = class_name.lower()

    if "up" in lowered:
        return "UP"

    if "down" in lowered:
        return "DOWN"

    return class_name.upper()


def empty_prediction(
    source: str,
    frame_id: int,
    sample_id: int,
    timestamp_s: float | None,
    width: int,
    height: int,
    inference_ms: float,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_frame_id": frame_id,
        "sample_id": sample_id,
        "timestamp_s": timestamp_s,
        "state": "NONE",
        "class_id": None,
        "class_name": "none",
        "confidence": 0.0,
        "detection_count": 0,
        "x_px": None,
        "y_px_top": None,
        "y_px_bottom": None,
        "x_norm": None,
        "y_norm_top": None,
        "y_norm_bottom": None,
        "box_x1": None,
        "box_y1": None,
        "box_x2": None,
        "box_y2": None,
        "image_width": width,
        "image_height": height,
        "inference_ms": inference_ms,
    }


def extract_best_prediction(
    result: Any,
    source: str,
    frame_id: int,
    sample_id: int,
    timestamp_s: float | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    speed = getattr(
        result,
        "speed",
        {},
    ) or {}

    inference_ms = float(
        speed.get(
            "inference",
            0.0,
        )
    )

    boxes = getattr(
        result,
        "boxes",
        None,
    )

    if boxes is None or len(boxes) == 0:
        return empty_prediction(
            source=source,
            frame_id=frame_id,
            sample_id=sample_id,
            timestamp_s=timestamp_s,
            width=width,
            height=height,
            inference_ms=inference_ms,
        )

    detection_count = len(
        boxes
    )

    best_index = int(
        boxes.conf.argmax().item()
    )

    confidence = float(
        boxes.conf[
            best_index
        ].item()
    )

    class_id = int(
        boxes.cls[
            best_index
        ].item()
    )

    box = (
        boxes.xyxy[
            best_index
        ]
        .detach()
        .cpu()
        .tolist()
    )

    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]

    center_x = (
        x1 + x2
    ) / 2.0

    center_y_top = (
        y1 + y2
    ) / 2.0

    # 左下角座標系：
    # 最上方為 height - 1
    # 最下方為 0
    center_y_bottom = (
        height
        - 1
        - center_y_top
    )

    class_name = resolve_class_name(
        result.names,
        class_id,
    )

    state = class_to_state(
        class_id,
        class_name,
    )

    return {
        "source": source,
        "source_frame_id": frame_id,
        "sample_id": sample_id,
        "timestamp_s": timestamp_s,
        "state": state,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "detection_count": detection_count,
        "x_px": int(round(center_x)),
        "y_px_top": int(round(center_y_top)),
        "y_px_bottom": int(round(center_y_bottom)),
        "x_norm": center_x / width,
        "y_norm_top": center_y_top / height,
        "y_norm_bottom": center_y_bottom / height,
        "box_x1": int(round(x1)),
        "box_y1": int(round(y1)),
        "box_x2": int(round(x2)),
        "box_y2": int(round(y2)),
        "image_width": width,
        "image_height": height,
        "inference_ms": inference_ms,
    }


def draw_prediction(
    frame: Any,
    prediction: dict[str, Any],
    line_width: int,
) -> Any:
    output = frame.copy()

    state = str(
        prediction["state"]
    )

    confidence = float(
        prediction["confidence"]
    )

    if state == "NONE":
        text = "NONE"

        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    x1 = int(
        prediction["box_x1"]
    )

    y1 = int(
        prediction["box_y1"]
    )

    x2 = int(
        prediction["box_x2"]
    )

    y2 = int(
        prediction["box_y2"]
    )

    center_x = int(
        prediction["x_px"]
    )

    center_y = int(
        prediction["y_px_top"]
    )

    if state == "UP":
        color = (
            0,
            255,
            255,
        )
    else:
        color = (
            0,
            0,
            255,
        )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color,
        line_width,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        (center_x, center_y),
        5,
        color,
        -1,
        cv2.LINE_AA,
    )

    cross_size = 12

    cv2.line(
        output,
        (
            center_x - cross_size,
            center_y,
        ),
        (
            center_x + cross_size,
            center_y,
        ),
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        output,
        (
            center_x,
            center_y - cross_size,
        ),
        (
            center_x,
            center_y + cross_size,
        ),
        color,
        1,
        cv2.LINE_AA,
    )

    label = (
        f"{state} {confidence:.3f} "
        f"({center_x}, {center_y})"
    )

    text_y = max(
        25,
        y1 - 10,
    )

    cv2.putText(
        output,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    return output


def run_model(
    model: Any,
    frame: Any,
    args: argparse.Namespace,
) -> Any:
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )

    return results[0]


def save_results(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    settings: dict[str, Any],
) -> None:
    csv_path = (
        output_dir
        / "predictions.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            predictions
        )

    json_path = (
        output_dir
        / "predictions.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "settings": settings,
                "prediction_count": len(
                    predictions
                ),
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"CSV：{csv_path}"
    )

    print(
        f"JSON：{json_path}"
    )


def image_paths_from_source(
    source_dir: Path,
    recursive: bool,
) -> list[Path]:
    iterator = (
        source_dir.rglob("*")
        if recursive
        else source_dir.iterdir()
    )

    return sorted(
        [
            path
            for path in iterator
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


def infer_images(
    model: Any,
    image_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    predictions: list[
        dict[str, Any]
    ] = []

    output_image_dir = (
        output_dir
        / "images"
    )

    if args.save:
        output_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    sample_id = 0

    for image_index, image_path in enumerate(
        image_paths
    ):
        if image_index % args.stride != 0:
            continue

        frame = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            print(
                f"⚠️ 圖片讀取失敗：{image_path}"
            )
            continue

        height, width = frame.shape[:2]

        result = run_model(
            model,
            frame,
            args,
        )

        prediction = extract_best_prediction(
            result=result,
            source=str(image_path),
            frame_id=image_index,
            sample_id=sample_id,
            timestamp_s=None,
            width=width,
            height=height,
        )

        predictions.append(
            prediction
        )

        annotated = draw_prediction(
            frame,
            prediction,
            args.line_width,
        )

        if args.save:
            output_path = (
                output_image_dir
                / image_path.name
            )

            cv2.imwrite(
                str(output_path),
                annotated,
            )

        print(
            f"[{sample_id + 1}] "
            f"{image_path.name} "
            f"state={prediction['state']} "
            f"conf={prediction['confidence']:.3f} "
            f"x={prediction['x_px']} "
            f"y={prediction['y_px_top']}"
        )

        if args.show:
            cv2.imshow(
                "YOLO Inference",
                annotated,
            )

            key = (
                cv2.waitKey(
                    IMAGE_DISPLAY_DELAY_MS
                )
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

        sample_id += 1

    return predictions


def infer_video(
    model: Any,
    source: str | int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(
        source
    )

    if not capture.isOpened():
        raise SystemExit(
            f"無法開啟影片或攝影機：{source}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        source_fps <= 0
        or source_fps > 240
    ):
        source_fps = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # 只儲存被推論的幀。
    # 30 FPS / stride 5 = 6 FPS，
    # 因此輸出影片播放時間仍接近原影片。
    output_fps = max(
        1.0,
        source_fps / args.stride,
    )

    video_writer = None

    if args.save:
        output_video_path = (
            output_dir
            / (
                f"inference_stride"
                f"{args.stride}.mp4"
            )
        )

        video_writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            output_fps,
            (
                width,
                height,
            ),
        )

        if not video_writer.isOpened():
            capture.release()

            raise SystemExit(
                "無法建立輸出影片："
                f"{output_video_path}"
            )

        print(
            f"輸出影片：{output_video_path}"
        )

    predictions: list[
        dict[str, Any]
    ] = []

    source_frame_id = 0
    sample_id = 0

    camera_start_time = (
        time.perf_counter()
    )

    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        if source_frame_id % args.stride != 0:
            source_frame_id += 1
            continue

        if isinstance(
            source,
            int,
        ):
            timestamp_s = (
                time.perf_counter()
                - camera_start_time
            )
        else:
            timestamp_s = (
                source_frame_id
                / source_fps
            )

        result = run_model(
            model,
            frame,
            args,
        )

        prediction = extract_best_prediction(
            result=result,
            source=str(source),
            frame_id=source_frame_id,
            sample_id=sample_id,
            timestamp_s=timestamp_s,
            width=width,
            height=height,
        )

        predictions.append(
            prediction
        )

        annotated = draw_prediction(
            frame,
            prediction,
            args.line_width,
        )

        info = (
            f"source frame={source_frame_id} "
            f"stride={args.stride} "
            f"infer={prediction['inference_ms']:.1f} ms"
        )

        cv2.putText(
            annotated,
            info,
            (
                15,
                height - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            info,
            (
                15,
                height - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if video_writer is not None:
            video_writer.write(
                annotated
            )

        print(
            f"frame={source_frame_id:>7} "
            f"state={prediction['state']:<5} "
            f"conf={prediction['confidence']:.3f} "
            f"x={str(prediction['x_px']):>4} "
            f"y={str(prediction['y_px_top']):>4} "
            f"infer={prediction['inference_ms']:.1f} ms"
        )

        if args.show:
            cv2.imshow(
                "YOLO Inference",
                annotated,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

        sample_id += 1
        source_frame_id += 1

    capture.release()

    if video_writer is not None:
        video_writer.release()

    return predictions


def main() -> int:
    args = parse_args()

    model_path = (
        args.model
        .expanduser()
        .resolve()
    )

    if not model_path.is_file():
        raise SystemExit(
            f"找不到模型：{model_path}"
        )

    try:
        from ultralytics import YOLO

    except ImportError as exc:
        raise SystemExit(
            "尚未安裝 ultralytics：\n"
            "pip install ultralytics"
        ) from exc

    run_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(
        str(model_path)
    )

    source_text = args.source

    source_path = (
        Path(source_text)
        .expanduser()
    )

    settings = {
        "model": str(
            model_path
        ),
        "source": source_text,
        "stride": args.stride,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "max_det": args.max_det,
    }

    if source_path.is_dir():
        images = image_paths_from_source(
            source_path,
            args.recursive,
        )

        if not images:
            raise SystemExit(
                f"資料夾內沒有圖片：{source_path}"
            )

        predictions = infer_images(
            model=model,
            image_paths=images,
            output_dir=output_dir,
            args=args,
        )

    elif (
        source_path.is_file()
        and source_path.suffix.lower()
        in IMAGE_EXTENSIONS
    ):
        predictions = infer_images(
            model=model,
            image_paths=[
                source_path
            ],
            output_dir=output_dir,
            args=args,
        )

    elif (
        source_path.is_file()
        and source_path.suffix.lower()
        in VIDEO_EXTENSIONS
    ):
        predictions = infer_video(
            model=model,
            source=str(
                source_path
            ),
            output_dir=output_dir,
            args=args,
        )

    elif source_text.isdigit():
        predictions = infer_video(
            model=model,
            source=int(
                source_text
            ),
            output_dir=output_dir,
            args=args,
        )

    else:
        raise SystemExit(
            "無法判斷 source 類型："
            f"{source_text}"
        )

    cv2.destroyAllWindows()

    save_results(
        output_dir=output_dir,
        predictions=predictions,
        settings=settings,
    )

    print(
        "\n✅ 推論完成"
    )

    print(
        f"輸出資料夾：{output_dir}"
    )

    print(
        f"已處理：{len(predictions)} 個樣本"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
