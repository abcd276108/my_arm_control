"""One entry point for same-view and cross-view PCB pin inference.

Images in the same rectified coordinate system use pixel-difference refinement.
Different-angle close-ups use Qwen only for row/column semantics.  A video is
treated as a fixed-camera same-view recording: its first frame is the empty
board baseline and its last frame is the current board, unless --base-image is
supplied.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contracts import DataContractError, read_json, validate_bbox_pixel, validate_grid_map, write_json
from camera_geometry import load_camera_profile, rectify_camera_frame
from local_qwen_vl import LocalQwenVL
from project_settings import (
    DEFAULT_GRID_MAP,
    DEFAULT_CAMERA_PROFILE,
    DEFAULT_INPUT_MEDIA,
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_IMAGE,
    OUTPUT_DIR,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Default paths are centralised in project_settings.py.  Command-line paths
# override them, but the rectified reference always defaults to output/.
DEFAULT_OUTPUT = OUTPUT_DIR / "component_inference.json"


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {path}")
    return image


def detect_new_component(base_image: str | Path, current_image: str | Path, model: LocalQwenVL) -> dict:
    prompt = """Compare the two rectified PCB images. Find one newly inserted component.
Return JSON only: {"detected": boolean, "component_type": string or null,
"bbox_pixel": [x_min,y_min,x_max,y_max] or null, "confidence": number}.
bbox_pixel MUST use pixels of the second image, not normalized coordinates.
If uncertain, set detected false and bbox_pixel null."""
    result = model.generate_json([{"role": "user", "content": [
        {"type": "text", "text": prompt}, model.image_content(base_image), model.image_content(current_image),
    ]}])
    if not isinstance(result.get("detected"), bool):
        raise DataContractError("模型回覆缺少 detected 布林值")
    if result["detected"] and not (isinstance(result.get("bbox_pixel"), list) and len(result["bbox_pixel"]) == 4):
        raise DataContractError("模型宣稱偵測成功但未提供 bbox_pixel")
    return result


def find_pin_endpoints(base_crop: np.ndarray, current_crop: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if base_crop.shape != current_crop.shape or base_crop.size == 0:
        raise DataContractError("基準影像與目前影像的 ROI 必須相同且非空")
    diff = cv2.absdiff(current_crop, base_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 20]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    direction = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)[:2]
    direction /= np.linalg.norm(direction)
    center = contour.mean(axis=0)
    projection = (contour - center) @ direction
    endpoints = [center + direction * projection.min(), center + direction * projection.max()]
    return tuple(np.rint(endpoints[0]).astype(int)), tuple(np.rint(endpoints[1]).astype(int))


def locate_component_pins(base_path: str | Path, current_path: str | Path, bbox: list[float], grid_path: str | Path) -> tuple[dict, np.ndarray]:
    base, current = read_image(base_path), read_image(current_path)
    if base.shape != current.shape:
        raise DataContractError("同視角流程的基準與目前影像必須有相同尺寸")
    height, width = current.shape[:2]
    x1, y1, x2, y2 = validate_bbox_pixel(bbox, width, height)
    padding = 10
    x1p, y1p, x2p, y2p = max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)
    endpoints = find_pin_endpoints(base[y1p:y2p, x1p:x2p], current[y1p:y2p, x1p:x2p])
    if endpoints is None:
        raise DataContractError("ROI 沒有找到足夠的新增元件輪廓")
    grid = validate_grid_map(read_json(grid_path))
    result: dict[str, dict] = {}
    for name, point in zip(("pin1", "pin2"), [(x + x1p, y + y1p) for x, y in endpoints]):
        match = min(grid, key=lambda item: math.hypot(point[0] - float(item["pixel_x"]), point[1] - float(item["pixel_y"])))
        distance = math.hypot(point[0] - float(match["pixel_x"]), point[1] - float(match["pixel_y"]))
        if distance > 8.0:
            raise DataContractError(f"{name} 距離最近格點 {distance:.1f}px，超過容許值 8.0px")
        result[name] = {"pixel": list(point), "grid_row": int(match["row"]), "grid_col": int(match["col"]), "grid_distance_px": round(distance, 3)}
    if (result["pin1"]["grid_row"], result["pin1"]["grid_col"]) == (result["pin2"]["grid_row"], result["pin2"]["grid_col"]):
        raise DataContractError("兩腳被匹配到同一格點，拒絕輸出")
    return result, current


def assert_reference_calibration(grid_map_path: str | Path) -> None:
    report_path = Path(grid_map_path).with_name("calibration_report.json")
    if report_path.is_file() and not read_json(report_path).get("inference_passed", read_json(report_path).get("passed", False)):
        raise DataContractError(f"空板校正未通過：{report_path}；請重新校正後再進行跨視角映射。")


def make_labeled_grid_reference(reference_image: str | Path, grid: list[dict[str, float | int]], output_path: str | Path) -> Path:
    canvas = read_image(reference_image)
    for point in grid:
        x, y = int(round(float(point["pixel_x"]))), int(round(float(point["pixel_y"])))
        cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(canvas, f"{point['row']},{point['col']}", (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1, cv2.LINE_AA)
    write_image(output_path, canvas)
    return Path(output_path)


def map_cross_view_component(reference_image: str | Path, closeup_image: str | Path, grid: list[dict[str, float | int]], model: LocalQwenVL, labelled_reference: str | Path) -> dict[str, Any]:
    prompt = """Image A is a labelled reference of an empty PCB: every hole label is row,col.
Image B is a close-up from a different angle. Do NOT estimate pixels from Image B.
Map its two component pins to Image A's labels. Return JSON only: {"pins":[{"name":"pin1","row":int,"col":int,"confidence":0.0},{"name":"pin2","row":int,"col":int,"confidence":0.0}],"overall_confidence":0.0}."""
    response = model.generate_json([{"role": "user", "content": [{"type": "text", "text": prompt}, model.image_content(labelled_reference), model.image_content(closeup_image)]}])
    pins = response.get("pins")
    if not isinstance(pins, list) or len(pins) != 2:
        raise DataContractError("Qwen 回覆必須包含剛好兩個 pins")
    lookup = {(int(item["row"]), int(item["col"])): item for item in grid}
    resolved = []
    warnings = []
    for pin in pins:
        if not isinstance(pin, dict) or not {"name", "row", "col"}.issubset(pin):
            raise DataContractError("Qwen pin 缺少 name、row 或 col")
        key = (int(pin["row"]), int(pin["col"]))
        exact = lookup.get(key)
        if exact is None:
            raise DataContractError(f"Qwen 回傳不存在的格點 {key}；拒絕以估算座標取代")
        resolved.append({"name": str(pin["name"]), "row": key[0], "col": key[1], "confidence": pin.get("confidence"), "pixel_on_reference": [float(exact["pixel_x"]), float(exact["pixel_y"])], "coordinate_source": "exact_grid_map"})
    if resolved[0]["name"] == resolved[1]["name"] or (resolved[0]["row"], resolved[0]["col"]) == (resolved[1]["row"], resolved[1]["col"]):
        raise DataContractError("兩個 pin 必須是不同名稱與不同格點")
    return {"pins": resolved, "overall_confidence": response.get("overall_confidence"), "warnings": warnings}


def render_mapping(reference_image: str | Path, result: dict[str, Any], output_path: str | Path) -> None:
    canvas = read_image(reference_image)
    points = []
    for pin in result["pins"]:
        x, y = (int(round(value)) for value in pin["pixel_on_reference"])
        points.append((x, y))
        cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(canvas, f"{pin['name']} ({pin['row']},{pin['col']})", (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    if len(points) == 2:
        cv2.line(canvas, points[0], points[1], (0, 0, 255), 1)
    write_image(output_path, canvas)


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write images through imencode so Windows paths with Unicode work too."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"無法寫入影像: {destination}")
    encoded.tofile(str(destination))


def rectify_same_view_input(image_path: str | Path, profile: dict[str, Any], output_path: str | Path) -> Path:
    """Convert one native robot-camera frame into the reference board coordinate system."""
    image = read_image(image_path)
    rectified = rectify_camera_frame(image, profile)
    write_image(output_path, rectified)
    return Path(output_path)


def extract_video_endpoints(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the first and last decodable frames of a non-empty video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")
    first: np.ndarray | None = None
    last: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            last = frame.copy()
    finally:
        capture.release()
    if first is None or last is None:
        raise DataContractError(f"影片沒有可讀取的影格: {video_path}")
    return first, last


def select_mode(reference_image: str | Path, input_image: str | Path, requested_mode: str, *, video: bool) -> str:
    if requested_mode != "auto":
        return requested_mode
    if video:
        return "same-view"
    reference, incoming = read_image(reference_image), read_image(input_image)
    # Equal dimensions are a useful default for rectified images, but callers can
    # explicitly choose cross-view when a close-up happens to share the dimensions.
    return "same-view" if reference.shape == incoming.shape else "cross-view"


def run_preflight(reference_image: str | Path, input_media: str | Path, grid_map: str | Path,
                  model_path: str | Path, profile: dict[str, Any], *, media_is_video: bool) -> dict[str, Any]:
    """Validate all local prerequisites without loading the large Qwen model."""
    problems: list[str] = []
    for label, path in (("空板校正圖", reference_image), ("格點表", grid_map), ("待測媒體", input_media)):
        if not Path(path).is_file():
            problems.append(f"找不到{label}: {path}")
    if not problems:
        try:
            validate_grid_map(read_json(grid_map))
            assert_reference_calibration(grid_map)
            if media_is_video:
                extract_video_endpoints(input_media)
            else:
                image = read_image(input_media)
                # A native frame is the normal fixed-camera input.  A different
                # size is allowed only for an explicitly selected cross-view photo.
                if (image.shape[1], image.shape[0]) == tuple(profile["native_image_size"]):
                    rectify_camera_frame(image, profile)
        except (OSError, ValueError, DataContractError) as error:
            problems.append(str(error))
    if not Path(model_path).is_dir():
        problems.append(f"找不到本機 Qwen 模型資料夾: {model_path}")
    return {"ready": not problems, "camera_profile": profile, "problems": problems}


def run_same_view(base_image: str | Path, current_image: str | Path, grid_map: str | Path,
                  model: LocalQwenVL, overlay_path: str | Path) -> dict[str, Any]:
    detection = detect_new_component(base_image, current_image, model)
    result: dict[str, Any] = {"mode": "same-view", "detection": detection}
    if not detection["detected"]:
        result["pins"] = None
        result["message"] = "未偵測到新增元件，因此未進行腳位定位。"
        return result
    pins, overlay = locate_component_pins(base_image, current_image, detection["bbox_pixel"], grid_map)
    x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox_pixel"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for pin in pins.values():
        cv2.circle(overlay, tuple(pin["pixel"]), 4, (0, 255, 0), -1)
    write_image(overlay_path, overlay)
    result["pins"] = pins
    result["overlay"] = str(Path(overlay_path))
    return result


def run_cross_view(reference_image: str | Path, closeup_image: str | Path, grid_map: str | Path,
                   model: LocalQwenVL, labelled_reference: str | Path, overlay_path: str | Path) -> dict[str, Any]:
    # Keep the same calibration gate as the standalone cross-view command.
    assert_reference_calibration(grid_map)
    grid = validate_grid_map(read_json(grid_map))
    labelled = make_labeled_grid_reference(reference_image, grid, labelled_reference)
    result = map_cross_view_component(reference_image, closeup_image, grid, model, labelled)
    render_mapping(reference_image, result, overlay_path)
    return {"mode": "cross-view", **result, "labelled_reference": str(labelled), "overlay": str(Path(overlay_path))}


def main() -> None:
    parser = argparse.ArgumentParser(description="單一入口：照片／影片的 PCB 元件腳位推論")
    parser.add_argument("reference_image", nargs="?", default=str(DEFAULT_REFERENCE_IMAGE), help="已校正空板影像，且必須與 grid_map 使用同一座標系")
    parser.add_argument("input_media", nargs="?", default=str(DEFAULT_INPUT_MEDIA), help="目前板子照片、不同角度近照，或固定相機錄影")
    parser.add_argument("grid_map", nargs="?", default=str(DEFAULT_GRID_MAP), help="hole_location.py 產生的 grid_map.json")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="本機 Qwen2.5-VL-Instruct 模型資料夾")
    parser.add_argument("--mode", choices=("auto", "same-view", "cross-view"), default="auto")
    parser.add_argument("--base-image", help="同視角照片的空板基準圖；影片未提供時使用第一幀")
    parser.add_argument("--camera-profile", default=str(DEFAULT_CAMERA_PROFILE), help="固定相機的 640x480 標定設定")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overlay", help="結果疊圖；預設為 output 同名 .jpg")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--preflight", action="store_true", help="只檢查本機推論所需檔案與設定，不載入 Qwen")
    args = parser.parse_args()

    media_is_video = is_video(args.input_media)
    if not media_is_video and Path(args.input_media).suffix.lower() not in IMAGE_SUFFIXES:
        parser.error("input_media 必須是支援的影像或影片格式")
    if media_is_video and args.mode == "cross-view":
        parser.error("影片目前僅支援固定相機的同視角流程；請先擷取近照後使用 --mode cross-view")

    output = Path(args.output)
    overlay = Path(args.overlay) if args.overlay else output.with_suffix(".jpg")
    profile = load_camera_profile(args.camera_profile)
    if args.preflight:
        check = run_preflight(args.reference_image, args.input_media, args.grid_map, args.model_path, profile, media_is_video=media_is_video)
        write_json(output.with_name("inference_preflight.json"), check)
        print(f"本機推論前置檢查: {'READY' if check['ready'] else 'NOT READY'}；詳見 {output.with_name('inference_preflight.json')}")
        if check["problems"]:
            for problem in check["problems"]:
                print(f"- {problem}")
        return
    model = LocalQwenVL(args.model_path, device_map=args.device_map)
    if media_is_video:
        first, last = extract_video_endpoints(args.input_media)
        raw_base_path = output.with_name(output.stem + "_video_first_frame.jpg")
        raw_current_path = output.with_name(output.stem + "_video_last_frame.jpg")
        base_path = Path(args.base_image) if args.base_image else output.with_name(output.stem + "_video_first_rectified.jpg")
        current_path = output.with_name(output.stem + "_video_last_rectified.jpg")
        if not args.base_image:
            write_image(raw_base_path, first)
            rectify_same_view_input(raw_base_path, profile, base_path)
        elif read_image(base_path).shape[:2] == (profile["native_image_size"][1], profile["native_image_size"][0]):
            base_path = rectify_same_view_input(base_path, profile, output.with_name(output.stem + "_base_rectified.jpg"))
        write_image(raw_current_path, last)
        rectify_same_view_input(raw_current_path, profile, current_path)
        mode = select_mode(args.reference_image, current_path, args.mode, video=True)
        result = run_same_view(base_path, current_path, args.grid_map, model, overlay)
        result["video_frames"] = {"baseline": str(base_path), "current": str(current_path)}
    else:
        incoming = read_image(args.input_media)
        # The normal robot-camera path supplies a raw native frame; it must be
        # rectified before comparison even though it differs in size from the reference.
        native_size = tuple(profile["native_image_size"])
        mode = "same-view" if args.mode == "auto" and (incoming.shape[1], incoming.shape[0]) == native_size else select_mode(args.reference_image, args.input_media, args.mode, video=False)
        if mode == "same-view":
            current_path = rectify_same_view_input(args.input_media, profile, output.with_name(output.stem + "_rectified_input.jpg"))
            base_path = Path(args.base_image) if args.base_image else Path(args.reference_image)
            if args.base_image and read_image(base_path).shape[:2] == (profile["native_image_size"][1], profile["native_image_size"][0]):
                base_path = rectify_same_view_input(base_path, profile, output.with_name(output.stem + "_base_rectified.jpg"))
            result = run_same_view(base_path, current_path, args.grid_map, model, overlay)
            result["rectified_input"] = str(current_path)
        else:
            labelled = output.with_name(output.stem + "_grid_reference.jpg")
            result = run_cross_view(args.reference_image, args.input_media, args.grid_map, model, labelled, overlay)
    result["mode_selected"] = mode
    result["reference_image"] = str(Path(args.reference_image))
    result["input_media"] = str(Path(args.input_media))
    result["camera_profile"] = str(Path(args.camera_profile))
    write_json(output, result)
    print(f"推論完成（{mode}），結果已寫入: {output}")


if __name__ == "__main__":
    main()
