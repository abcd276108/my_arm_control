"""Rectify the workboard image and create the shared pixel grid map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from contracts import write_json
from camera_geometry import load_camera_profile, rectify_camera_frame
from project_settings import DEFAULT_CAMERA_PROFILE, DEFAULT_EMPTY_BOARD_IMAGE, INPUT_DIR, OUTPUT_DIR

DEFAULT_CORNERS = np.array([[154, 148], [498, 148], [497, 350], [154, 351]], dtype=np.float32)


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {path}")
    return image


def build_grid(rows: int, cols: int, width: int, height: int, margin_x: float, margin_y: float) -> list[dict[str, float | int]]:
    if rows < 2 or cols < 2:
        raise ValueError("rows 與 cols 至少必須為 2")
    if not (0 <= margin_x * 2 < width and 0 <= margin_y * 2 < height):
        raise ValueError("格點邊界超出校正影像")
    return [
        {"row": row, "col": col,
         "pixel_x": round(margin_x + col * (width - 1 - 2 * margin_x) / (cols - 1), 3),
         "pixel_y": round(margin_y + row * (height - 1 - 2 * margin_y) / (rows - 1), 3)}
        for row in range(rows) for col in range(cols)
    ]


def detect_hole_centers(rectified: np.ndarray) -> np.ndarray:
    """Find candidate plated-through-hole centres in the rectified board image."""
    gray = cv2.medianBlur(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.1, minDist=8,
                               param1=70, param2=10, minRadius=2, maxRadius=6)
    return np.empty((0, 2), dtype=np.float32) if circles is None else circles[0, :, :2].astype(np.float32)


def assign_detections(predicted: np.ndarray, detected: np.ndarray, max_distance_px: float) -> dict[int, int]:
    """Globally greedy one-to-one assignment, avoiding one circle serving two grid cells."""
    candidates: list[tuple[float, int, int]] = []
    for grid_index, point in enumerate(predicted):
        for detected_index, center in enumerate(detected):
            distance = float(np.linalg.norm(point - center))
            if distance <= max_distance_px:
                candidates.append((distance, grid_index, detected_index))
    matches: dict[int, int] = {}
    used: set[int] = set()
    for _, grid_index, detected_index in sorted(candidates):
        if grid_index not in matches and detected_index not in used:
            matches[grid_index] = detected_index
            used.add(detected_index)
    return matches


def local_hole_center(gray: np.ndarray, predicted: np.ndarray, radius: int = 8) -> np.ndarray | None:
    """A second-pass detector for a known grid cell missed by the global Hough pass."""
    x, y = (int(round(float(value))) for value in predicted)
    left, top = max(0, x - radius), max(0, y - radius)
    patch = gray[top:min(gray.shape[0], y + radius + 1), left:min(gray.shape[1], x + radius + 1)]
    circles = cv2.HoughCircles(patch, cv2.HOUGH_GRADIENT, dp=1, minDist=6,
                               param1=50, param2=3, minRadius=2, maxRadius=6)
    if circles is None:
        return None
    candidates = circles[0, :, :2] + np.array([left, top], dtype=np.float32)
    best = min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - predicted)))
    return best.astype(np.float32) if np.linalg.norm(best - predicted) <= radius else None


def refine_grid_with_holes(grid: list[dict[str, float | int]], rectified: np.ndarray, *, max_distance_px: float = 7.0,
                           inference_min_match_ratio: float = 0.85, motion_min_match_ratio: float = 0.95) -> tuple[list[dict[str, float | int]], list[dict], dict]:
    """Fit a board correction from measured holes and report each cell's residual.

    Missing holes retain a model prediction but are explicitly marked ``measured: false``.
    Consumers should reject a calibration whose report is not passed.
    """
    detected = detect_hole_centers(rectified)
    nominal = np.array([[float(point["pixel_x"]), float(point["pixel_y"])] for point in grid], dtype=np.float32)
    first_matches = assign_detections(nominal, detected, max_distance_px)
    corrected = nominal.copy()
    if len(first_matches) >= 4:
        source = nominal[list(first_matches)]
        destination = detected[[first_matches[index] for index in first_matches]]
        transform, _ = cv2.findHomography(source, destination, cv2.RANSAC, 2.0)
        if transform is not None:
            corrected = cv2.perspectiveTransform(nominal[None, :, :], transform)[0]
    matches = assign_detections(corrected, detected, max_distance_px)
    centers: dict[int, np.ndarray] = {index: detected[detected_index] for index, detected_index in matches.items()}
    gray = cv2.medianBlur(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY), 5)
    local_matches = 0
    for index, predicted in enumerate(corrected):
        if index not in centers:
            center = local_hole_center(gray, predicted)
            if center is not None:
                centers[index] = center
                local_matches += 1
    measured_residuals: list[float] = []
    refined: list[dict[str, float | int]] = []
    measurements: list[dict] = []
    for index, point in enumerate(grid):
        predicted = corrected[index]
        measured = index in centers
        center = centers[index] if measured else predicted
        residual = float(np.linalg.norm(center - predicted)) if measured else None
        if residual is not None:
            measured_residuals.append(residual)
        refined.append({"row": int(point["row"]), "col": int(point["col"]),
                        "pixel_x": round(float(center[0]), 3), "pixel_y": round(float(center[1]), 3)})
        measurements.append({"row": int(point["row"]), "col": int(point["col"]),
                             "predicted_pixel": [round(float(predicted[0]), 3), round(float(predicted[1]), 3)],
                             "pixel": [round(float(center[0]), 3), round(float(center[1]), 3)],
                             "measured": measured,
                             "residual_px": None if residual is None else round(residual, 3)})
    match_ratio = len(centers) / len(grid)
    report = {"expected_holes": len(grid), "detected_candidates": len(detected), "matched_holes": len(centers), "local_refinement_matches": local_matches,
              "match_ratio": round(match_ratio, 4), "mean_residual_px": round(float(np.mean(measured_residuals)), 3) if measured_residuals else None,
              "rms_residual_px": round(float(math.sqrt(np.mean(np.square(measured_residuals)))), 3) if measured_residuals else None,
              "max_residual_px": round(float(max(measured_residuals)), 3) if measured_residuals else None,
              "inference_passed": bool(match_ratio >= inference_min_match_ratio and measured_residuals and max(measured_residuals) <= 3.0),
              "motion_passed": bool(match_ratio >= motion_min_match_ratio and measured_residuals and max(measured_residuals) <= 3.0),
              "passed": bool(match_ratio >= inference_min_match_ratio and measured_residuals and max(measured_residuals) <= 3.0),
              "pass_criteria": {"inference_min_match_ratio": inference_min_match_ratio,
                                "motion_min_match_ratio": motion_min_match_ratio, "max_residual_px": 3.0}}
    return refined, measurements, report


def calibrate_board(image_path: str | Path, output_dir: str | Path, *, corners: np.ndarray = DEFAULT_CORNERS,
                    width: int = 344, height: int = 203, rows: int = 14, cols: int = 24,
                    margin_x: float = 10.0, margin_y: float = 9.0, inference_min_match_ratio: float = 0.85) -> list[dict[str, float | int]]:
    """Create a rectified image, diagnostic overlay, and ``grid_map.json``.

    Corners are ordered top-left, top-right, bottom-right, bottom-left.  The supplied
    defaults are calibrated for the included sample only; measure them again for a new camera setup.
    """
    image = read_image(image_path)
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("corners 必須是四個 [x, y] 點")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    rectified = cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, target), (width, height))
    nominal_grid = build_grid(rows, cols, width, height, margin_x, margin_y)
    grid, measurements, report = refine_grid_with_holes(nominal_grid, rectified, inference_min_match_ratio=inference_min_match_ratio)

    overlay = rectified.copy()
    for point, measurement in zip(grid, measurements):
        x, y = round(float(point["pixel_x"])), round(float(point["pixel_y"]))
        if not measurement["measured"]:
            colour = (0, 165, 255)  # orange: model prediction only
        elif measurement["residual_px"] > 3.0:
            colour = (0, 0, 255)  # red: measured but outside tolerance
        else:
            colour = (0, 255, 0)  # green: measured within tolerance
        cv2.circle(overlay, (x, y), 2, colour, -1)
        cv2.putText(overlay, f"{point['row']},{point['col']}", (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1, cv2.LINE_AA)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "board_rectified.jpg"), rectified)
    cv2.imwrite(str(output / "grid_result.jpg"), overlay)
    write_json(output / "grid_map.json", grid)
    write_json(output / "grid_measurements.json", measurements)
    write_json(output / "calibration_report.json", report)
    return grid


def ensure_input_folder() -> bool:
    """Create the shared input folder and tell first-time users what belongs in it."""
    created = not INPUT_DIR.exists()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if created:
        print(f"已建立輸入資料夾: {INPUT_DIR}")
        print("請將原始空板照片放入 input/empty_board.jpg，並將待測照片或影片放入 input/。")
        print("完成後重新執行；校正結果會全部寫入 output/。")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="校正 PCB 並產生 grid_map.json")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_EMPTY_BOARD_IMAGE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--camera-profile", default=str(DEFAULT_CAMERA_PROFILE), help="固定相機的 640x480 標定設定")
    parser.add_argument("--rows", type=int, help="覆寫 profile 的格點列數")
    parser.add_argument("--cols", type=int, help="覆寫 profile 的格點欄數")
    args = parser.parse_args()
    ensure_input_folder()
    if Path(args.image) == DEFAULT_EMPTY_BOARD_IMAGE and not DEFAULT_EMPTY_BOARD_IMAGE.is_file():
        parser.error(
            f"找不到預設空板照片: {DEFAULT_EMPTY_BOARD_IMAGE}。"
            "請匯入原始空板照片並命名為 empty_board.jpg；待測照片／影片也請放入 input/。"
        )
    profile = load_camera_profile(args.camera_profile)
    # Validate the source image at native camera resolution before using its profile corners.
    source = read_image(args.image)
    rectify_camera_frame(source, profile)
    rows, cols = profile["grid_shape"]
    margin_x, margin_y = profile["grid_margins_px"]
    grid = calibrate_board(args.image, args.output_dir, corners=np.asarray(profile["board_corners_px"], dtype=np.float32),
                           width=profile["rectified_size"][0], height=profile["rectified_size"][1],
                           rows=args.rows or rows, cols=args.cols or cols, margin_x=margin_x, margin_y=margin_y,
                           inference_min_match_ratio=float(profile.get("inference_min_match_ratio", 0.85)))
    write_json(Path(args.output_dir) / "camera_profile_used.json", profile)
    report_path = Path(args.output_dir) / "calibration_report.json"
    import json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"已建立 {len(grid)} 個格點，匹配率 {report['match_ratio']:.1%}，RMS {report['rms_residual_px']} px")
    print(f"電腦推論校正: {'PASS' if report['inference_passed'] else 'FAIL'}；實機動作校正: {'PASS' if report['motion_passed'] else 'FAIL'}；詳見 {report_path}")


if __name__ == "__main__":
    main()
