"""Fit and validate the 2-D camera-pixel to robot-base XY calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from camera_geometry import load_camera_profile
from contracts import DataContractError, read_json, write_json
from project_settings import DEFAULT_CAMERA_PROFILE, DEFAULT_ROBOT_CALIBRATION, OUTPUT_DIR


def fit_pixel_to_arm(correspondences: list[dict], *, min_points: int = 6) -> dict:
    if not isinstance(correspondences, list) or len(correspondences) < min_points:
        raise DataContractError(f"至少需要 {min_points} 個 pixel/arm_xy 對應點")
    try:
        pixels = np.asarray([item["pixel"] for item in correspondences], dtype=np.float32)
        arm_xy = np.asarray([item["arm_xy_mm"] for item in correspondences], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise DataContractError("每個對應點必須是 pixel:[x,y] 與 arm_xy_mm:[X,Y]") from error
    if pixels.shape != arm_xy.shape or pixels.shape[1:] != (2,):
        raise DataContractError("pixel 與 arm_xy_mm 均必須是兩個數字")
    matrix, mask = cv2.findHomography(pixels, arm_xy, cv2.RANSAC, 1.0)
    if matrix is None or mask is None:
        raise DataContractError("無法由對應點求得像素到手臂的 Homography")
    projected = cv2.perspectiveTransform(pixels[None, :, :], matrix)[0]
    errors = np.linalg.norm(projected - arm_xy, axis=1)
    inliers = mask.reshape(-1).astype(bool)
    return {
        "pixel_to_arm_homography": np.round(matrix, 10).tolist(),
        "correspondence_count": int(len(pixels)), "inlier_count": int(inliers.sum()),
        "rms_error_mm": round(float(np.sqrt(np.mean(np.square(errors)))), 4),
        "max_error_mm": round(float(errors.max()), 4),
        "passed": bool(inliers.sum() >= min_points and np.sqrt(np.mean(np.square(errors))) <= 1.0 and errors.max() <= 2.0),
        "pass_criteria": {"min_inlier_count": min_points, "max_rms_error_mm": 1.0, "max_error_mm": 2.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="以實測點標定相機像素到機械臂 base XY（mm）")
    parser.add_argument("correspondences", help="JSON 陣列，每筆為 pixel 與 arm_xy_mm")
    parser.add_argument("--camera-profile", default=str(DEFAULT_CAMERA_PROFILE))
    parser.add_argument("--surface-z", type=float, required=True, help="PCB 表面在機械臂 base 座標的 Z（mm）")
    parser.add_argument("--safe-height", type=float, default=20.0, help="安全抬升高度（mm）")
    parser.add_argument("--workspace", nargs=4, type=float, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--output", default=str(DEFAULT_ROBOT_CALIBRATION))
    args = parser.parse_args()
    profile = load_camera_profile(args.camera_profile)
    result = fit_pixel_to_arm(read_json(args.correspondences))
    result.update({"camera_profile": str(Path(args.camera_profile)), "native_image_size": profile["native_image_size"],
                   "z_board_surface_mm": args.surface_z, "safe_height_offset_mm": args.safe_height,
                   "workspace_xy_mm": args.workspace})
    write_json(args.output, result)
    print(f"手眼平面標定: {'PASS' if result['passed'] else 'FAIL'}，RMS={result['rms_error_mm']} mm，最大誤差={result['max_error_mm']} mm")
    print(f"結果已寫入: {args.output}")


if __name__ == "__main__":
    main()
