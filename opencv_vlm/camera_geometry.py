"""Shared geometry for the fixed 640x480 camera mounted on the robot arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contracts import DataContractError
from project_settings import DEFAULT_CAMERA_PROFILE


def load_camera_profile(path: str | Path = DEFAULT_CAMERA_PROFILE) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"找不到相機設定檔: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    try:
        native_width, native_height = (int(value) for value in profile["native_image_size"])
        rectified_width, rectified_height = (int(value) for value in profile["rectified_size"])
        corners = np.asarray(profile["board_corners_px"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise DataContractError(f"相機設定格式錯誤: {profile_path}") from error
    if native_width <= 0 or native_height <= 0 or rectified_width <= 0 or rectified_height <= 0 or corners.shape != (4, 2):
        raise DataContractError(f"相機設定內容無效: {profile_path}")
    profile["native_image_size"] = [native_width, native_height]
    profile["rectified_size"] = [rectified_width, rectified_height]
    profile["board_corners_px"] = corners.tolist()
    return profile


def assert_native_camera_frame(image: np.ndarray, profile: dict[str, Any], *, image_name: str = "影像") -> None:
    """Reject scaled/cropped camera frames; calibration is only valid at native size."""
    expected = tuple(profile["native_image_size"])
    actual = (int(image.shape[1]), int(image.shape[0]))
    if actual != expected:
        raise DataContractError(
            f"{image_name} 尺寸為 {actual[0]}x{actual[1]}，但相機標定要求原生 {expected[0]}x{expected[1]}。"
            "請關閉相機端縮放／裁切，或重新標定 camera_profile.json。"
        )


def rectify_camera_frame(image: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    assert_native_camera_frame(image, profile)
    width, height = profile["rectified_size"]
    source = np.asarray(profile["board_corners_px"], dtype=np.float32)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (width, height))
