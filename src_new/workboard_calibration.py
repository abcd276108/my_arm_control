from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class CalibrationError(ValueError):
    pass


CORNER_ORDER = ("left_bottom", "right_bottom", "right_top", "left_top")
ROBOT_REFERENCE_ORDER = ("left_bottom", "right_bottom", "left_top")


def _point(value: Any, dimensions: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimensions,) or not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} 必須包含 {dimensions} 個有效數值")
    return array


def image_to_board_homography(config: dict[str, Any]) -> np.ndarray:
    width = float(config["board_width_mm"])
    height = float(config["board_height_mm"])
    if width <= 0 or height <= 0:
        raise CalibrationError("工作板長寬必須大於 0")

    corners = config.get("image_corners", {})
    source = [_point(corners.get(name), 2, f"畫面角點 {name}") for name in CORNER_ORDER]
    target = [
        np.array([0.0, 0.0]),
        np.array([width, 0.0]),
        np.array([width, height]),
        np.array([0.0, height]),
    ]

    rows: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(source, target):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
        values.append(float(x))
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
        values.append(float(y))
    try:
        solution = np.linalg.solve(np.asarray(rows), np.asarray(values))
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("四個畫面角點無法形成有效工作板平面") from exc
    return np.append(solution, 1.0).reshape(3, 3)


def pixel_to_board(config: dict[str, Any], pixel_xy: Iterable[float]) -> np.ndarray:
    u, v = _point(pixel_xy, 2, "畫面座標")
    mapped = image_to_board_homography(config) @ np.array([u, v, 1.0])
    if abs(mapped[2]) < 1e-9:
        raise CalibrationError("畫面座標無法投影到工作板")
    return mapped[:2] / mapped[2]


def board_to_robot_surface(config: dict[str, Any], board_xy: Iterable[float]) -> np.ndarray:
    x, y = _point(board_xy, 2, "工作板座標")
    width = float(config["board_width_mm"])
    height = float(config["board_height_mm"])
    refs = config.get("robot_reference_points", {})
    origin = _point(refs.get("left_bottom"), 3, "JetArm 左下角")
    right = _point(refs.get("right_bottom"), 3, "JetArm 右下角")
    top = _point(refs.get("left_top"), 3, "JetArm 左上角")
    return origin + (x / width) * (right - origin) + (y / height) * (top - origin)


def board_normal(config: dict[str, Any]) -> np.ndarray:
    refs = config.get("robot_reference_points", {})
    origin = _point(refs.get("left_bottom"), 3, "JetArm 左下角")
    right = _point(refs.get("right_bottom"), 3, "JetArm 右下角")
    top = _point(refs.get("left_top"), 3, "JetArm 左上角")
    normal = np.cross(right - origin, top - origin)
    length = float(np.linalg.norm(normal))
    if length < 1e-6:
        raise CalibrationError("JetArm三個參考點不能位於同一直線")
    normal /= length
    if normal[2] < 0:
        normal *= -1
    return normal


def pixel_to_robot(config: dict[str, Any], pixel_xy: Iterable[float], state: str) -> np.ndarray:
    surface = board_to_robot_surface(config, pixel_to_board(config, pixel_xy))
    heights = config.get("heights_mm", {})
    state_name = state.strip().upper()
    if state_name == "DOWN":
        offset = float(heights.get("down_dry_run_mm", 30.0))
    elif state_name == "UP":
        offset = float(heights.get("up_safe_mm", 80.0))
    else:
        raise CalibrationError("狀態必須是 UP 或 DOWN")
    return surface + board_normal(config) * offset


def calibration_status(config: dict[str, Any]) -> str:
    try:
        image_to_board_homography(config)
    except (CalibrationError, KeyError, TypeError, ValueError):
        return "empty"
    try:
        board_normal(config)
    except (CalibrationError, KeyError, TypeError, ValueError):
        return "image_only"
    return "ready"


def classify_down_segment(points_xy_mm: Iterable[Iterable[float]], config: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(list(points_xy_mm), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise CalibrationError("DOWN區段至少需要一個二維座標")
    if not np.all(np.isfinite(points)):
        raise CalibrationError("DOWN區段包含無效座標")

    rules = config.get("process_detection", {})
    line_min = float(rules.get("line_min_displacement_mm", 15.0))
    point_max = float(rules.get("point_max_spread_mm", 8.0))
    centre = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - centre, axis=1)))
    displacement = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
    process = "line_weld" if displacement >= line_min or spread > point_max else "spot_weld"
    return {
        "process": process,
        "point_count": int(len(points)),
        "centre_xy_mm": [round(float(value), 3) for value in centre],
        "spread_mm": round(spread, 3),
        "displacement_mm": round(displacement, 3),
        "path_length_mm": round(path_length, 3),
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized["schema_version"] = 1
    normalized["board_name"] = str(config.get("board_name", "competition_workboard")).strip() or "competition_workboard"
    normalized["board_width_mm"] = float(config["board_width_mm"])
    normalized["board_height_mm"] = float(config["board_height_mm"])
    normalized.setdefault("heights_mm", {"down_dry_run_mm": 30.0, "up_safe_mm": 80.0})
    normalized.setdefault("process_detection", {"line_min_displacement_mm": 15.0, "point_max_spread_mm": 8.0})
    normalized["calibration_status"] = calibration_status(normalized)
    if normalized["calibration_status"] == "empty":
        raise CalibrationError("請輸入工作板長寬並依序標記四個角")
    return normalized
