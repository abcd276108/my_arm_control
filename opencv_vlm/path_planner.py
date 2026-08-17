"""Convert a validated welding plan into safe, simulator-ready Cartesian waypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contracts import DataContractError, read_json, validate_grid_map, write_json
from project_settings import DEFAULT_ROBOT_CALIBRATION, OUTPUT_DIR


class ArmCoordinateConverter:
    def __init__(self, homography_matrix: np.ndarray, z_board_surface: float) -> None:
        matrix = np.asarray(homography_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography_matrix 必須是有限值 3x3 矩陣")
        self.H, self.z_surface = matrix, float(z_board_surface)

    def pixel_to_arm_3d(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        point = self.H @ np.array([pixel_x, pixel_y, 1.0])
        if abs(point[2]) < 1e-9:
            raise DataContractError(f"像素 {(pixel_x, pixel_y)} 轉換為無效的齊次座標")
        point /= point[2]
        return round(float(point[0]), 2), round(float(point[1]), 2), round(self.z_surface, 2)


def load_welding_sequence(path: str | Path) -> list[dict]:
    data = read_json(path)
    sequence = data.get("welding_sequence") if isinstance(data, dict) else data
    if not isinstance(sequence, list) or not sequence:
        raise DataContractError("焊接計畫必須包含非空 welding_sequence")
    return sequence


def assert_calibration_passed(grid_map_path: str | Path, *, allow_failed: bool) -> None:
    """Prevent a motion plan from silently using a calibration known to be inaccurate."""
    report_path = Path(grid_map_path).with_name("calibration_report.json")
    if not report_path.is_file() or allow_failed:
        return
    report = read_json(report_path)
    if not report.get("motion_passed", report.get("passed", False)):
        raise DataContractError(
            f"校正未通過：{report_path}（匹配率 {report.get('match_ratio')}、最大殘差 {report.get('max_residual_px')} px）。"
            "請重新校正；僅供離線除錯時才可使用 --allow-failed-calibration。"
        )


def generate_arm_waypoints(welding_sequence: list[dict], grid_map: list[dict], converter: ArmCoordinateConverter,
                           *, safe_height_offset: float = 20.0, workspace: tuple[float, float, float, float] | None = None) -> list[dict]:
    if safe_height_offset <= 0:
        raise ValueError("safe_height_offset 必須大於 0")
    grid_lookup = {(int(point["row"]), int(point["col"])): point for point in grid_map}
    trajectory: list[dict] = []
    seen_steps: set[int] = set()
    for item in welding_sequence:
        try:
            step = int(item["step"]); component_id = str(item["component_id"]); target = item["target_grid"]
            row, col = int(target["row"]), int(target["col"])
        except (KeyError, TypeError, ValueError) as error:
            raise DataContractError(f"無效的焊接步驟: {item!r}") from error
        if step in seen_steps or (row, col) not in grid_lookup:
            raise DataContractError(f"重複 step 或不存在的目標格點: step={step}, grid={(row, col)}")
        seen_steps.add(step)
        grid_point = grid_lookup[(row, col)]
        x, y, z_weld = converter.pixel_to_arm_3d(float(grid_point["pixel_x"]), float(grid_point["pixel_y"]))
        if workspace and not (workspace[0] <= x <= workspace[1] and workspace[2] <= y <= workspace[3]):
            raise DataContractError(f"目標 {(row, col)} 的座標 {(x, y)} 超出工作區")
        z_safe = round(z_weld + safe_height_offset, 2)
        trajectory.append({"step": step, "component_id": component_id, "target_grid": {"row": row, "col": col},
                           "actions": [
                               {"action": "MOVE_TO_APPROACH", "pos": [x, y, z_safe], "speed": "FAST"},
                               {"action": "LOWER_TO_SOLDER", "pos": [x, y, z_weld], "speed": "SLOW"},
                               {"action": "EXECUTE_SOLDERING", "dwell_time_sec": float(item.get("dwell_time_sec", 2.5))},
                               {"action": "RETRACT_TO_SAFE", "pos": [x, y, z_safe], "speed": "MEDIUM"},
                           ]})
    return trajectory


def load_robot_calibration(path: str | Path) -> tuple[ArmCoordinateConverter, tuple[float, float, float, float], float]:
    calibration = read_json(path)
    required = {"pixel_to_arm_homography", "z_board_surface_mm", "safe_height_offset_mm", "workspace_xy_mm"}
    if not isinstance(calibration, dict) or not required.issubset(calibration):
        raise DataContractError(f"無效的機械臂標定檔: {path}")
    if not calibration.get("passed", False):
        raise DataContractError(f"機械臂標定未通過: {path}；不可用於實機軌跡")
    workspace = tuple(float(value) for value in calibration["workspace_xy_mm"])
    if len(workspace) != 4 or workspace[0] >= workspace[1] or workspace[2] >= workspace[3]:
        raise DataContractError("workspace_xy_mm 必須是 [xmin, xmax, ymin, ymax]")
    safe_height = float(calibration["safe_height_offset_mm"])
    if safe_height <= 0:
        raise DataContractError("safe_height_offset_mm 必須大於 0")
    return ArmCoordinateConverter(np.asarray(calibration["pixel_to_arm_homography"], dtype=float), float(calibration["z_board_surface_mm"])), workspace, safe_height


def main() -> None:
    parser = argparse.ArgumentParser(description="產生已驗證的機械臂焊接軌跡")
    parser.add_argument("welding_plan"); parser.add_argument("grid_map")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "robot_trajectory.json"))
    parser.add_argument("--robot-calibration", default=str(DEFAULT_ROBOT_CALIBRATION), help="robot_calibration.py 產生且通過的標定檔")
    parser.add_argument("--allow-failed-calibration", action="store_true", help="僅供離線除錯；不可用於實機")
    args = parser.parse_args()
    assert_calibration_passed(args.grid_map, allow_failed=args.allow_failed_calibration)
    converter, workspace, safe_height = load_robot_calibration(args.robot_calibration)
    trajectory = generate_arm_waypoints(load_welding_sequence(args.welding_plan), validate_grid_map(read_json(args.grid_map)), converter,
                                        workspace=workspace, safe_height_offset=safe_height)
    write_json(args.output, trajectory)
    print(f"已寫入 {len(trajectory)} 個焊接步驟: {args.output}")


if __name__ == "__main__":
    main()
