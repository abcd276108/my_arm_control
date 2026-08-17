from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FIELDS = [
    "waypoint_id",
    "source_frame_id",
    "timestamp_s",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "state",
    "segment_type",
    "segment_id",
    "coordinate_frame",
    "source_x_mm",
    "source_y_mm",
    "source_z_mm",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_true(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_transform(path: Path | None) -> tuple[np.ndarray, str]:
    if path is None:
        return np.eye(4, dtype=np.float64), "camera"

    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("matrix"), dtype=np.float64)

    if matrix.shape != (4, 4):
        raise SystemExit("transform JSON 的 matrix 必須是 4 x 4。")

    frame = str(payload.get("target_frame", "target")).strip() or "target"
    return matrix, frame


def read_trajectory(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        required = {"camera_x_mm", "camera_y_mm", "camera_z_mm", "valid"}
        missing = sorted(required - fields)
        if missing:
            raise SystemExit(
                "輸入必須是 export_trajectory.py 產生的 trajectory.csv；"
                f"缺少欄位：{', '.join(missing)}"
            )
        return list(reader)


def transform_point(matrix: np.ndarray, xyz_mm: tuple[float, float, float]) -> np.ndarray:
    source = np.array([*xyz_mm, 1.0], dtype=np.float64)
    target = matrix @ source
    if abs(float(target[3])) < 1e-12:
        raise SystemExit("座標轉換結果的齊次比例為 0。")
    return target[:3] / target[3]


def build_waypoints(
    source_rows: list[dict[str, str]],
    matrix: np.ndarray,
    frame: str,
    relative: bool,
    state_filter: set[str],
    orientation: tuple[float, float, float],
    one_per_segment: bool = False,
    merge_gap_seconds: float = 0.50,
    min_segment_points: int = 4,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    for row in source_rows:
        state = str(row.get("state_filtered", "NONE")).upper()
        if not is_true(row.get("valid")) or state not in state_filter:
            continue

        xyz = (
            to_float(row.get("camera_x_mm")),
            to_float(row.get("camera_y_mm")),
            to_float(row.get("camera_z_mm")),
        )
        if any(value is None for value in xyz):
            continue

        source_xyz = tuple(float(value) for value in xyz if value is not None)
        target_xyz = transform_point(matrix, source_xyz)

        points.append({
            "source_frame_id": row.get("source_frame_id", ""),
            "timestamp_s": row.get("timestamp_s", ""),
            "target_xyz_mm": target_xyz,
            "state": state,
            "segment_type": row.get("segment_type", ""),
            "segment_id": row.get("segment_id", ""),
            "source_xyz_mm": source_xyz,
        })

    if not points:
        raise SystemExit("沒有符合條件的有效軌跡點。")

    if one_per_segment:
        groups: list[list[dict[str, Any]]] = []
        for point in points:
            timestamp = to_float(point.get("timestamp_s"))
            previous_timestamp = to_float(groups[-1][-1].get("timestamp_s")) if groups else None
            if (
                not groups
                or timestamp is None
                or previous_timestamp is None
                or timestamp - previous_timestamp > merge_gap_seconds
            ):
                groups.append([point])
            else:
                groups[-1].append(point)
        # Brief UP/NONE flicker shorter than merge_gap_seconds is treated as
        # the same physical press. Keep the temporal midpoint of each press.
        stable_groups = [group for group in groups if len(group) >= min_segment_points]
        if not stable_groups:
            raise SystemExit("沒有持續時間足夠的穩定 DOWN 落點。")
        # One stable press becomes one waypoint. Use the median of the whole
        # dwell instead of one midpoint frame so a bad depth frame cannot
        # move the weld point by several centimetres.
        robust_points: list[dict[str, Any]] = []
        for group in stable_groups:
            representative = dict(group[len(group) // 2])
            target_values = np.asarray(
                [point["target_xyz_mm"] for point in group], dtype=np.float64
            )
            source_values = np.asarray(
                [point["source_xyz_mm"] for point in group], dtype=np.float64
            )
            representative["target_xyz_mm"] = np.median(target_values, axis=0)
            representative["source_xyz_mm"] = tuple(
                float(value) for value in np.median(source_values, axis=0)
            )
            robust_points.append(representative)
        points = robust_points

    origin = points[0]["target_xyz_mm"].copy() if relative else np.zeros(3)
    roll, pitch, yaw = orientation
    output: list[dict[str, Any]] = []

    for index, point in enumerate(points):
        xyz_m = (point["target_xyz_mm"] - origin) / 1000.0
        source_xyz = point["source_xyz_mm"]
        output.append({
            "waypoint_id": index,
            "source_frame_id": point["source_frame_id"],
            "timestamp_s": point["timestamp_s"],
            "x": round(float(xyz_m[0]), 6),
            "y": round(float(xyz_m[1]), 6),
            "z": round(float(xyz_m[2]), 6),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "state": point["state"],
            "segment_type": point["segment_type"],
            "segment_id": point["segment_id"],
            "coordinate_frame": f"{frame}_relative" if relative else frame,
            "source_x_mm": round(source_xyz[0], 3),
            "source_y_mm": round(source_xyz[1], 3),
            "source_z_mm": round(source_xyz[2], 3),
        })

    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將相機 XYZ 軌跡轉成 Gazebo/MoveIt 可讀的公尺 waypoint CSV。"
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--transform",
        type=Path,
        default=None,
        help="選填：包含 target_frame 與 4x4 matrix 的 JSON。",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="輸出目標座標系的絕對位置；預設以第一個有效點為原點。",
    )
    parser.add_argument(
        "--states",
        default="UP,DOWN",
        help="要保留的狀態，逗號分隔。",
    )
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--one-per-segment",
        action="store_true",
        help="每個連續狀態區段只輸出時間中點；適合只保留 DOWN 落點。",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=0.50,
        help="DOWN 點間隔小於此秒數時視為同一次落下，用於抑制狀態跳動。",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=4,
        help="一個 DOWN 落點至少需持續的有效畫面數。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_path = resolve_path(args.trajectory)
    output_path = resolve_path(args.output)
    transform_path = resolve_path(args.transform) if args.transform else None

    source_rows = read_trajectory(trajectory_path)
    matrix, frame = load_transform(transform_path)
    states = {item.strip().upper() for item in args.states.split(",") if item.strip()}
    waypoints = build_waypoints(
        source_rows=source_rows,
        matrix=matrix,
        frame=frame,
        relative=not args.absolute,
        state_filter=states,
        orientation=(args.roll, args.pitch, args.yaw),
        one_per_segment=args.one_per_segment,
        merge_gap_seconds=args.merge_gap_seconds,
        min_segment_points=args.min_segment_points,
    )
    write_csv(output_path, waypoints)

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "source": str(trajectory_path),
        "output": str(output_path),
        "waypoint_count": len(waypoints),
        "press_point_count": len(waypoints) if args.one_per_segment else None,
        "coordinate_frame": waypoints[0]["coordinate_frame"],
        "unit": "meter",
        "relative": not args.absolute,
        "transform_applied": transform_path is not None,
        "robot_execution_ready": transform_path is not None,
        "warning": (
            "Camera-frame preview only. Do not send to the physical robot."
            if transform_path is None
            else "Verify the transform and TCP in simulation before physical execution."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Gazebo waypoint CSV: {output_path}")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Coordinate frame: {waypoints[0]['coordinate_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
