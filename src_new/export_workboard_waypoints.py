from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from workboard_calibration import pixel_to_board, pixel_to_robot, validate_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIELDS = [
    "waypoint_id", "source_frame_id", "timestamp_s", "state",
    "pixel_x", "pixel_y", "point_source", "confidence_median",
    "dwell_frames", "dwell_seconds", "board_x_mm", "board_y_mm",
    "robot_x_mm", "robot_y_mm", "robot_z_mm",
    "robot_x_m", "robot_y_m", "robot_z_m", "coordinate_frame",
    "approach_x_m", "approach_y_m", "approach_z_m",
    "height_mode", "robot_execution_ready",
]
GAZEBO_FIELDS = [
    "waypoint_id", "x_mm", "y_mm", "state", "dwell_seconds",
    "coordinate_frame",
]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def group_down_rows(
    rows: list[dict[str, str]], gap_seconds: float, split_up_frames: int,
    confidence: float,
) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    raw_up_run = 0
    for row in rows:
        raw_state = str(row.get("raw_state", "NONE")).upper()
        raw_confidence = number(row.get("raw_confidence")) or 0.0
        if raw_state == "UP" and raw_confidence >= confidence:
            raw_up_run += 1
            if raw_up_run >= split_up_frames and current:
                groups.append(current)
                current = []
            continue
        raw_up_run = 0
        final_down = str(row.get("final_state", "NONE")).upper() == "DOWN"
        raw_down = raw_state == "DOWN" and raw_confidence >= confidence
        # A short press may not survive the temporal state filter. Retain a
        # confident raw candidate; the segment-length check rejects flicker.
        if not (final_down or raw_down):
            continue
        timestamp = number(row.get("timestamp_s"))
        previous = number(current[-1].get("timestamp_s")) if current else None
        if current and (
            timestamp is None or previous is None or timestamp - previous > gap_seconds
        ):
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def segment_point(group: list[dict[str, str]], confidence: float) -> tuple[float, float, str, float]:
    raw: list[tuple[float, float, float]] = []
    filtered: list[tuple[float, float, float]] = []
    for row in group:
        conf = number(row.get("raw_confidence")) or 0.0
        raw_x, raw_y = number(row.get("x_px_raw")), number(row.get("y_px_raw"))
        filtered_x = number(row.get("x_px_filtered"))
        filtered_y = number(row.get("y_px_filtered"))
        if (
            str(row.get("raw_state", "NONE")).upper() == "DOWN"
            and raw_x is not None and raw_y is not None and conf >= confidence
        ):
            raw.append((raw_x, raw_y, conf))
        if filtered_x is not None and filtered_y is not None:
            filtered.append((filtered_x, filtered_y, conf))
    selected, source = (raw, "raw_yolo_segment_median") if raw else (filtered, "filtered_fallback_median")
    if not selected:
        raise ValueError("DOWN segment has no usable tip point")
    values = np.asarray(selected, dtype=float)
    return (
        float(np.median(values[:, 0])),
        float(np.median(values[:, 1])),
        source,
        float(np.median(values[:, 2])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export stable DOWN points in JetArm workboard coordinates")
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--calibration", default="config/workboard_calibration.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.50)
    parser.add_argument("--min-segment-frames", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--split-up-frames", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    fusion_path = resolve(args.fusion)
    calibration_path = resolve(args.calibration)
    output_path = resolve(args.output)
    with fusion_path.open("r", newline="", encoding="utf-8-sig") as file:
        source_rows = list(csv.DictReader(file))
    config = validate_config(json.loads(calibration_path.read_text(encoding="utf-8")))
    if config.get("calibration_status") != "ready":
        raise SystemExit("Workboard calibration is not complete (image corners + three JetArm points required)")
    heights = config.get("heights_mm", {})
    down_height_mm = float(heights.get("down_dry_run_mm", 15.0))
    height_mode = f"DOWN_dry_run_{down_height_mm:g}mm"

    groups = [
        group for group in group_down_rows(
            source_rows, args.merge_gap_seconds, args.split_up_frames,
            args.confidence,
        )
        if len(group) >= args.min_segment_frames
    ]
    if not groups:
        raise SystemExit("No stable DOWN segment found")

    output: list[dict[str, Any]] = []
    skipped = 0
    for group in groups:
        try:
            pixel_x, pixel_y, point_source, confidence = segment_point(group, args.confidence)
        except ValueError:
            skipped += 1
            continue
        board = pixel_to_board(config, (pixel_x, pixel_y))
        robot = pixel_to_robot(config, (pixel_x, pixel_y), "DOWN")
        approach = pixel_to_robot(config, (pixel_x, pixel_y), "UP")
        midpoint = group[len(group) // 2]
        first_time = number(group[0].get("timestamp_s"))
        last_time = number(group[-1].get("timestamp_s"))
        dwell = (
            max(0.0, last_time - first_time + 1.0 / args.fps)
            if first_time is not None and last_time is not None
            else len(group) / args.fps
        )
        output.append({
            "waypoint_id": len(output),
            "source_frame_id": midpoint.get("frame_id", ""),
            "timestamp_s": midpoint.get("timestamp_s", ""),
            "state": "DOWN",
            "pixel_x": round(pixel_x, 3), "pixel_y": round(pixel_y, 3),
            "point_source": point_source,
            "confidence_median": round(confidence, 4),
            "dwell_frames": len(group), "dwell_seconds": round(dwell, 3),
            "board_x_mm": round(float(board[0]), 3),
            "board_y_mm": round(float(board[1]), 3),
            "robot_x_mm": round(float(robot[0]), 3),
            "robot_y_mm": round(float(robot[1]), 3),
            "robot_z_mm": round(float(robot[2]), 3),
            "robot_x_m": round(float(robot[0]) / 1000.0, 6),
            "robot_y_m": round(float(robot[1]) / 1000.0, 6),
            "robot_z_m": round(float(robot[2]) / 1000.0, 6),
            "coordinate_frame": "jetarm_base_workboard_calibrated",
            "approach_x_m": round(float(approach[0]) / 1000.0, 6),
            "approach_y_m": round(float(approach[1]) / 1000.0, 6),
            "approach_z_m": round(float(approach[2]) / 1000.0, 6),
            "height_mode": height_mode,
            "robot_execution_ready": False,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    gazebo_path = output_path.with_name("gazebo_down_points.csv")
    with gazebo_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=GAZEBO_FIELDS)
        writer.writeheader()
        for row in output:
            writer.writerow({
                "waypoint_id": row["waypoint_id"],
                "x_mm": row["board_x_mm"], "y_mm": row["board_y_mm"],
                "state": "DOWN",
                "dwell_seconds": row["dwell_seconds"],
                "coordinate_frame": "workboard_left_bottom",
            })
    summary = {
        "source": str(fusion_path), "calibration": str(calibration_path),
        "waypoint_count": len(output), "stable_down_segments": len(groups),
        "skipped_segments": skipped, "height_mode": height_mode,
        "robot_execution_ready": False,
        "warning": "Coordinates are for validation/dry-run only. Run IK and workspace checks before motion.",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Robot workboard waypoint CSV: {output_path}")
    print(f"Gazebo DOWN-only CSV: {gazebo_path}")
    print(f"Waypoints: {len(output)}")
    print(f"Height mode: DOWN dry-run {down_height_mm:g} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
