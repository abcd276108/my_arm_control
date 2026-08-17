from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FX, FY, CX, CY = 456.0071, 456.0071, 325.24936, 240.54808

FIELDS = [
    "source_frame_id", "sample_id", "timestamp_s", "image_name", "source", "depth_path",
    "state_raw", "state_filtered", "confidence", "x_px_raw", "y_px_raw",
    "x_px_filtered", "y_px_filtered", "image_width", "image_height",
    "depth_temporal_median_mm", "depth_mapping", "depth_source",
    "camera_x_mm", "camera_y_mm", "camera_z_mm", "step_distance_mm",
    "interpolated", "outlier", "valid", "invalid_reason", "quality",
    "segment_type", "segment_id", "coordinate_frame", "robot_execution_ready",
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


def centered_mean(values: list[float | None], index: int, window: int) -> float | None:
    radius = window // 2
    selected = [v for v in values[max(0, index-radius):min(len(values), index+radius+1)] if v is not None]
    return float(np.mean(selected)) if selected else None


def main() -> int:
    parser = argparse.ArgumentParser(description="將 depth fusion CSV 轉成平滑相機 XYZ 軌跡。")
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--state-source", choices=("final", "raw"), default="final",
        help="Use fused final_state or direct YOLO raw_state.",
    )
    parser.add_argument(
        "--point-source", choices=("filtered", "raw"), default="filtered",
        help="Use the filtered tip point or the direct YOLO tip point.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.smooth_window <= 0 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window 必須是正奇數。")

    fusion_path, output_path = resolve(args.fusion), resolve(args.output)
    with fusion_path.open("r", newline="", encoding="utf-8-sig") as file:
        source_rows = list(csv.DictReader(file))

    raw_xyz: list[tuple[float, float, float] | None] = []
    for row in source_rows:
        point_suffix = "raw" if args.point_source == "raw" else "filtered"
        x = number(row.get(f"x_px_{point_suffix}"))
        y = number(row.get(f"y_px_{point_suffix}"))
        z = number(row.get("tip_depth_mm"))
        if x is None or y is None or z is None:
            raw_xyz.append(None)
        else:
            raw_xyz.append(((x - CX) * z / FX, (y - CY) * z / FY, z))

    xs = [p[0] if p else None for p in raw_xyz]
    ys = [p[1] if p else None for p in raw_xyz]
    zs = [p[2] if p else None for p in raw_xyz]
    rows: list[dict[str, Any]] = []
    previous: tuple[float, float, float] | None = None
    segment_id, previous_type = 0, "none"

    for index, source in enumerate(source_rows):
        state_field = "raw_state" if args.state_source == "raw" else "final_state"
        state = str(source.get(state_field, "NONE")).upper()
        x, y, z = (centered_mean(values, index, args.smooth_window) for values in (xs, ys, zs))
        valid = state in {"UP", "DOWN"} and x is not None and y is not None and z is not None
        segment_type = "welding" if valid and state == "DOWN" else "travel" if valid else "none"
        if segment_type != "none" and segment_type != previous_type:
            segment_id += 1
        step = None
        if valid and previous is not None:
            step = math.dist(previous, (x, y, z))
        if valid:
            previous = (x, y, z)
        rows.append({
            "source_frame_id": source.get("frame_id", index), "sample_id": index,
            "timestamp_s": number(source.get("timestamp_s")) if number(source.get("timestamp_s")) is not None else round(index / args.fps, 6), "image_name": source.get("image_name", ""),
            "source": source.get("rgb_path", ""), "depth_path": source.get("depth_path", ""),
            "state_raw": source.get("raw_state", "NONE"), "state_filtered": state,
            "confidence": source.get("raw_confidence", 0), "x_px_raw": source.get("x_px_raw", ""),
            "y_px_raw": source.get("y_px_raw", ""), "x_px_filtered": source.get("x_px_filtered", ""),
            "y_px_filtered": source.get("y_px_filtered", ""), "image_width": 640, "image_height": 480,
            "depth_temporal_median_mm": round(z, 3) if z is not None else None,
            "depth_mapping": source.get("depth_mapping", ""), "depth_source": "fusion_tip_depth_smoothed",
            "camera_x_mm": round(x, 3) if x is not None else None,
            "camera_y_mm": round(y, 3) if y is not None else None,
            "camera_z_mm": round(z, 3) if z is not None else None,
            "step_distance_mm": round(step, 3) if step is not None else None,
            "interpolated": False, "outlier": False, "valid": valid,
            "invalid_reason": "" if valid else "missing_fused_depth_or_state",
            "quality": "A" if valid and source.get("depth_reliable") == "True" else "B" if valid else "D",
            "segment_type": segment_type, "segment_id": segment_id if segment_type != "none" else None,
            "coordinate_frame": "camera_fusion_smoothed", "robot_execution_ready": False,
        })
        previous_type = segment_type

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    step_values = [float(row["step_distance_mm"]) for row in rows if row["step_distance_mm"] is not None]
    summary = {
        "source": str(fusion_path), "output": str(output_path), "total_samples": len(rows),
        "valid_trajectory_samples": sum(bool(row["valid"]) for row in rows),
        "smooth_window": args.smooth_window,
        "state_source": args.state_source, "point_source": args.point_source,
        "coordinate_frame": "camera_fusion_smoothed",
        "median_step_mm": float(np.median(step_values)) if step_values else None,
        "maximum_step_mm": max(step_values) if step_values else None,
        "steps_over_10mm": sum(value > 10.0 for value in step_values),
        "robot_execution_ready": False,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fusion XYZ CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
