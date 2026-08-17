from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UP_COLOR = (122, 72, 31)       # dark blue, BGR
DOWN_COLOR = (55, 61, 153)     # brick red, BGR
TEXT_COLOR = (48, 48, 48)
MUTED_COLOR = (125, 125, 125)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def stable_state(history: deque[str], minimum_votes: int) -> str:
    valid = [state for state in history if state in {"UP", "DOWN"}]
    if len(valid) < minimum_votes:
        return "NONE"
    state, count = Counter(valid).most_common(1)[0]
    return state if count >= minimum_votes else "NONE"


def put(canvas: np.ndarray, text: str, x: int, y: int, scale: float = 0.62,
        color: tuple[int, int, int] = TEXT_COLOR, thickness: int = 1) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a clean JetArm inference video")
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--state-window", type=int, default=5)
    parser.add_argument("--minimum-votes", type=int, default=3)
    parser.add_argument("--point-window", type=int, default=3)
    parser.add_argument("--panel-width", type=int, default=310)
    args = parser.parse_args()

    fusion_path, output_path = resolve(args.fusion), resolve(args.output)
    with fusion_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("fusion CSV has no rows")

    first = cv2.imread(str(resolve(rows[0]["rgb_path"])))
    if first is None:
        raise SystemExit(f"Cannot read RGB frame: {rows[0].get('rgb_path')}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (width + args.panel_width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create video: {output_path}")

    states: deque[str] = deque(maxlen=max(1, args.state_window))
    points: deque[tuple[float, float]] = deque(maxlen=max(1, args.point_window))
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        frame = cv2.imread(str(resolve(row["rgb_path"])))
        if frame is None:
            continue
        raw_state = str(row.get("raw_state", "NONE")).upper()
        confidence = number(row.get("raw_confidence")) or 0.0
        raw_x, raw_y = number(row.get("x_px_raw")), number(row.get("y_px_raw"))
        accepted = (
            raw_state in {"UP", "DOWN"} and confidence >= args.confidence
            and raw_x is not None and raw_y is not None
        )
        states.append(raw_state if accepted else "NONE")
        shown_state = stable_state(states, args.minimum_votes)
        if accepted and shown_state == raw_state:
            points.append((raw_x, raw_y))
        elif shown_state == "NONE":
            points.clear()

        canvas = np.full((height, width + args.panel_width, 3), 248, np.uint8)
        canvas[:, :width] = frame
        cv2.line(canvas, (width, 0), (width, height), (220, 220, 220), 1)

        shown_point: tuple[int, int] | None = None
        if shown_state in {"UP", "DOWN"} and points:
            point_array = np.asarray(points, dtype=float)
            px, py = np.median(point_array, axis=0)
            shown_point = (int(round(px)), int(round(py)))
            color = UP_COLOR if shown_state == "UP" else DOWN_COLOR
            cv2.circle(canvas, shown_point, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, shown_point, 4, color, -1, cv2.LINE_AA)

        left = width + 28
        put(canvas, "JETARM TIP TRACKING", left, 42, 0.65, TEXT_COLOR, 2)
        put(canvas, f"Frame  {index} / {total}", left, 76, 0.48, MUTED_COLOR)
        put(canvas, "STATE", left, 126, 0.46, MUTED_COLOR, 2)
        if shown_state in {"UP", "DOWN"}:
            state_color = UP_COLOR if shown_state == "UP" else DOWN_COLOR
            put(canvas, shown_state, left, 168, 0.90, state_color, 2)
            put(canvas, f"Confidence  {confidence:.3f}", left, 219, 0.55)
            if shown_point is not None:
                put(canvas, "TIP POSITION", left, 273, 0.46, MUTED_COLOR, 2)
                put(canvas, f"x = {shown_point[0]} px", left, 310, 0.53)
                put(canvas, f"y = {shown_point[1]} px", left, 342, 0.53)
            put(canvas, "Stable detection", left, height - 38, 0.50, state_color, 2)
        else:
            put(canvas, "ANALYZING", left, 171, 0.86, MUTED_COLOR, 2)
            put(canvas, "Low-confidence frames hidden", left, 219, 0.46, MUTED_COLOR)
        writer.write(canvas)

    writer.release()
    print(f"Clean inference video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
