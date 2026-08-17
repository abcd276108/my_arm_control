"""Shared file formats and validation for the PCB soldering proof of concept."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataContractError(ValueError):
    """Raised when an intermediate pipeline file has an unexpected shape."""


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def validate_grid_map(data: Any) -> list[dict[str, float | int]]:
    if not isinstance(data, list) or not data:
        raise DataContractError("grid_map 必須是非空的格點陣列")

    required = {"row", "col", "pixel_x", "pixel_y"}
    seen: set[tuple[int, int]] = set()
    validated: list[dict[str, float | int]] = []
    for point in data:
        if not isinstance(point, dict) or not required.issubset(point):
            raise DataContractError(f"無效的 grid point: {point!r}")
        row, col = int(point["row"]), int(point["col"])
        if row < 0 or col < 0 or (row, col) in seen:
            raise DataContractError(f"重複或無效的格點: {(row, col)}")
        seen.add((row, col))
        validated.append({"row": row, "col": col,
                          "pixel_x": float(point["pixel_x"]),
                          "pixel_y": float(point["pixel_y"])})
    return validated


def validate_bbox_pixel(bbox: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise DataContractError("bbox_pixel 必須是 [x_min, y_min, x_max, y_max]")
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise DataContractError(f"bbox_pixel 超出影像範圍: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2
