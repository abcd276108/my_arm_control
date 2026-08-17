"""Use local Qwen2.5-VL as a semantic planner; motion validation remains deterministic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import DataContractError, read_json, write_json
from local_qwen_vl import LocalQwenVL
from project_settings import DEFAULT_MODEL_PATH, OUTPUT_DIR


def plan_welding_sequence(pins_data: list[dict], obstacles: list[dict], model: LocalQwenVL) -> dict:
    prompt = """Return JSON only with {"welding_sequence": [...]}. Every step must contain
step (unique positive integer), component_id, and target_grid {row, col}. Choose only pins
provided in pending_components and never choose a cell in obstacles_or_soldered."""
    result = model.generate_json([{"role": "user", "content": [{"type": "text", "text": prompt + "\n" + json.dumps({"pending_components": pins_data, "obstacles_or_soldered": obstacles}, ensure_ascii=False)}]}])
    if not isinstance(result.get("welding_sequence"), list) or not result["welding_sequence"]:
        raise DataContractError("規劃結果缺少非空 welding_sequence")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="使用本機 Qwen2.5-VL 產生焊接順序")
    parser.add_argument("pins_json"); parser.add_argument("--obstacles-json")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH)); parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "welding_plan.json"))
    args = parser.parse_args()
    pins = read_json(args.pins_json)
    obstacles = read_json(args.obstacles_json) if args.obstacles_json else []
    write_json(args.output, plan_welding_sequence(pins, obstacles, LocalQwenVL(args.model_path, device_map=args.device_map)))
    print(f"焊接計畫已寫入: {args.output}")


if __name__ == "__main__":
    main()
