"""Simulation-only trajectory executor with fail-closed local Qwen2.5-VL AOI."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from contracts import DataContractError, read_json, write_json
from local_qwen_vl import LocalQwenVL
from project_settings import DEFAULT_MODEL_PATH, OUTPUT_DIR


class SimulatedArmController:
    """Deliberately non-hardware controller. Replace through a reviewed adapter only."""
    def move_to_pose(self, x: float, y: float, z: float, speed: str) -> None:
        if speed not in {"FAST", "MEDIUM", "SLOW"}:
            raise DataContractError(f"未知速度: {speed}")
        print(f"[SIM] move {speed}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        time.sleep(0.05)

    def execute_soldering_action(self, dwell_time: float) -> None:
        if not 0 < dwell_time <= 30:
            raise DataContractError(f"不安全的焊接停留時間: {dwell_time}")
        print(f"[SIM] solder for {dwell_time:.2f}s")
        time.sleep(0.05)


def unknown_aoi(reason: str) -> dict:
    return {"inspection_result": "UNKNOWN", "defect_type": "Unknown", "confidence": 0.0,
            "description": reason, "remedy_suggestion": "停止自動流程，人工複檢焊點與相機／模型。"}


def inspect_solder_quality_with_vlm(image_path: str | Path, model: LocalQwenVL) -> dict:
    image = Path(image_path)
    if not image.is_file():
        return unknown_aoi(f"找不到 AOI 影像: {image}")
    prompt = """Inspect this PCB solder joint. Return JSON only with inspection_result (PASS, FAIL, or UNKNOWN),
defect_type, confidence from 0 to 1, description, and remedy_suggestion. If image quality is insufficient, use UNKNOWN."""
    try:
        result = model.generate_json([{"role": "user", "content": [{"type": "text", "text": prompt}, model.image_content(image)]}])
        if result.get("inspection_result") not in {"PASS", "FAIL", "UNKNOWN"}:
            return unknown_aoi("Qwen2.5-VL 回覆缺少有效 inspection_result")
        return result
    except (OSError, RuntimeError, DataContractError) as error:
        return unknown_aoi(f"本機 AOI 無法完成: {error}")


def execute_trajectory(trajectory: list[dict], *, aoi_dir: str | Path | None = None, model: LocalQwenVL | None = None) -> list[dict]:
    arm, reports = SimulatedArmController(), []
    for step in trajectory:
        step_id = step.get("step")
        for action in step.get("actions", []):
            if "pos" in action:
                arm.move_to_pose(*action["pos"], speed=action.get("speed", "MEDIUM"))
            elif action.get("action") == "EXECUTE_SOLDERING":
                arm.execute_soldering_action(float(action.get("dwell_time_sec", 0)))
        result = inspect_solder_quality_with_vlm(Path(aoi_dir) / f"solder_inspection_step_{step_id}.jpg", model) if aoi_dir and model else unknown_aoi("未提供本機 AOI 模型與影像目錄；未宣告焊接合格。")
        reports.append({"step": step_id, "component_id": step.get("component_id"), "aoi_result": result})
        if result["inspection_result"] != "PASS":
            print(f"[STOP] Step {step_id} AOI={result['inspection_result']}: {result['description']}")
            break
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="僅模擬執行軌跡；本機 AOI 非 PASS 即停止")
    parser.add_argument("trajectory"); parser.add_argument("--report", default=str(OUTPUT_DIR / "final_aoi_report.json"))
    parser.add_argument("--aoi-dir"); parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH)); parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    # The default model path is used only when AOI is explicitly requested.
    model = LocalQwenVL(args.model_path, device_map=args.device_map) if args.aoi_dir else None
    write_json(args.report, execute_trajectory(read_json(args.trajectory), aoi_dir=args.aoi_dir, model=model))
    print(f"模擬 AOI 報告已寫入: {args.report}")


if __name__ == "__main__":
    main()
