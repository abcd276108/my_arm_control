"""Direct, offline Qwen2.5-VL inference via Transformers (no HTTP/API client)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import DataContractError


class LocalQwenVL:
    """Lazy-loaded local Qwen2.5-VL model shared by detector, planner, and AOI."""

    def __init__(self, model_path: str | Path, *, device_map: str = "auto") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"找不到本機 Qwen2.5-VL 模型資料夾: {self.model_path}")
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("缺少本地 Qwen2.5-VL 依賴，請執行 pip install -r requirements.txt") from error
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path), torch_dtype="auto", device_map=device_map, local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.input_device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def image_content(path: str | Path) -> dict[str, str]:
        image = Path(path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"找不到影像: {image}")
        return {"type": "image", "image": image.as_uri()}

    def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int = 256) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.input_device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_json(self, messages: list[dict[str, Any]], *, max_new_tokens: int = 256) -> dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens)
        cleaned = _extract_json_object(text)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise DataContractError(f"Qwen2.5-VL 沒有輸出有效 JSON: {text[:300]}") from error
        if not isinstance(result, dict):
            raise DataContractError("Qwen2.5-VL JSON 根節點必須是物件")
        return result


def _extract_json_object(text: str) -> str:
    """Extract the first balanced JSON object when a local model adds prose/fences."""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text.strip()
