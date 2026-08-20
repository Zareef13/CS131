"""Local SAM 2 tiny adapter using box and positive-point prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


MODEL = "facebook/sam2.1-hiera-tiny"


@dataclass
class SamResult:
    masks: list[np.ndarray]
    scores: list[float]
    selected_index: int
    latency_seconds: float
    cached: bool


class Sam2Segmenter:
    def __init__(self, model_name: str = MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch; self.device = device; self.model_name = model_name
        self.processor = Sam2Processor.from_pretrained(model_name, local_files_only=True)
        self.model = Sam2Model.from_pretrained(model_name, local_files_only=True).to(device).eval()

    def segment(self, image_rgb: np.ndarray, box: list[int], point: list[int], cache_path: Path) -> SamResult:
        if cache_path.exists():
            data = np.load(cache_path)
            masks = [mask.astype(bool) for mask in data["masks"]]
            scores = [float(x) for x in data["scores"]]
            return SamResult(masks, scores, int(data["selected_index"]), float(data["latency_seconds"]), True)
        inputs = self.processor(images=Image.fromarray(image_rgb), input_points=[[[point]]], input_labels=[[[1]]], input_boxes=[[box]], return_tensors="pt").to(self.device)
        start = perf_counter()
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        latency = perf_counter() - start
        processed = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        candidates = processed[0].numpy() > 0
        raw_scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        scores = [float(x) for x in raw_scores[:len(candidates)]]
        selected = int(np.argmax(scores))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, masks=candidates.astype(np.uint8), scores=np.asarray(scores), selected_index=selected, latency_seconds=latency)
        return SamResult([x for x in candidates], scores, selected, latency, False)
