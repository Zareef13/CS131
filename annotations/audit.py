"""Visual audit rendering for Gemini + SAM annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def panels(record: dict[str, Any], root: Path) -> list[np.ndarray]:
    source_bgr = cv2.imread(str(root / record["source_frame"])); source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    boxes = source.copy()
    individual = source.copy()
    colors = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 64)]
    for index, obstacle in enumerate(record["obstacles"]):
        color = colors[index % len(colors)]; x0, y0, x1, y1 = obstacle["box_pixels"]
        cv2.rectangle(boxes, (x0, y0), (x1, y1), color, 2)
        cv2.putText(boxes, obstacle["label"], (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        candidate_path = obstacle.get("selected_mask_file")
        if candidate_path:
            mask = cv2.imread(str(root / candidate_path), cv2.IMREAD_GRAYSCALE) > 0
            tint = np.zeros_like(individual); tint[:] = color
            individual[mask] = (0.55 * individual[mask] + 0.45 * tint[mask]).astype(np.uint8)
    merged = cv2.imread(str(root / record["mask"]), cv2.IMREAD_GRAYSCALE)
    overlay = source.copy(); active = merged > 0
    overlay[active] = (0.55 * overlay[active] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    return [source, boxes, individual, merged, overlay]


def save_audits(records: list[dict[str, Any]], root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = ["Source", "Gemini boxes / labels", "Individual SAM masks", "Merged binary mask", "Merged overlay"]
    all_panels = []
    for record in records:
        images = panels(record, root); all_panels.append(images)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for axis, image, title in zip(axes, images, titles):
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None); axis.set_title(title); axis.axis("off")
        fig.tight_layout(); fig.savefig(output_dir / f"{record['domain']}_frame_{record['frame_index']:04d}.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(len(records), 5, figsize=(18, max(4, 3 * len(records))), squeeze=False)
    for row, (record, images) in enumerate(zip(records, all_panels)):
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray" if image.ndim == 2 else None); axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title(titles[col])
        axes[row, 0].set_ylabel(f"{record['domain']} {record['frame_index']}")
    fig.tight_layout(); fig.savefig(output_dir / "pilot_contact_sheet.png", dpi=120); plt.close(fig)
