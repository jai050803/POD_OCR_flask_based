"""Two-Tier OCR Recognition Pipeline.

Tier 1: PaddleOCR (DBNet detection + CRNN recognition)
Tier 2: TrOCR fallback for low-confidence blocks (microsoft/trocr-base-handwritten)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import PIL.Image
from PIL import Image

# Ensure PaddlePaddle oneDNN flags are set for CPU inference stability
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_EAGER_INIT"] = "0"

# Make PaddleX initialization idempotent to prevent 'PDX has already been initialized' crash
try:
    import paddlex.repo_manager.core as pdx_core

    _orig_pdx_initialize = pdx_core.initialize

    def _safe_pdx_initialize(*args, **kwargs):
        if pdx_core._GlobalContext.is_initialized():
            return
        return _orig_pdx_initialize(*args, **kwargs)

    pdx_core.initialize = _safe_pdx_initialize
except Exception:
    pass


class OCREngine:
    """Singleton pattern manager for PaddleOCR and TrOCR engines."""

    _paddle_instance = None
    _trocr_processor = None
    _trocr_model = None

    @classmethod
    def get_paddle(cls):
        """Retrieve or initialize the PaddleOCR singleton instance."""
        if cls._paddle_instance is None:
            from paddleocr import PaddleOCR

            try:
                cls._paddle_instance = PaddleOCR(
                    use_angle_cls=True,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    lang="en",
                    show_log=False,
                )
            except (ValueError, TypeError):
                cls._paddle_instance = PaddleOCR(
                    use_angle_cls=True,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    lang="en",
                )
        return cls._paddle_instance

    @classmethod
    def get_trocr(cls) -> Tuple[Optional[Any], Optional[Any]]:
        """Retrieve or lazily initialize the TrOCR processor and model."""
        if cls._trocr_processor is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                cls._trocr_processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                cls._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
            except Exception:
                cls._trocr_processor = None
                cls._trocr_model = None
        return cls._trocr_processor, cls._trocr_model


def run_trocr_batch(pil_crops: List[Image.Image]) -> List[str]:
    """Run TrOCR on a batch of text line crops using torch.inference_mode.

    Args:
        pil_crops: List of PIL Image crops containing handwritten/low-confidence text.

    Returns:
        List of recognized text strings.
    """
    if not pil_crops:
        return []

    processor, model = OCREngine.get_trocr()
    if processor is None or model is None:
        return ["" for _ in pil_crops]

    try:
        import torch

        rgb_crops = [c.convert("RGB") for c in pil_crops]
        inputs = processor(images=rgb_crops, return_tensors="pt")
        with torch.inference_mode():
            generated_ids = model.generate(inputs.pixel_values, max_new_tokens=18)
        recognized_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in recognized_texts]
    except Exception:
        return ["" for _ in pil_crops]


def run_trocr_on_crop(pil_crop: Image.Image) -> str:
    """Run TrOCR on a single text line crop."""
    results = run_trocr_batch([pil_crop])
    return results[0] if results else ""


def _parse_paddle_raw(ocr_output: Any) -> List[Dict[str, Any]]:
    """Parse raw PaddleOCR output into structured block dictionaries."""
    blocks: List[Dict[str, Any]] = []

    if not ocr_output:
        return blocks

    # Handle PaddleX output format (list of dicts containing dt_polys, rec_texts, rec_scores)
    if isinstance(ocr_output, list) and len(ocr_output) > 0 and isinstance(ocr_output[0], dict):
        item = ocr_output[0]
        polys = item.get("dt_polys", [])
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            pts = np.array(poly)
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))
            clean_text = str(text).strip()
            if clean_text:
                blocks.append({
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "text": clean_text,
                    "confidence": float(score),
                    "method": "PADDLE_CRNN",
                })
        return blocks

    # Handle classic PaddleOCR format: [[ [box, (text, score)], ... ]]
    lines = ocr_output[0] if (
        isinstance(ocr_output, list)
        and len(ocr_output) > 0
        and isinstance(ocr_output[0], list)
    ) else ocr_output

    if isinstance(lines, list):
        for line in lines:
            if not line or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text, score = rec[0], rec[1]
            else:
                text, score = str(rec), 1.0

            pts = np.array(box)
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))
            clean_text = str(text).strip()
            if clean_text:
                blocks.append({
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "text": clean_text,
                    "confidence": float(score),
                    "method": "PADDLE_CRNN",
                })

    return blocks


def extract_text(pil_image: Image.Image) -> Dict[str, Any]:
    """Run two-tier OCR recognition on a document image.

    Tier 1: PaddleOCR extracts text blocks with DBNet detection + CRNN recognition.
    Tier 2: TrOCR fallback for blocks where confidence < 0.75.

    Returns:
        dict:
            blocks: List[dict] with bbox, text, confidence, and method
            raw_text: str (blocks joined with newlines)
            mean_confidence: float
            total_blocks: int
            trocr_used: bool
    """
    img_rgb = pil_image.convert("RGB")
    w_img, h_img = img_rgb.size
    img_np = np.array(img_rgb)

    paddle_engine = OCREngine.get_paddle()

    # Run PaddleOCR inference
    if hasattr(paddle_engine, "predict"):
        raw_output = list(paddle_engine.predict(img_np))
    else:
        raw_output = paddle_engine.ocr(img_np, cls=True)

    blocks = _parse_paddle_raw(raw_output)
    trocr_used = False

    # Tier 2: TrOCR fallback for low-confidence blocks (batched for speed)
    candidates = []
    for idx, block in enumerate(blocks):
        clean_t = block["text"].strip()
        if block["confidence"] < 0.72 and len(clean_t) > 1:
            candidates.append((idx, block))

    # Cap to max 6 lowest-confidence candidates to prevent CPU inference lag
    candidates.sort(key=lambda x: x[1]["confidence"])
    candidates = candidates[:6]

    if candidates:
        crops = []
        for _, block in candidates:
            xs = [pt[0] for pt in block["bbox"]]
            ys = [pt[1] for pt in block["bbox"]]
            crop_x1 = max(0, min(xs) - 4)
            crop_y1 = max(0, min(ys) - 4)
            crop_x2 = min(w_img, max(xs) + 4)
            crop_y2 = min(h_img, max(ys) + 4)
            crops.append(img_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2)))

        trocr_results = run_trocr_batch(crops)

        for (b_idx, block), trocr_text in zip(candidates, trocr_results):
            if trocr_text and any(c.isalnum() for c in trocr_text):
                block["text"] = trocr_text
                block["method"] = "TROCR_FALLBACK"
                trocr_used = True

    # Sort all blocks top-to-bottom, left-to-right by bbox y1 coordinate
    blocks.sort(key=lambda b: (b["bbox"][0][1], b["bbox"][0][0]))

    # Build raw_text and calculate mean confidence
    raw_text = "\n".join(b["text"] for b in blocks)
    mean_confidence = (
        float(np.mean([b["confidence"] for b in blocks])) if blocks else 0.0
    )

    return {
        "blocks": blocks,
        "raw_text": raw_text,
        "mean_confidence": round(mean_confidence, 4),
        "total_blocks": len(blocks),
        "trocr_used": trocr_used,
    }
