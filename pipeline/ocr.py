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


def run_trocr_on_crop(pil_crop: Image.Image) -> str:
    """Run TrOCR on a single text line crop.

    Args:
        pil_crop: A PIL Image crop containing handwritten or low-confidence text.

    Returns:
        Recognized string, or empty string if TrOCR is unavailable or recognition fails.
    """
    processor, model = OCREngine.get_trocr()
    if processor is None or model is None:
        return ""

    try:
        rgb_crop = pil_crop.convert("RGB")
        pixel_values = processor(images=rgb_crop, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        recognized_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return recognized_text.strip()
    except Exception:
        return ""


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

    # Tier 2: TrOCR fallback for low-confidence blocks
    for block in blocks:
        if block["confidence"] < 0.75:
            xs = [pt[0] for pt in block["bbox"]]
            ys = [pt[1] for pt in block["bbox"]]
            crop_x1 = max(0, min(xs) - 4)
            crop_y1 = max(0, min(ys) - 4)
            crop_x2 = min(w_img, max(xs) + 4)
            crop_y2 = min(h_img, max(ys) + 4)

            crop = img_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            trocr_result = run_trocr_on_crop(crop)

            if trocr_result:
                block["text"] = trocr_result
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


def ocr_stamp_crops(stamp_crops: List[Image.Image]) -> List[str]:
    """Run PaddleOCR on each stamp crop using fast batch inference.

    Args:
        stamp_crops: List of cropped PIL Images representing stamp regions.

    Returns:
        List of recognized text strings, one per stamp crop.
    """
    if not stamp_crops:
        return []

    paddle_engine = OCREngine.get_paddle()
    extracted_strings: List[str] = ["" for _ in stamp_crops]

    # Collect valid crops
    valid_crops: List[np.ndarray] = []
    valid_indices: List[int] = []
    for idx, crop in enumerate(stamp_crops):
        if crop.width >= 10 and crop.height >= 10:
            valid_crops.append(np.array(crop.convert("RGB")))
            valid_indices.append(idx)

    if not valid_crops:
        return extracted_strings

    try:
        if hasattr(paddle_engine, "predict"):
            raw_outputs = list(paddle_engine.predict(valid_crops))
            for orig_idx, raw_out in zip(valid_indices, raw_outputs):
                blocks = _parse_paddle_raw([raw_out])
                crop_text = " ".join(b["text"] for b in blocks if b.get("text")).strip()
                extracted_strings[orig_idx] = crop_text
        else:
            for orig_idx, crop_np in zip(valid_indices, valid_crops):
                raw_out = paddle_engine.ocr(crop_np, cls=True)
                blocks = _parse_paddle_raw(raw_out)
                crop_text = " ".join(b["text"] for b in blocks if b.get("text")).strip()
                extracted_strings[orig_idx] = crop_text
    except Exception:
        pass

    return extracted_strings
