
from __future__ import annotations

from typing import Any, Dict, List

DELIVERY_KEYWORDS: List[str] = [
    "delivery",
    "delivered",
    "consignee",
    "consignor",
    "lorry",
    "receipt",
    "pod",
    "proof",
    "vehicle",
    "truck",
    "freight",
    "dispatch",
    "shipment",
    "cargo",
    "goods",
    "transport",
    "gr no",
    "lr no",
    "waybill",
    "invoice",
    "quantity",
]


def is_delivery_document(raw_text: str) -> Dict[str, Any]:
    """Determine if a document is a valid delivery or logistics document.

    Checks raw OCR text for delivery and transport keywords.

    Args:
        raw_text: Full raw OCR text string.

    Returns:
        dict:
            is_delivery_document: bool (True if matched_count >= 3)
            matched_keywords: List[str]
            matched_count: int
    """
    text_lower = (raw_text or "").lower()
    matched_keywords: List[str] = [
        kw for kw in DELIVERY_KEYWORDS if kw in text_lower
    ]
    matched_count = len(matched_keywords)
    is_delivery = matched_count >= 3

    return {
        "is_delivery_document": is_delivery,
        "matched_keywords": matched_keywords,
        "matched_count": matched_count,
    }


def assess_document_quality(
    ocr_result: Dict[str, Any], preprocess_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess document OCR quality and generate review recommendations.

    Args:
        ocr_result: Dictionary containing 'mean_confidence', 'blocks', etc.
        preprocess_meta: Metadata from preprocessing steps (rotation, stamps).

    Returns:
        Flat dictionary containing quality metrics, counts, and recommendation.
    """
    mean_confidence = float(ocr_result.get("mean_confidence", 0.0))
    blocks = ocr_result.get("blocks", [])

    # Quality tier assignment
    if mean_confidence >= 0.75:
        quality_tier = "GOOD"
    elif mean_confidence >= 0.50:
        quality_tier = "DEGRADED"
    else:
        quality_tier = "POOR"

    # Count block metrics
    low_confidence_blocks = sum(
        1 for b in blocks if float(b.get("confidence", 0.0)) < 0.75
    )
    trocr_fallback_count = sum(
        1 for b in blocks if b.get("method") == "TROCR_FALLBACK"
    )

    # Preprocessing metadata extraction
    rotation_corrected = bool(preprocess_meta.get("rotation_corrected", False))
    rotation_angle = int(preprocess_meta.get("rotation_angle", 0))
    stamp_detected = bool(preprocess_meta.get("stamp_detected", False))
    stamp_count = int(preprocess_meta.get("stamp_count", 0))

    # Recommendation determination
    if quality_tier == "POOR":
        recommendation = (
            "MANUAL_REVIEW — OCR confidence too low to trust extraction"
        )
    elif trocr_fallback_count > 5:
        recommendation = (
            "VERIFY — Multiple handwritten fields detected, review flagged fields"
        )
    elif stamp_detected:
        recommendation = (
            "CHECK_STAMP — Stamp may obscure content, verify covered fields"
        )
    else:
        recommendation = (
            "AUTO_APPROVED — Document quality sufficient for automated processing"
        )

    return {
        "mean_confidence": round(mean_confidence, 4),
        "quality_tier": quality_tier,
        "low_confidence_blocks": low_confidence_blocks,
        "trocr_fallback_count": trocr_fallback_count,
        "rotation_corrected": rotation_corrected,
        "rotation_angle": rotation_angle,
        "stamp_detected": stamp_detected,
        "stamp_count": stamp_count,
        "recommendation": recommendation,
    }
