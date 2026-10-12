"""Structured Field Extractor for Proof of Delivery (POD) Documents.

Extracts key logistics metadata from OCR results:
- pod_number
- delivery_date
- vehicle_number
- customer_name
- receiver_name
- quantity
- remarks (including exception keywords)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import PIL.Image
from PIL import Image

EXCEPTION_KEYWORDS = [
    "damage",
    "damaged",
    "short",
    "shortage",
    "missing",
    "broken",
    "wet",
    "leak",
    "leakage",
    "refused",
    "reject",
    "partial",
    "excess",
    "delay",
    "lost",
    "torn",
    "open",
]


def _not_found_field() -> Dict[str, Any]:
    """Return standard payload when a field is not found."""
    return {
        "value": None,
        "confidence": None,
        "method": "NOT_FOUND",
        "needs_review": True,
    }


def _get_box_coords(bbox: List[List[int]]) -> Tuple[int, int, int, int]:
    """Extract (x1, y1, x2, y2) from bbox [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _find_nearby_blocks(
    anchor_idx: int,
    blocks: List[Dict[str, Any]],
    max_dx: int = 350,
    max_dy: int = 120,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Find blocks within spatial proximity of the anchor block.

    Returns list of (combined_distance, block) sorted by proximity.
    """
    anchor_box = blocks[anchor_idx]["bbox"]
    ax1, ay1, ax2, ay2 = _get_box_coords(anchor_box)

    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for i, blk in enumerate(blocks):
        if i == anchor_idx:
            continue

        tx1, ty1, tx2, ty2 = _get_box_coords(blk["bbox"])

        # Compute horizontal and vertical distance
        if tx1 >= ax2:
            dx = tx1 - ax2
        elif tx2 <= ax1:
            dx = ax1 - tx2
        else:
            dx = 0

        if ty1 >= ay2:
            dy = ty1 - ay2
        elif ty2 <= ay1:
            dy = ay1 - ty2
        else:
            dy = 0

        if dx < max_dx and dy < max_dy:
            combined_dist = float(dx + dy)
            candidates.append((combined_dist, blk))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _extract_vehicle_number(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract vehicle registration number.

    Pattern: r'\\b([A-Z]{2}\\s?\\d{1,2}\\s?[A-Z]{1,2}\\s?\\d{4})\\b'
    Anchors: ["TRUCK NO", "VEHICLE NO", "VEHICLE NUMBER", "LORRY NO", "LR NO"]
    """
    veh_regex = re.compile(
        r"\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4})\b", re.IGNORECASE
    )
    anchor_labels = [
        "TRUCK NO",
        "VEHICLE NO",
        "VEHICLE NUMBER",
        "LORRY NO",
        "LR NO",
    ]

    # 1. Check anchor blocks and nearby blocks
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        if any(label in text_upper for label in anchor_labels):
            # Check if value is in same block
            match = veh_regex.search(blk["text"])
            if match:
                clean_val = re.sub(r"\s+", "", match.group(1)).upper()
                return {
                    "value": clean_val,
                    "confidence": "HIGH",
                    "method": "NEAR_LABEL",
                    "needs_review": False,
                }

            # Check spatially adjacent blocks
            nearby = _find_nearby_blocks(
                idx, blocks, max_dx=350, max_dy=120
            )
            for _, near_blk in nearby:
                match = veh_regex.search(near_blk["text"])
                if match:
                    clean_val = re.sub(r"\s+", "", match.group(1)).upper()
                    return {
                        "value": clean_val,
                        "confidence": "HIGH",
                        "method": "NEAR_LABEL",
                        "needs_review": False,
                    }

    # 2. Fallback regex on all blocks / raw_text
    match = veh_regex.search(raw_text)
    if match:
        clean_val = re.sub(r"\s+", "", match.group(1)).upper()
        return {
            "value": clean_val,
            "confidence": "MEDIUM",
            "method": "REGEX",
            "needs_review": False,
        }

    return _not_found_field()


def _extract_delivery_date(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract delivery date.

    Patterns: r'\\b(\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4})\\b' and r'\\b(\\d{4}[-/]\\d{1,2}[-/]\\d{1,2})\\b'
    """
    date_patterns = [
        re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    ]

    # Search blocks
    for blk in blocks:
        for pat in date_patterns:
            match = pat.search(blk["text"])
            if match:
                return {
                    "value": match.group(1).strip(),
                    "confidence": "HIGH",
                    "method": "REGEX_DATE",
                    "needs_review": False,
                }

    # Search raw_text
    for pat in date_patterns:
        match = pat.search(raw_text)
        if match:
            return {
                "value": match.group(1).strip(),
                "confidence": "HIGH",
                "method": "REGEX_DATE",
                "needs_review": False,
            }

    return _not_found_field()


def _extract_pod_number(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract POD / Invoice / LR number.

    Anchors: ["INV", "INVOICE NO", "GR NO", "LR NO", "POD", "SERIAL NO", "GR No", "LR No"]
    Fallback: r'(?:INV|GR|LR|POD|S\\.NO)[-\\s#]?([A-Z0-9\\-/]{3,})'
    """
    anchor_labels = [
        "INVOICE NO",
        "GR NO",
        "LR NO",
        "SERIAL NO",
        "GR NO",
        "LR NO",
        "POD",
        "INV",
    ]
    pod_val_pat = re.compile(r"([A-Z0-9\-/]{3,})", re.IGNORECASE)

    # 1. Search near labels
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in anchor_labels:
            if label in text_upper:
                # Value in same block?
                after = re.split(re.escape(label), blk["text"], flags=re.IGNORECASE)[-1]
                m = pod_val_pat.search(after.strip(" :#-"))
                if m and len(m.group(1)) >= 3:
                    return {
                        "value": m.group(1).strip(),
                        "confidence": "HIGH",
                        "method": "NEAR_LABEL",
                        "needs_review": False,
                    }

                # Value in nearby blocks
                nearby = _find_nearby_blocks(
                    idx, blocks, max_dx=350, max_dy=120
                )
                for _, near_blk in nearby:
                    m = pod_val_pat.search(near_blk["text"].strip(" :#-"))
                    if m and len(m.group(1)) >= 3:
                        return {
                            "value": m.group(1).strip(),
                            "confidence": "HIGH",
                            "method": "NEAR_LABEL",
                            "needs_review": False,
                        }

    # 2. Fallback regex
    fallback_pat = re.compile(
        r"(?:INV|GR|LR|POD|S\.NO)[-\s#:]*([A-Z0-9\-/]{3,})", re.IGNORECASE
    )
    m = fallback_pat.search(raw_text)
    if m:
        return {
            "value": m.group(1).strip(),
            "confidence": "MEDIUM",
            "method": "REGEX",
            "needs_review": False,
        }

    return _not_found_field()


def _extract_name_near_label(
    blocks: List[Dict[str, Any]], labels: List[str]
) -> Dict[str, Any]:
    """Helper to extract names near specified anchor labels (smallest combined x+y distance)."""
    known_field_labels = [
        "INVOICE", "INV", "GR NO", "LR NO", "POD", "DATE", "DELIVERY",
        "TRUCK", "VEHICLE", "LORRY", "CONSIGNEE", "BILL TO", "CUSTOMER",
        "RECEIVER", "RECEIVED BY", "SIGNATORY", "QTY", "QUANTITY", "REMARK"
    ]

    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in labels:
            if label in text_upper:
                # Check if name is in the same block after label
                after = re.split(re.escape(label), blk["text"], flags=re.IGNORECASE)[-1].strip(" :-#\t")
                if len(after) >= 3 and not re.match(r"^[\d\W]+$", after):
                    conf = "LOW" if blk.get("method") == "TROCR_FALLBACK" else "MEDIUM"
                    return {
                        "value": after,
                        "confidence": conf,
                        "method": "NEAR_LABEL",
                        "needs_review": (conf == "LOW"),
                    }

                # Find nearest block that is not another form label, prioritizing same row / below
                anchor_box = blk["bbox"]
                ax1, ay1, ax2, ay2 = _get_box_coords(anchor_box)

                candidates = []
                for j, other_blk in enumerate(blocks):
                    if j == idx:
                        continue
                    other_upper = other_blk["text"].upper()
                    if any(l in other_upper for l in known_field_labels):
                        continue
                    tx1, ty1, tx2, ty2 = _get_box_coords(other_blk["bbox"])

                    # Must be to the right (dx) or below (dy)
                    if tx1 >= ax2 - 20 and abs(ty1 - ay1) < 60:
                        dist = float(abs(tx1 - ax2) + abs(ty1 - ay1))
                    elif ty1 >= ay2 - 10 and abs(tx1 - ax1) < 200:
                        dist = float(abs(tx1 - ax1) + abs(ty1 - ay2) * 1.5)
                    else:
                        continue

                    candidates.append((dist, other_blk))

                candidates.sort(key=lambda item: item[0])

                for _, near_blk in candidates:
                    cand_text = near_blk["text"].strip(" :-#\t")
                    if len(cand_text) >= 3 and not re.match(r"^[\d\W]+$", cand_text):
                        conf = "LOW" if near_blk.get("method") == "TROCR_FALLBACK" else "MEDIUM"
                        return {
                            "value": cand_text,
                            "confidence": conf,
                            "method": "NEAR_LABEL",
                            "needs_review": (conf == "LOW"),
                        }

    return _not_found_field()


def _extract_customer_name(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Search near labels: ["CONSIGNEE", "BILL TO", "CUSTOMER", "DELIVERY TO"]."""
    labels = ["CONSIGNEE", "BILL TO", "CUSTOMER", "DELIVERY TO"]
    return _extract_name_near_label(blocks, labels)


def _extract_receiver_name(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Search near labels: ["RECEIVER", "RECEIVED BY", "AUTHORISED SIGNATORY", "RECIPIENT"]."""
    labels = ["RECEIVER", "RECEIVED BY", "AUTHORISED SIGNATORY", "RECIPIENT"]
    return _extract_name_near_label(blocks, labels)


def _extract_quantity(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract shipment quantity.

    Anchors: ["QTY", "QUANTITY", "ITEMS IN TOTAL", "TOTAL ITEMS", "NO OF PACKAGES"]
    Fallback: r'(?:qty|quantity|pieces?|pcs|units?|nos?)[^\\d]*(\\d+[\\.,]?\\d*)'
    """
    labels = ["ITEMS IN TOTAL", "TOTAL ITEMS", "NO OF PACKAGES", "QUANTITY", "QTY"]
    qty_pat = re.compile(r"(\d+(?:[.,]\d+)?)")

    # 1. Search near labels
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in labels:
            if label in text_upper:
                # Value in same block?
                after = re.split(re.escape(label), blk["text"], flags=re.IGNORECASE)[-1]
                m = qty_pat.search(after)
                if m:
                    clean_val = m.group(1).replace(",", "")
                    return {
                        "value": clean_val,
                        "confidence": "HIGH",
                        "method": "NEAR_LABEL",
                        "needs_review": False,
                    }

                # Value in nearby blocks
                nearby = _find_nearby_blocks(
                    idx, blocks, max_dx=350, max_dy=120
                )
                for _, near_blk in nearby:
                    m = qty_pat.search(near_blk["text"])
                    if m:
                        clean_val = m.group(1).replace(",", "")
                        return {
                            "value": clean_val,
                            "confidence": "HIGH",
                            "method": "NEAR_LABEL",
                            "needs_review": False,
                        }

    # 2. Fallback regex
    fallback_pat = re.compile(
        r"(?:qty|quantity|pieces?|pcs|units?|nos?)[^\d]*(\d+[\.,]?\d*)",
        re.IGNORECASE,
    )
    m = fallback_pat.search(raw_text)
    if m:
        clean_val = m.group(1).replace(",", "")
        return {
            "value": clean_val,
            "confidence": "MEDIUM",
            "method": "REGEX",
            "needs_review": False,
        }

    return _not_found_field()


def _extract_remarks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract remarks or notes and detect exception keywords.

    Search blocks containing 'REMARK' or 'NOTE'.
    Collect next 3 blocks after label block.
    """
    remark_idx = -1
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        if "REMARK" in text_upper or "NOTE" in text_upper:
            remark_idx = idx
            break

    if remark_idx == -1:
        return _not_found_field()

    label_block = blocks[remark_idx]
    after_in_label = re.split(
        r"REMARKS?|NOTES?", label_block["text"], flags=re.IGNORECASE
    )[-1].strip(" :-#\t")

    collected_texts: List[str] = []
    collected_blocks: List[Dict[str, Any]] = []

    if after_in_label:
        collected_texts.append(after_in_label)
        collected_blocks.append(label_block)

    # Collect next 3 blocks after the label block
    for i in range(remark_idx + 1, min(len(blocks), remark_idx + 4)):
        blk = blocks[i]
        collected_texts.append(blk["text"].strip())
        collected_blocks.append(blk)

    remarks_val = " ".join(t for t in collected_texts if t).strip()
    if not remarks_val:
        return _not_found_field()

    # Check for exception keywords
    found_exceptions: List[str] = []
    val_lower = remarks_val.lower()
    for kw in EXCEPTION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", val_lower):
            found_exceptions.append(kw)

    # Check if TrOCR was used
    trocr_used = any(b.get("method") == "TROCR_FALLBACK" for b in collected_blocks)
    conf = "LOW" if trocr_used else "MEDIUM"
    needs_review = trocr_used or len(found_exceptions) > 0

    result: Dict[str, Any] = {
        "value": remarks_val,
        "confidence": conf,
        "method": "NEAR_LABEL",
        "needs_review": needs_review,
    }

    if found_exceptions:
        result["exception_keywords"] = found_exceptions

    return result


def extract_fields(
    ocr_result: Dict[str, Any], original_image: Image.Image
) -> Dict[str, Any]:
    """Extract structured fields from OCR output.

    Args:
        ocr_result: Dictionary containing 'blocks' (List[dict]) and 'raw_text' (str).
        original_image: PIL Image of the original document.

    Returns:
        Dictionary containing extracted fields:
        - pod_number
        - delivery_date
        - vehicle_number
        - customer_name
        - receiver_name
        - quantity
        - remarks
    """
    blocks = ocr_result.get("blocks", [])
    raw_text = ocr_result.get("raw_text", "")

    return {
        "pod_number": _extract_pod_number(blocks, raw_text),
        "delivery_date": _extract_delivery_date(blocks, raw_text),
        "vehicle_number": _extract_vehicle_number(blocks, raw_text),
        "customer_name": _extract_customer_name(blocks),
        "receiver_name": _extract_receiver_name(blocks),
        "quantity": _extract_quantity(blocks, raw_text),
        "remarks": _extract_remarks(blocks),
    }
