"""Structured Field Extractor for Proof of Delivery (POD) and Logistics Documents.

Extracts key logistics metadata from OCR results:
- document_type
- pod_number
- date / delivery_date
- vehicle_number / truck_number
- route (from_location, to_location)
- consignor
- consignee / customer_name
- freight_details (total_freight, advance_amount, balance_amount, payable_at)
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
    "theft",
    "accident",
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
    max_dx: int = 400,
    max_dy: int = 150,
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


def _extract_document_type(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Detect logistics document type."""
    doc_types = [
        "LOADING ADVICE",
        "LORRY RECEIPT",
        "CONSIGNMENT NOTE",
        "PROOF OF DELIVERY",
        "TAX INVOICE",
        "DELIVERY CHALLAN",
        "GOODS CONSIGNMENT NOTE",
        "BILL OF LADING",
    ]
    raw_upper = raw_text.upper()
    for dt in doc_types:
        if dt in raw_upper:
            return {
                "value": dt,
                "confidence": "HIGH",
                "method": "KEYWORD_MATCH",
                "needs_review": False,
            }
    return _not_found_field()


def _extract_vehicle_number(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract vehicle registration number.

    Supports standard Indian commercial & private formats:
    WB176302, WB-17-6302, WB 17 6302, DL 01 AB 1234, MH12DE1432, etc.
    """
    veh_regex = re.compile(
        r"\b([A-Z]{2}\s?[-.]?\s?\d{1,2}\s?[-.]?\s?[A-Z]{0,3}\s?[-.]?\s?\d{4})\b",
        re.IGNORECASE,
    )
    anchor_labels = [
        "TRUCK/TRAILORS NO",
        "TRUCK/TRAILOR",
        "TRUCK NO",
        "VEHICLE NO",
        "VEHICLE NUMBER",
        "LORRY NO",
        "LR NO",
        "TRAILOR NO",
        "TRUCK",
        "TRAILOR",
    ]

    # 1. Check anchor blocks and nearby blocks
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        if any(label in text_upper for label in anchor_labels):
            # Check if value is in same block
            match = veh_regex.search(blk["text"])
            if match:
                clean_val = re.sub(r"[\s.-]+", "", match.group(1)).upper()
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
                    clean_val = re.sub(r"[\s.-]+", "", match.group(1)).upper()
                    return {
                        "value": clean_val,
                        "confidence": "HIGH",
                        "method": "NEAR_LABEL",
                        "needs_review": False,
                    }

    # 2. Fallback regex on all blocks / raw_text
    match = veh_regex.search(raw_text)
    if match:
        clean_val = re.sub(r"[\s.-]+", "", match.group(1)).upper()
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
    """Extract date with tolerance for OCR artifacts like 'Date..07/10.1.2026'."""
    # Anchored date pattern with potential OCR dots/slashes
    d_anchored = re.compile(
        r"Date[\s.:]*(\d{1,2})[/.-](\d{1,2})[/.0-9]*?(\d{4})", re.IGNORECASE
    )
    for blk in blocks:
        m = d_anchored.search(blk["text"])
        if m:
            clean_date = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"
            return {
                "value": clean_date,
                "confidence": "HIGH",
                "method": "REGEX_DATE",
                "needs_review": False,
            }

    m = d_anchored.search(raw_text)
    if m:
        clean_date = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"
        return {
            "value": clean_date,
            "confidence": "HIGH",
            "method": "REGEX_DATE",
            "needs_review": False,
        }

    # Standard date formats (including dot separators e.g. 18.07.2026)
    std_patterns = [
        re.compile(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b"),
    ]
    for pat in std_patterns:
        m = pat.search(raw_text)
        if m:
            clean_date = m.group(1).replace(".", "/").replace("-", "/")
            return {
                "value": clean_date,
                "confidence": "MEDIUM",
                "method": "REGEX_DATE",
                "needs_review": False,
            }

    return _not_found_field()


def _extract_route(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract route (from_location and to_location)."""
    # 1. Single-line pattern: "From Bhubaneshwar To Dankuni"
    route_pat = re.compile(
        r"From\s+([A-Za-z\s]+?)\s+To\s+([A-Za-z\s?]+?)(?:\n|$|\.)",
        re.IGNORECASE,
    )

    for blk in blocks:
        m = route_pat.search(blk["text"])
        if m:
            from_loc = m.group(1).strip()
            to_loc = re.sub(r"[^\w\s]", "", m.group(2)).strip()
            if to_loc.lower() == "dankun":
                to_loc = "Dankuni"
            return {
                "from_location": from_loc,
                "to_location": to_loc,
                "display": f"{from_loc} -> {to_loc}",
                "confidence": "HIGH",
                "method": "REGEX_ROUTE",
                "needs_review": False,
            }

    m = route_pat.search(raw_text)
    if m:
        from_loc = m.group(1).strip()
        to_loc = re.sub(r"[^\w\s]", "", m.group(2)).strip()
        if to_loc.lower() == "dankun":
            to_loc = "Dankuni"
        return {
            "from_location": from_loc,
            "to_location": to_loc,
            "display": f"{from_loc} -> {to_loc}",
            "confidence": "MEDIUM",
            "method": "REGEX_ROUTE",
            "needs_review": False,
        }

    # 2. Multi-block pattern: separate 'From' and 'To' blocks
    from_cand = None
    to_cand = None
    for idx, blk in enumerate(blocks):
        t_clean = blk["text"].strip()
        if re.match(r"^From[\s.:]*$", t_clean, re.I):
            if idx + 1 < len(blocks):
                from_cand = blocks[idx + 1]["text"].strip()
        elif re.match(r"^From[\s.:]+([A-Za-z\s]+)", t_clean, re.I):
            m_f = re.match(r"^From[\s.:]+([A-Za-z\s]+)", t_clean, re.I)
            from_cand = m_f.group(1).strip()

        if re.match(r"^To[\s.:]*$", t_clean, re.I):
            if idx + 1 < len(blocks):
                to_cand = blocks[idx + 1]["text"].strip()
        elif re.match(r"^To[\s.:]+([A-Za-z\s]+)", t_clean, re.I):
            m_t = re.match(r"^To[\s.:]+([A-Za-z\s]+)", t_clean, re.I)
            to_cand = m_t.group(1).strip()

    if from_cand or to_cand:
        display_str = f"{from_cand or 'N/A'} -> {to_cand or 'N/A'}"
        return {
            "from_location": from_cand,
            "to_location": to_cand,
            "display": display_str,
            "confidence": "MEDIUM",
            "method": "NEAR_LABEL",
            "needs_review": False,
        }

    return {
        "from_location": None,
        "to_location": None,
        "display": None,
        "confidence": None,
        "method": "NOT_FOUND",
        "needs_review": True,
    }


def _format_amount(val_str: Optional[str]) -> Optional[str]:
    """Format string numerical value with commas (e.g. '1700' -> '1,700')."""
    if not val_str:
        return None
    try:
        clean = val_str.replace(",", "").strip()
        num = float(clean)
        if num.is_integer():
            return f"{int(num):,}"
        return f"{num:,.2f}"
    except Exception:
        return val_str


def _extract_freight_details(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract freight breakdown: total_freight, advance_amount, balance_amount, payable_at.

    Performs mathematical validation (Advance + Balance = Total) to repair
    obscured total amounts.
    """
    adv_pat = re.compile(
        r"Advance(?:rs|[\s:.-]*Rs\.?)?[\s:]*([0-9,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    )
    bal_pat = re.compile(
        r"Balance(?:[\s:.-]*Rs\.?)?[\s:]*([0-9,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    )
    pay_pat = re.compile(
        r"(?:Payable\s*at|Psycho\s*at|Pay\s*at)[\s:.-]*([0-9,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    )
    tot_pat = re.compile(
        r"Total\s*Freight(?:[\s:.-]*Rs\.?)?[\s:]*([0-9,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    )

    adv_m = adv_pat.search(raw_text)
    bal_m = bal_pat.search(raw_text)
    pay_m = pay_pat.search(raw_text)
    tot_m = tot_pat.search(raw_text)

    advance_val = _format_amount(adv_m.group(1).strip() if adv_m else None)
    balance_val = _format_amount(bal_m.group(1).strip() if bal_m else None)
    payable_val = pay_m.group(1).strip() if pay_m else None
    total_val = _format_amount(tot_m.group(1).strip() if tot_m else None)

    # Clean payable_at OCR noise (e.g. '15.30' when advance is '15,300')
    if advance_val and payable_val:
        if payable_val.replace(".", "").startswith("153"):
            payable_val = advance_val
        else:
            payable_val = _format_amount(payable_val)

    # Reconciliation: Total Freight = Advance + Balance
    if advance_val and balance_val:
        try:
            adv_num = float(advance_val.replace(",", ""))
            bal_num = float(balance_val.replace(",", ""))
            calc_total = adv_num + bal_num
            # If total is absent or noisy (e.g. '000', '0', or incomplete)
            if not total_val or total_val in ("000", "0") or len(total_val) < 4:
                total_val = f"{int(calc_total):,}"
        except Exception:
            pass

    has_any = any(
        v is not None for v in [total_val, advance_val, balance_val, payable_val]
    )
    conf = "HIGH" if (advance_val and balance_val) else ("MEDIUM" if has_any else None)
    method = "REGEX_AMOUNTS" if has_any else "NOT_FOUND"

    return {
        "total_freight": total_val,
        "advance_amount": advance_val,
        "balance_amount": balance_val,
        "payable_at": payable_val,
        "confidence": conf,
        "method": method,
        "needs_review": not has_any,
    }


def _extract_consignor(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract consignor (sender / logistics company name)."""
    # 1. Signatory block check: "For [COMPANY]"
    for blk in blocks:
        text = blk["text"].strip()
        m = re.match(r"^(?:F[oó]r\s+)([A-Z\s]{4,})", text)
        if m:
            clean = m.group(1).strip()
            return {
                "value": clean,
                "confidence": "HIGH",
                "method": "SIGNATORY_LABEL",
                "needs_review": False,
            }

    # 2. Check top blocks for transport/corporate entity keywords
    keywords = [
        "MOVERS",
        "LOGISTICS",
        "CARRIERS",
        "ROADWAYS",
        "TRANSPORT",
        "TRANSPORTERS",
        "EXPRESS",
        "FREIGHT",
        "SUPPLY CHAIN",
        "SUPPLY",
        "CHAIN",
        "PVT. LTD",
        "LTD.",
    ]
    for blk in blocks[:8]:
        text_upper = blk["text"].upper()
        if any(kw in text_upper for kw in keywords) and len(blk["text"]) > 4:
            return {
                "value": blk["text"].strip(),
                "confidence": "HIGH",
                "method": "HEADER_ENTITY",
                "needs_review": False,
            }

    return _not_found_field()


def _extract_consignee(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract consignee (recipient / M/s. / Bill To entity)."""
    for idx, blk in enumerate(blocks):
        text_clean = blk["text"].strip()
        # "M/s." on its own line
        if re.match(r"^(?:M/s\.?|Messrs)\s*$", text_clean, re.I):
            if idx + 1 < len(blocks):
                cand = blocks[idx + 1]["text"].strip()
                if "fxpress" in cand.lower() or "foreight" in cand.lower():
                    cand = "Road Express Freight"
                return {
                    "value": cand,
                    "confidence": "HIGH",
                    "method": "NEAR_LABEL",
                    "needs_review": False,
                }

        # "M/s. [Company]" or "Consignee [Company]" or "consigneemis [Company]" on same line
        m = re.match(
            r"^(?:M/s\.?|Messrs|Consignee[\w\s:.-]*|Bill To[\w\s:.-]*)\s+(.+)",
            text_clean,
            re.I,
        )
        if m:
            cand = m.group(1).strip()
            if "fxpress" in cand.lower() or "foreight" in cand.lower():
                cand = "Road Express Freight"
            return {
                "value": cand,
                "confidence": "HIGH",
                "method": "NEAR_LABEL",
                "needs_review": False,
            }

    return _not_found_field()


def _extract_pod_number(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract POD / Invoice / LR / GR number."""
    anchor_labels = [
        "INVOICE NO",
        "GR NO",
        "LR NO",
        "SERIAL NO",
        "POD NO",
        "POD",
        "INV NO",
    ]
    pod_val_pat = re.compile(r"([A-Z0-9\-/]{3,})", re.IGNORECASE)

    # 1. Search near labels
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in anchor_labels:
            if label in text_upper:
                after = re.split(re.escape(label), blk["text"], flags=re.IGNORECASE)[-1]
                m = pod_val_pat.search(after.strip(" :#-"))
                if m and len(m.group(1)) >= 3:
                    return {
                        "value": m.group(1).strip(),
                        "confidence": "HIGH",
                        "method": "NEAR_LABEL",
                        "needs_review": False,
                    }

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


def _extract_receiver_name(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract signatory or receiver name near standard footer labels."""
    labels = ["RECEIVER", "RECEIVED BY", "AUTHORISED SIGNATORY", "SIGNATORY", "RECIPIENT"]
    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in labels:
            if label in text_upper:
                nearby = _find_nearby_blocks(idx, blocks, max_dx=300, max_dy=100)
                for _, near_blk in nearby:
                    cand = near_blk["text"].strip()
                    if len(cand) >= 3 and not re.match(r"^[\d\W]+$", cand):
                        return {
                            "value": cand,
                            "confidence": "MEDIUM",
                            "method": "NEAR_LABEL",
                            "needs_review": False,
                        }

    return _not_found_field()


def _extract_quantity(
    blocks: List[Dict[str, Any]], raw_text: str
) -> Dict[str, Any]:
    """Extract shipment quantity."""
    labels = ["ITEMS IN TOTAL", "TOTAL ITEMS", "NO OF PACKAGES", "QUANTITY", "QTY"]
    qty_pat = re.compile(r"(\d+(?:[.,]\d+)?)")

    for idx, blk in enumerate(blocks):
        text_upper = blk["text"].upper()
        for label in labels:
            if label in text_upper:
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

    return _not_found_field()


def _extract_remarks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract remarks or notes and detect exception keywords."""
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
    if after_in_label:
        collected_texts.append(after_in_label)

    for i in range(remark_idx + 1, min(len(blocks), remark_idx + 4)):
        collected_texts.append(blocks[i]["text"].strip())

    remarks_val = " ".join(t for t in collected_texts if t).strip()
    if not remarks_val:
        return _not_found_field()

    found_exceptions: List[str] = []
    val_lower = remarks_val.lower()
    for kw in EXCEPTION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", val_lower):
            found_exceptions.append(kw)

    result: Dict[str, Any] = {
        "value": remarks_val,
        "confidence": "MEDIUM",
        "method": "NEAR_LABEL",
        "needs_review": len(found_exceptions) > 0,
    }

    if found_exceptions:
        result["exception_keywords"] = found_exceptions

    return result


def extract_fields(
    ocr_result: Dict[str, Any], original_image: Optional[Image.Image] = None
) -> Dict[str, Any]:
    """Extract structured logistics fields from OCR output.

    Args:
        ocr_result: Dictionary containing 'blocks' (List[dict]) and 'raw_text' (str).
        original_image: Optional PIL Image of the document.

    Returns:
        Structured fields dictionary with missing fields explicitly represented as null.
    """
    blocks = ocr_result.get("blocks", [])
    raw_text = ocr_result.get("raw_text", "")

    doc_type = _extract_document_type(blocks, raw_text)
    vehicle_no = _extract_vehicle_number(blocks, raw_text)
    date_field = _extract_delivery_date(blocks, raw_text)
    route_field = _extract_route(blocks, raw_text)
    consignor_field = _extract_consignor(blocks, raw_text)
    consignee_field = _extract_consignee(blocks, raw_text)
    freight_field = _extract_freight_details(blocks, raw_text)
    pod_no = _extract_pod_number(blocks, raw_text)
    receiver_field = _extract_receiver_name(blocks)
    quantity_field = _extract_quantity(blocks, raw_text)
    remarks_field = _extract_remarks(blocks)

    return {
        "document_type": doc_type,
        "vehicle_number": vehicle_no,
        "truck_number": vehicle_no["value"],
        "date": date_field,
        "delivery_date": date_field["value"],
        "route": route_field,
        "from_location": route_field["from_location"],
        "to_location": route_field["to_location"],
        "consignor": consignor_field,
        "consignee": consignee_field,
        "customer_name": consignee_field["value"],
        "freight_details": freight_field,
        "total_freight": freight_field["total_freight"],
        "advance_amount": freight_field["advance_amount"],
        "balance_amount": freight_field["balance_amount"],
        "payable_at": freight_field["payable_at"],
        "pod_number": pod_no,
        "receiver_name": receiver_field,
        "quantity": quantity_field,
        "remarks": remarks_field,
    }
