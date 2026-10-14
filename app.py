
import os

# Set PaddlePaddle environment flags for CPU inference stability
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

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pipeline import classifier, extractor, ocr, preprocess

app = Flask(__name__)
# Preserve insertion order so extracted_texts appears first
app.config["JSON_SORT_KEYS"] = False
try:
    app.json.sort_keys = False
except AttributeError:
    pass

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".webp"}


def is_allowed_file(filename: str) -> bool:
    """Validate file extension against allowed formats."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Serve the web interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint validating core engine and library availability."""
    paddle_ok = False
    try:
        import paddle
        import paddleocr

        paddle_ok = True
    except Exception:
        paddle_ok = False

    trocr_ok = False
    try:
        import torch
        import transformers

        trocr_ok = True
    except Exception:
        trocr_ok = False

    pdf_ok = False
    try:
        import pdf2image

        pdf_ok = True
    except Exception:
        pdf_ok = False

    return jsonify({
        "status": "running",
        "paddle": paddle_ok,
        "trocr": trocr_ok,
        "pdf_support": pdf_ok,
    })


def load_document_pages(file_path: str) -> List[Image.Image]:
    """Load all pages from a document (PDF, multi-frame TIFF, or single image) into RGB PIL Images."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        pages = []
        # 1. Prefer pypdfium2 (self-contained, no external Poppler binary needed on Windows)
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(file_path)
            if len(pdf) == 0:
                raise ValueError("PDF document contains no renderable pages.")
            scale = 200.0 / 72.0
            pages = [
                pdf[i].render(scale=scale).to_pil().convert("RGB")
                for i in range(len(pdf))
            ]
        except Exception:
            pages = []

        # 2. Fallback to pdf2image if pypdfium2 is not available
        if not pages:
            try:
                from pdf2image import convert_from_path

                pdf_images = convert_from_path(file_path, dpi=200)
                if not pdf_images:
                    raise ValueError("PDF document contains no renderable pages.")
                pages = [img.convert("RGB") for img in pdf_images]
            except Exception as e:
                raise RuntimeError(
                    f"Unable to render PDF. Please ensure pypdfium2 is installed or Poppler is added to PATH. Error: {e}"
                )
        return pages

    # For images (PNG, JPG, TIFF, WEBP)
    with Image.open(file_path) as img:
        from PIL import ImageSequence

        raw_frames = list(ImageSequence.Iterator(img))
        pages = []
        for frame in raw_frames:
            if frame.mode in ("RGBA", "LA") or (
                frame.mode == "P" and "transparency" in frame.info
            ):
                rgba = frame.convert("RGBA")
                white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
                white_bg.paste(rgba, mask=rgba.split()[3])
                pages.append(white_bg)
            else:
                pages.append(frame.convert("RGB"))
        return pages


def run_pipeline(file_path: str) -> dict:
    """Run the end-to-end POD OCR extraction pipeline across all pages of a document.

    Args:
        file_path: Absolute or relative path to document file (image or PDF).

    Returns:
        dict: Full multi-page extraction response dictionary.
    """
    import numpy as np

    pages_images = load_document_pages(file_path)
    total_pages = len(pages_images)
    if total_pages == 0:
        raise ValueError("No renderable pages found in document.")

    pages_data = []
    all_blocks = []
    all_extracted_texts = []
    raw_text_parts = []
    trocr_used_any = False
    confidences = []
    low_conf_blocks_total = 0
    trocr_fallback_count_total = 0
    rotations = []
    pages_structured_fields = []

    for page_idx, page_img in enumerate(pages_images, start=1):
        # Step a: Preprocessing (rotation correction and resolution cap)
        preprocessed_img, prep_meta = preprocess.preprocess(page_img)
        rotations.append({
            "page": page_idx,
            "rotation_corrected": prep_meta.get("rotation_corrected", False),
            "rotation_angle": prep_meta.get("rotation_angle", 0),
        })

        # Step b: Two-tier OCR text extraction
        page_ocr = ocr.extract_text(preprocessed_img)
        if page_ocr.get("trocr_used"):
            trocr_used_any = True

        page_blocks = page_ocr.get("blocks", [])
        for blk in page_blocks:
            blk["page"] = page_idx
            all_blocks.append(blk)

        page_texts = [b["text"] for b in page_blocks if b.get("text")]
        all_extracted_texts.extend(page_texts)

        if page_ocr.get("raw_text"):
            page_header = f"--- PAGE {page_idx} ---" if total_pages > 1 else ""
            if page_header:
                raw_text_parts.append(page_header)
            raw_text_parts.append(page_ocr["raw_text"])

        for b in page_blocks:
            c = float(b.get("confidence", 0.0))
            confidences.append(c)
            if c < 0.75:
                low_conf_blocks_total += 1
            if b.get("method") == "TROCR_FALLBACK":
                trocr_fallback_count_total += 1

        # Step c: Extract structured fields for this specific page
        page_fields = extractor.extract_fields(page_ocr, preprocessed_img)
        pages_structured_fields.append((page_idx, page_fields))

        pages_data.append({
            "page_number": page_idx,
            "extracted_texts": page_texts,
            "total_blocks": len(page_blocks),
            "mean_confidence": page_ocr.get("mean_confidence", 0.0),
            "rotation_corrected": prep_meta.get("rotation_corrected", False),
            "rotation_angle": prep_meta.get("rotation_angle", 0),
        })

    # Combined OCR output across all pages
    combined_raw_text = "\n".join(raw_text_parts)
    combined_ocr = {
        "blocks": all_blocks,
        "raw_text": combined_raw_text,
    }
    combined_fields = extractor.extract_fields(combined_ocr)

    # Step d: Intelligent multi-page field merge
    structured_fields = extractor.merge_multipage_fields(
        pages_structured_fields, combined_fields
    )

    # Step e: Classify document as delivery document
    doc_classification = classifier.is_delivery_document(combined_raw_text)

    # Step f: Assess document and extraction quality across all pages
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    rotation_corrected_any = any(r["rotation_corrected"] for r in rotations)
    rot_angle = rotations[0]["rotation_angle"] if rotations else 0

    quality_assessment = classifier.assess_document_quality(
        {
            "mean_confidence": mean_confidence,
            "blocks": all_blocks,
        },
        {
            "rotation_corrected": rotation_corrected_any,
            "rotation_angle": rot_angle,
        },
    )
    quality_assessment["total_pages"] = total_pages
    quality_assessment["pages_rotation"] = rotations

    # Assemble response
    return {
        "status": "success",
        "total_pages": total_pages,
        "extracted_texts": all_extracted_texts,
        "pages": pages_data,
        "structured_fields": structured_fields,
        "full_ocr": {
            "total_pages": total_pages,
            "raw_text": combined_raw_text,
            "total_blocks": len(all_blocks),
            "mean_confidence": round(mean_confidence, 4),
            "trocr_used": trocr_used_any,
            "ocr_blocks": all_blocks,
            "ocr_engine": (
                "PaddleOCR+TrOCR"
                if trocr_used_any
                else "PaddleOCR"
            ),
        },
        "document_quality": quality_assessment,
        "document_classification": doc_classification,
    }


@app.route("/extract", methods=["POST"])
def extract():
    """Extract structured logistics fields and full OCR from an uploaded document."""
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file field in request"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"status": "error", "error": "No file selected"}), 400

    filename = uploaded_file.filename
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            jsonify({
                "status": "error",
                "error": f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            }),
            400,
        )

    # Write upload to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)

    try:
        uploaded_file.save(temp_path)
        response_payload = run_pipeline(temp_path)
        if request.args.get("pure", "").lower() in ("true", "1") or request.args.get("format") == "texts":
            return jsonify(response_payload["extracted_texts"])
        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    finally:
        # Guarantee cleanup of temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    # Pre-warm PaddleOCR and TrOCR engines once at process startup
    try:
        ocr.OCREngine.get_paddle()
    except Exception as e:
        print(f"PaddleOCR warmup notice: {e}")

    try:
        ocr.OCREngine.get_trocr()
    except Exception as e:
        print(f"TrOCR warmup notice: {e}")

    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
