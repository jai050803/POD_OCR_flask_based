
import os
import tempfile
from pathlib import Path
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


def run_pipeline(file_path: str) -> dict:
    """Run the end-to-end POD OCR extraction pipeline on a file path (image or PDF).

    Args:
        file_path: Absolute or relative path to document file.

    Returns:
        dict: Full extraction response dictionary.
    """
    ext = Path(file_path).suffix.lower()

    # Load document into PIL Image
    if ext == ".pdf":
        pil_image = None
        # 1. Prefer pypdfium2 (self-contained, no external Poppler binary needed on Windows)
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(file_path)
            if len(pdf) == 0:
                raise ValueError("PDF document contains no renderable pages.")
            scale = 200.0 / 72.0
            pil_image = pdf[0].render(scale=scale).to_pil().convert("RGB")
        except Exception:
            pass

        # 2. Fallback to pdf2image if pypdfium2 is not available
        if pil_image is None:
            try:
                from pdf2image import convert_from_path

                pdf_images = convert_from_path(file_path, dpi=200)
                if not pdf_images:
                    raise ValueError("PDF document contains no renderable pages.")
                pil_image = pdf_images[0].convert("RGB")
            except Exception as e:
                raise RuntimeError(
                    f"Unable to render PDF. Please ensure pypdfium2 is installed or Poppler is added to PATH. Error: {e}"
                )
    else:
        with Image.open(file_path) as img:
            pil_image = img.convert("RGB")

    # Step a: Preprocessing (rotation correction, resolution cap, CLAHE, thresholding)
    preprocessed_image, preprocess_meta = preprocess.preprocess(pil_image)

    # Step b: Detect and mask stamps
    stamp_result = preprocess.detect_and_mask_stamps(preprocessed_image)
    preprocess_meta["stamp_detected"] = stamp_result["stamp_count"] > 0
    preprocess_meta["stamp_count"] = stamp_result["stamp_count"]
    preprocess_meta["stamp_crops"] = stamp_result["stamp_crops"]

    # Step c: Two-tier OCR text extraction
    ocr_result = ocr.extract_text(preprocessed_image)

    # Step d: OCR on isolated stamp crops
    stamp_texts = ocr.ocr_stamp_crops(stamp_result["stamp_crops"])
    ocr_result["stamp_texts"] = stamp_texts

    # Step e: Extract structured logistics fields
    structured_fields = extractor.extract_fields(
        ocr_result, preprocessed_image
    )

    # Step f: Classify document as delivery document
    doc_classification = classifier.is_delivery_document(
        ocr_result["raw_text"]
    )

    # Step g: Assess document and extraction quality
    quality_assessment = classifier.assess_document_quality(
        ocr_result, preprocess_meta
    )

    # Clean list of only the extracted texts one by one (no bboxes or confidence scores)
    extracted_texts = [b["text"] for b in ocr_result.get("blocks", []) if b.get("text")]

    # Assemble response
    return {
        "status": "success",
        "extracted_texts": extracted_texts,
        "structured_fields": structured_fields,
        "full_ocr": {
            "raw_text": ocr_result["raw_text"],
            "total_blocks": ocr_result["total_blocks"],
            "mean_confidence": ocr_result["mean_confidence"],
            "trocr_used": ocr_result["trocr_used"],
            "ocr_blocks": ocr_result["blocks"],
            "stamp_texts": stamp_texts,
            "ocr_engine": (
                "PaddleOCR+TrOCR"
                if ocr_result["trocr_used"]
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
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
