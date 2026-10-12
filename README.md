# Proof of Delivery (POD) OCR & Extraction Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![OCR Engine](https://img.shields.io/badge/OCR-PaddleOCR%20%2B%20TrOCR-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An automated logistics document processing pipeline designed to ingest, clean, and extract structured data from Proof of Delivery (POD) documents, freight bills, and transport receipts. 

The system features **computer vision preprocessing**, **rubber stamp detection and masking**, a **two-tier OCR engine** (combining PaddleOCR DBNet+CRNN with a TrOCR handwriting fallback), and a **spatial/heuristic field extractor** that maps unstructured text to structured logistics records.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                              Document Ingestion                                   |
|                Supports PDF (pypdfium2 / pdf2image) and Images (PIL)             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           Computer Vision Preprocessing                           |
|  - Resolution Cap (1800px max edge via LANCZOS downsampling)                      |
|  - Orientation Classification & Deskew (0°, 90°, 180°, 270°)                       |
|  - Gray-World Color Balance (LAB space channel gain normalization)                |
|  - Contrast Limited Adaptive Histogram Equalization (CLAHE)                      |
|  - Adaptive Gaussian Binarization                                                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        Stamp Detection & Color Masking                            |
|  - HSV Color Distance Segmentation (Red & Blue/Violet ink profiles)              |
|  - Contour Bounding Area & Aspect Ratio Filtering                                 |
|  - Background Text Masking & Dedicated Batched Stamp OCR                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            Two-Tier OCR Recognition Engine                        |
|  - Tier 1: PaddleOCR (DBNet text detection + CRNN text recognition)               |
|  - Tier 2 Fallback: TrOCR (VisionEncoderDecoder) for blocks with confidence < 0.75|
+-----------------------------------------------------------------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
+---------------------------------------+ +---------------------------------------+
|      Structured Field Extraction      | |    Quality & Document Verification    |
|  - Vehicle / Truck Number             | |  - Keyword Density Classifier         |
|  - POD / Invoice Number               | |  - Document Quality Tier (GOOD/       |
|  - Delivery Date                      | |    DEGRADED/POOR)                     |
|  - Customer / Consignee Name          | |  - Operational ERP Recommendation     |
|  - Receiver / Signatory Name          | +---------------------------------------+
|  - Quantity / Units                   |
|  - Remarks (17 Exception Keywords)    |
+---------------------------------------+
                    |
                    +--------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               Output Delivery                                     |
|  - Web UI: 5-Tab Dark Dashboard (Pure JSON, Fields, Quality, Full OCR, Table)     |
|  - REST API: JSON First payload with optional pure text array format              |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

1. **Pure Text JSON Output First**: Returns an enumerated, cleaned array of all identified text lines in insertion order without bounding boxes or confidence clutter.
2. **Robust Handling of Real-World Physical Documents**: Specifically handles faint carbon copies, 9-pin dot-matrix printouts, mobile camera tilt/shadows, and crumpled paper through targeted CLAHE and adaptive binarization.
3. **Rubber Stamp Isolation**: Detects official company warehouse ink stamps, masks them to prevent distortion of background text, and transcribes stamp contents separately.
4. **Two-Tier OCR Engine**:
   - **Tier 1 (PaddleOCR)**: High-throughput DBNet detection and CRNN printed recognition.
   - **Tier 2 (TrOCR)**: Automatically engages Microsoft's `trocr-base-handwritten` VisionEncoderDecoder for low-confidence blocks (signatures, handwritten driver names).
5. **Spatial & Regex Logistics Field Extractor**:
   - `vehicle_number`: Indian license plate regex matching with spatial proximity search to anchor labels (`TRUCK NO`, `LORRY NO`, `VEHICLE NO`).
   - `pod_number`: Alphanumeric token extraction near consignment and invoice labels.
   - `delivery_date`: Multi-format date parser (`DD/MM/YYYY`, `YYYY-MM-DD`, etc.) with calendar validation.
   - `customer_name`: Spatial proximity extraction below consignee headers.
   - `receiver_name`: Bottom-quadrant signature block extraction.
   - `quantity`: Quantity and unit pattern detection.
   - `remarks`: Defect and exception scanning across 17 freight keywords (`damage`, `torn`, `leak`, `seal broken`, `shortage`, etc.).
6. **Built-in Quality Assurance**: Assesses mean confidence, rotation correction, and stamp detection, returning actionable operational recommendations (`READY FOR ERP` vs `MANUAL REVIEW`).

---

## Repository Structure

```
.
|-- app.py                     # Main Flask web application and REST API endpoints
|-- test_pipeline.py           # CLI script to test the end-to-end extraction pipeline
|-- requirements.txt           # Python dependencies
|-- sample_pod.pdf             # Sample delivery document for immediate testing
|-- .gitignore                 # Standard Python gitignore rules
|-- pipeline/
|   |-- __init__.py            # Pipeline package initializer
|   |-- preprocess.py          # Resolution capping, CLAHE, deskew, and stamp masking
|   |-- ocr.py                 # Singleton OCREngine, PaddleOCR + TrOCR fallback
|   |-- extractor.py           # Regex and spatial logistics field extraction
|   `-- classifier.py          # Document verification and quality grading
`-- templates/
    `-- index.html             # Responsive dark-theme frontend dashboard
```

---

## Prerequisites & Installation

### 1. Prerequisites
* **Python**: `3.10` or higher recommended.
* **Operating System**: Linux, macOS, or Windows.
* **System Libraries (Linux only)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 poppler-utils
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/jai050803/POD_OCR_flask_based.git
cd POD_OCR_flask_based
```

### 3. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Web Interface (Flask)
Start the local server:
```bash
python app.py
```
Open your browser at:
👉 **`http://localhost:5000`**

The web dashboard allows you to drag and drop PDFs or images and view:
* **Tab 1: Identified Texts (Pure JSON)** — Clean array of text lines with a 1-click copy button.
* **Tab 2: Structured Output** — Cards for each extracted logistics field with confidence and review badges.
* **Tab 3: Document Quality** — Quality tier, mean confidence %, stamp analysis, and system recommendation.
* **Tab 4: Full OCR JSON** — Complete diagnostic payload.
* **Tab 5: OCR Blocks** — Tabular view of all detected blocks with confidence bars and recognition engine source (`PADDLE_CRNN` vs `TROCR_FALLBACK`).

---

### Option B: CLI Pipeline Test
You can run the end-to-end pipeline directly on any document from the command line:

```bash
python test_pipeline.py sample_pod.pdf
```

**Output Generated**:
* Prints extracted text lines one-by-one.
* Displays structured field values and confidence ratings.
* Displays document quality assessment and recommendations.
* Saves `texts_sample_pod.json` *(pure text array only)*.
* Saves `output_sample_pod.pdf.json` *(full pipeline payload)*.

---

## REST API Reference

### 1. Health Check
```http
GET /health
```
**Response**:
```json
{
  "status": "running",
  "paddle": true,
  "trocr": true,
  "pdf2image": true
}
```

---

### 2. Extract Document Data
```http
POST /extract
Content-Type: multipart/form-data
```

#### Parameters:
* `file` *(required)*: Binary file upload (`.pdf`, `.jpg`, `.jpeg`, `.png`, `.webp`).
* `pure` *(optional query parameter, e.g. `/extract?pure=true`)*: Returns *only* the raw JSON array of identified text strings.

#### Example 1 — Standard Request:
```bash
curl -X POST http://localhost:5000/extract \
     -F "file=@sample_pod.pdf"
```

**Response Structure (`200 OK`)**:
```json
{
  "extracted_texts": [
    "TAX INVOICE / DELIVERY CHALLAN",
    "ABC LOGISTICS PRIVATE LIMITED",
    "Consignee: Global Enterprises Ltd",
    "Vehicle No: MH 12 AB 1234",
    "Invoice No: INV-2024-9841",
    "Date: 12/10/2026",
    "Total Qty: 450 BAGS"
  ],
  "status": "success",
  "structured_fields": {
    "vehicle_number": {
      "value": "MH12AB1234",
      "confidence": "HIGH",
      "method": "NEAR_LABEL",
      "needs_review": false
    },
    "pod_number": {
      "value": "INV-2024-9841",
      "confidence": "HIGH",
      "method": "NEAR_LABEL",
      "needs_review": false
    },
    "delivery_date": {
      "value": "12/10/2026",
      "confidence": "HIGH",
      "method": "REGEX_DATE",
      "needs_review": false
    },
    "customer_name": {
      "value": "Global Enterprises Ltd",
      "confidence": "MEDIUM",
      "method": "NEAR_LABEL",
      "needs_review": false
    },
    "receiver_name": {
      "value": null,
      "confidence": null,
      "method": "NOT_FOUND",
      "needs_review": true
    },
    "quantity": {
      "value": "450",
      "confidence": "HIGH",
      "method": "REGEX",
      "needs_review": false
    },
    "remarks": {
      "value": null,
      "confidence": "HIGH",
      "method": "NO_EXCEPTIONS_FOUND",
      "needs_review": false,
      "exception_keywords": []
    }
  },
  "document_quality": {
    "mean_confidence": 0.892,
    "quality_tier": "GOOD",
    "recommendation": "Legible delivery document. Ready for automated ERP ingestion.",
    "low_confidence_blocks": 2,
    "trocr_fallback_count": 1,
    "rotation_corrected": false,
    "rotation_angle": 0,
    "stamp_detected": true,
    "stamp_count": 1
  },
  "document_classification": {
    "is_delivery_document": true,
    "matched_keywords": ["delivery", "consignee", "vehicle", "invoice", "quantity"],
    "matched_count": 5
  },
  "full_ocr": {
    "raw_text": "...",
    "total_blocks": 38,
    "mean_confidence": 0.892,
    "ocr_engine": "PaddleOCR+TrOCR"
  }
}
```

#### Example 2 — Pure Text Only:
```bash
curl -X POST "http://localhost:5000/extract?pure=true" \
     -F "file=@sample_pod.pdf"
```

**Response (`200 OK`)**:
```json
[
  "TAX INVOICE / DELIVERY CHALLAN",
  "ABC LOGISTICS PRIVATE LIMITED",
  "Consignee: Global Enterprises Ltd",
  "Vehicle No: MH 12 AB 1234",
  "Invoice No: INV-2024-9841",
  "Date: 12/10/2026",
  "Total Qty: 450 BAGS"
]
```

---

## Production Deployment (WSGI / Gunicorn)

For production deployment on Linux servers, run via **Gunicorn** using single-worker multithreading (to preserve model memory singletons):

```bash
gunicorn --bind 0.0.0.0:5000 \
         --workers 1 \
         --threads 4 \
         --timeout 120 \
         app:app
```

---

## License
Distributed under the MIT License.
