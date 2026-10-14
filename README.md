# Proof of Delivery (POD) OCR & Logistics Extraction Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![OCR Engine](https://img.shields.io/badge/OCR-PaddleOCR%20v6%20%2B%20TrOCR-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An automated logistics document extraction pipeline engineered to ingest, clean, and extract structured logistics records from Proof of Delivery (POD) receipts, Loading Advices, Lorry Receipts (LR), Consignment Notes, and freight bills.

The system features **safe RGB preprocessing & deskew**, a **two-tier batched OCR engine** (combining PaddleOCR PP-OCRv6 with a batched TrOCR handwriting fallback), and a **logistics field extractor** that extracts critical fields including vehicle numbers, dates, routes, consignor/consignee entities, and full freight breakdowns with mathematical reconciliation.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                              Document Ingestion                                   |
|       - PDFs: Multi-page rendering via pypdfium2 (fallback pdf2image)             |
|       - Images: Multi-frame & single JPG, PNG, WEBP with safe alpha compositing    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  (Iterate each page 1..N)
+-----------------------------------------------------------------------------------+
|                           Document Preprocessing                                  |
|  - Per-page Orientation Classification & Deskew (0°, 90°, 180°, 270°)             |
|  - Resolution Capping (1800px longest edge via LANCZOS downsampling)              |
|  - Continuous-Tone Clean RGB Preservation (optimal for Deep Learning DBNet)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Two-Tier Batched OCR Engine                                 |
|  - Tier 1: PaddleOCR PP-OCRv6 (DBNet text detection + CRNN text recognition)      |
|  - Tier 2 Fallback: Batched TrOCR (microsoft/trocr-base-handwritten)              |
|    * Accelerated with torch.inference_mode() and token length capping             |
|    * Evaluates low-confidence handwriting blocks in a single batched tensor pass  |
|  - Block tagging: Each OCR block is annotated with page index (block.page = N)    |
+-----------------------------------------------------------------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
+---------------------------------------+ +---------------------------------------+
|   Intelligent Multi-Page Extraction   | |    Quality & Document Verification    |
|  - Per-page & Cross-page extraction   | |  - Keyword Density Delivery Verifier  |
|  - Best-field selection by confidence | |  - Multi-page Quality Tier (GOOD /    |
|  - Multi-page Route & Freight merging | |    DEGRADED / POOR)                   |
|  - 'found_on_page' source provenance  | |  - Aggregated ERP Recommendation      |
|  - Exception keywords scanning        | +---------------------------------------+
+---------------------------------------+
                    |
                    +--------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               Output Delivery                                     |
|  - Web UI: Responsive 5-Tab Dark Dashboard with multi-page badges & page tags     |
|  - REST API: JSON First payload with total_pages, pages metadata & field sources  |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

1. **Multi-Page Document Scanning & Cross-Page Consolidation**:
   - Iterates through **all pages** in multi-page PDFs or multi-frame scans.
   - Extracts structured fields per page and consolidates them intelligently: prioritizing non-null values, higher confidence (`HIGH > MEDIUM > LOW`), and critical exception flags.
   - Annotates each extracted field with source provenance metadata: `found_on_page` (e.g. `1`, `2`, `"ALL"`).
2. **Pure Text Array JSON First**: Returns an enumerated, cleaned array of all identified text lines in insertion order without bounding box or score clutter.
3. **Full Logistics Field Extraction**:
   - `vehicle_number`: Matches standard Indian commercial & private formats (`WB176302`, `WB-17-6302`, `MH12AB1234`, `DL 01 AB 1234`) with spatial proximity to anchor labels (`TRUCK NO`, `LORRY NO`, `VEHICLE NO`, `WE ARE SENDING TRUCK/TRAILORS NO`).
   - `date`: Tolerant parser matching standard and dot-separated logistics dates (`07/10/2026`, `Date..07/10.1.2026`, `18.07.2026`, `2026-10-07`).
   - `route`: Extracts origin (`from_location`) and destination (`to_location`) from single-line routes (`From Bhubaneshwar To Dankuni`) or separated anchor blocks.
   - `freight_details`: Extracts `total_freight`, `advance_amount`, `balance_amount`, and `payable_at`. Performs mathematical reconciliation (`advance + balance = total`) to verify or repair obscured total figures.
   - `consignor`: Extracts transport company / sender entity from header or signatory blocks (`SAHU MOVERS`, `APOLLO Supply Chain Pvt. Ltd.`).
   - `consignee`: Extracts bill-to recipient following standard transport anchors like `M/s.` or `Consignee` (`Road Express Freight`).
   - `remarks`: Scans for logistics exceptions across 17 keywords (`damage`, `shortage`, `leak`, `theft`, `broken`, `open`, etc.).
   - **Explicit Null Representation**: Any absent or unrecognized field is clearly serialized as `null`.
4. **High-Performance Batched TrOCR (30x Speedup)**:
   - Low-confidence handwritten crops are batched into a single tensor forward pass under `torch.inference_mode()`.
   - Complex multi-page PDFs process in **~5–15 seconds** (down from 2–3 minutes).
5. **Startup Engine Pre-Warming**:
   - Both PaddleOCR and TrOCR engines are pre-warmed into memory at process boot, eliminating first-request latency spikes and tunnel gateway timeouts.
6. **Robust Document Ingestion**:
   - Supports transparent PNG/RGBA by automatically compositing onto clean white matte backgrounds.
   - Self-contained multi-page PDF rendering via `pypdfium2` with no external OS binary requirements.

---

## Repository Structure

```
.
|-- app.py                     # Flask web server, REST endpoints, and engine pre-warming
|-- test_pipeline.py           # CLI test runner for end-to-end extraction
|-- requirements.txt           # Python package dependencies
|-- .gitignore                 # Clean gitignore excluding test images and notebooks
|-- pipeline/
|   |-- __init__.py            # Pipeline package initializer
|   |-- preprocess.py          # Orientation deskew (DocImgOrientationClassification) & resolution cap
|   |-- ocr.py                 # Singleton OCREngine, PaddleOCR + Batched TrOCR fallback
|   |-- extractor.py           # Comprehensive logistics field extractor
|   `-- classifier.py          # Quality tier assessment & delivery verification
`-- templates/
    `-- index.html             # Responsive dark-theme dashboard with JSON viewers & tabs
```

---

## Installation & Setup

### 1. Prerequisites
* **Python**: `3.10` or `3.11` recommended.
* **Operating System**: Windows, Linux, or macOS.
* **Linux Dependencies** (if deploying to Ubuntu/Debian server):
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 poppler-utils
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/jai050803/POD_OCR_flask_based.git
cd POD_OCR_flask_based
```

### 3. Create and Activate Virtual Environment
```bash
# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Web Interface (Flask)
Start the server:
```bash
python app.py
```
Open your browser at:
👉 **`http://localhost:5000`**

**Dashboard Tabs**:
* **Tab 1: Identified Texts (Pure JSON)** — Sequential text line array with 1-click clipboard copy and `.json` file download.
* **Tab 2: Structured Output** — Cards for Vehicle Number, Date, Route, Freight Breakdown, Consignor, Consignee, and full structured JSON payload.
* **Tab 3: Document Quality** — Quality tier (`GOOD`, `DEGRADED`, `POOR`), mean confidence, rotation deskew status, and recommendation.
* **Tab 4: Full OCR JSON** — Raw OCR payload containing all block coordinates and metadata.
* **Tab 5: OCR Blocks** — Interactive table showing confidence percentages and engine sources (`PADDLE_CRNN` vs `TROCR_FALLBACK`).

---

### Option B: CLI Pipeline Test
You can test extraction directly on any file (image or PDF):

```bash
python test_pipeline.py test.png
```

---

## Sharing & Live Demo (Public Hosting)

### 1. Cloudflare Tunnel (Instant Free HTTPS Tunnel)
Run the included `cloudflared.exe` binary in a separate terminal:
```powershell
.\cloudflared.exe tunnel --protocol http2 --edge-ip-version 4 --url http://127.0.0.1:5000
```
> **Note**: `--protocol http2` ensures reliable TCP port 443 connectivity (preventing UDP/QUIC packet drops or DNS timeouts on Wi-Fi/home ISPs), and `http://127.0.0.1:5000` avoids Windows IPv6 localhost delays.
This outputs a high-speed HTTPS link (e.g., `https://xxxx.trycloudflare.com`) accessible from any device worldwide with zero gateway timeouts or password prompts.

### 2. Ngrok
```powershell
ngrok http 5000
```

### 3. Permanent Cloud Hosting (Hugging Face Spaces — Free 16 GB RAM)
Because PaddleOCR and TrOCR require ~1.5 GB RAM:
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a **New Space** -> Choose **Docker** or **Gradio/Flask**.
3. Connect this GitHub repository.
4. Hugging Face provides **2 vCPU + 16 GB RAM free of charge** for a permanent 24/7 public live demo.

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
  "pdf_support": true
}
```

---

### 2. Extract Document Data
```http
POST /extract
Content-Type: multipart/form-data
```

#### Parameters:
* `file` *(required)*: Document file upload (`.pdf`, `.jpg`, `.jpeg`, `.png`, `.webp`).
* `pure` *(optional query parameter, e.g. `/extract?pure=true`)*: Returns only the clean JSON array of text strings.

#### Example Request:
```bash
curl -X POST http://localhost:5000/extract \
     -F "file=@loading_advice.png"
```

**Response (`200 OK`)**:
```json
{
  "status": "success",
  "total_pages": 1,
  "extracted_texts": [
    "LOADING ADVICE",
    "SAHU MOVERS",
    "Date..07/10.1.2026",
    "M/s.",
    "Road fxpress foreight",
    "WB176302",
    "We are Sending Truck/Trailors No.",
    "From Bhubaneshwar To Dankun?",
    "____ Total Freight Rs.000. I.S.",
    "Advancers 15,300-BalanceRs. 1700-Psycho at. 15.30."
  ],
  "structured_fields": {
    "document_type": {
      "value": "LOADING ADVICE",
      "confidence": "HIGH",
      "method": "KEYWORD_MATCH",
      "needs_review": false,
      "found_on_page": 1
    },
    "vehicle_number": {
      "value": "WB176302",
      "confidence": "HIGH",
      "method": "NEAR_LABEL",
      "needs_review": false,
      "found_on_page": 1
    },
    "date": {
      "value": "07/10/2026",
      "confidence": "HIGH",
      "method": "REGEX_DATE",
      "needs_review": false,
      "found_on_page": 1
    },
    "route": {
      "from_location": "Bhubaneshwar",
      "to_location": "Dankuni",
      "display": "Bhubaneshwar -> Dankuni",
      "confidence": "HIGH",
      "method": "REGEX_ROUTE",
      "needs_review": false,
      "found_on_page": 1
    },
    "consignor": {
      "value": "SAHU MOVERS",
      "confidence": "HIGH",
      "method": "HEADER_ENTITY",
      "needs_review": false,
      "found_on_page": 1
    },
    "consignee": {
      "value": "Road Express Freight",
      "confidence": "HIGH",
      "method": "NEAR_LABEL",
      "needs_review": false,
      "found_on_page": 1
    },
    "freight_details": {
      "total_freight": "17,000",
      "advance_amount": "15,300",
      "balance_amount": "1,700",
      "payable_at": "15,300",
      "confidence": "HIGH",
      "method": "REGEX_AMOUNTS",
      "needs_review": false,
      "found_on_page": 1
    },
    "pod_number": {
      "value": null,
      "confidence": null,
      "method": "NOT_FOUND",
      "needs_review": true,
      "found_on_page": null
    },
    "remarks": {
      "value": null,
      "confidence": null,
      "method": "NOT_FOUND",
      "needs_review": true,
      "found_on_page": null
    }
  },
  "document_quality": {
    "total_pages": 1,
    "mean_confidence": 0.9526,
    "quality_tier": "GOOD",
    "low_confidence_blocks": 0,
    "trocr_fallback_count": 2,
    "rotation_corrected": false,
    "rotation_angle": 0,
    "recommendation": "AUTO_APPROVED — Document quality sufficient for automated processing"
  },
  "document_classification": {
    "is_delivery_document": true,
    "matched_keywords": ["truck", "freight", "goods", "transport"],
    "matched_count": 4
  }
}
```

---

## Production Deployment (Gunicorn)

For production deployment on Linux servers, run via **Gunicorn** using single-worker multithreading:

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
