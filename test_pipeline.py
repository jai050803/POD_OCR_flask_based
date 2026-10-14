"""CLI test script to run the end-to-end POD OCR pipeline on a document file.

Usage:
    python test_pipeline.py <path_to_image_or_pdf>
"""

import json
import os
import sys
from pathlib import Path

# Ensure pod_ocr directory is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    input_file = Path(file_path)

    if not input_file.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Run full pipeline
    result = run_pipeline(str(input_file))
    total_pages = result.get("total_pages", 1)
    print(f"=== DOCUMENT SCAN: {total_pages} PAGE(S) PROCESSED ===")
    if total_pages > 1 and "pages" in result:
        for p in result["pages"]:
            print(f"  - Page {p['page_number']}: {len(p.get('extracted_texts', []))} lines, {p.get('total_blocks', 0)} blocks, conf: {((p.get('mean_confidence', 0))*100):.1f}%")

    # 1. Print Extracted Texts (one by one, no bboxes or extra metadata)
    print("\n=== EXTRACTED TEXTS (ONE BY ONE) ===")
    extracted_texts = result.get("extracted_texts", [])
    for idx, text in enumerate(extracted_texts, 1):
        print(f"{idx}. {text}")

    # 2. Print structured fields
    print("\n=== STRUCTURED FIELDS ===")
    structured_fields = result.get("structured_fields", {})
    for field_name, data in structured_fields.items():
        if isinstance(data, dict):
            val = data.get("value")
            conf = data.get("confidence")
            needs_rev = data.get("needs_review")
            page_info = f", page={data.get('found_on_page')}" if data.get('found_on_page') is not None else ""
            print(f"{field_name}: value={val}, confidence={conf}{page_info}, needs_review={needs_rev}")
        else:
            print(f"{field_name}: {data}")

    # 3. Print document quality
    print("\n=== DOCUMENT QUALITY ===")
    doc_quality = result.get("document_quality", {})
    print(f"quality_tier: {doc_quality.get('quality_tier')}")
    print(f"recommendation: {doc_quality.get('recommendation')}")
    print(f"rotation_corrected: {doc_quality.get('rotation_corrected')}")

    # 4. Print raw text (first 500 chars)
    print("\n=== RAW TEXT (first 500 chars) ===")
    raw_text = result.get("full_ocr", {}).get("raw_text", "")
    print(raw_text[:500])

    # 5. Save pure text list to texts_<filename>.json (ONLY the texts extracted, nothing else)
    texts_filename = f"texts_{input_file.stem}.json"
    with open(texts_filename, "w", encoding="utf-8") as f:
        json.dump(extracted_texts, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved pure extracted texts JSON to {texts_filename}]")

    # 6. Save full JSON output to output_<filename>.json
    output_filename = f"output_{input_file.name}.json"
    output_path = Path(output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if input_file.suffix and input_file.stem != input_file.name:
        alt_output_path = Path(f"output_{input_file.stem}.json")
        with open(alt_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[Saved full pipeline output JSON to {output_filename}]")


if __name__ == "__main__":
    main()
