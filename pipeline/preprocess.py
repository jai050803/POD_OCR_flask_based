"""POD Document OCR Preprocessing Pipeline.

This module prepares document images (such as carbon copies and stamped POD slips)
for high-accuracy OCR by performing rotation correction, resolution capping,
gray-world color balancing, CLAHE contrast enhancement, and adaptive thresholding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import PIL.Image
from PIL import Image

# Global cached orientation classifier instance to avoid reloading weights
_ORIENTATION_CLASSIFIER: Optional[Any] = None


def get_orientation_classifier() -> Optional[Any]:
    """Lazy-load and cache the PaddleOCR document orientation classifier."""
    global _ORIENTATION_CLASSIFIER
    if _ORIENTATION_CLASSIFIER is None:
        try:
            from paddleocr import DocImgOrientationClassification

            _ORIENTATION_CLASSIFIER = DocImgOrientationClassification()
        except Exception:
            try:
                from paddleocr import PaddleOCR

                _ORIENTATION_CLASSIFIER = PaddleOCR(use_doc_orientation_classify=True)
            except Exception:
                _ORIENTATION_CLASSIFIER = None
    return _ORIENTATION_CLASSIFIER


def correct_rotation(image: Image.Image) -> Tuple[Image.Image, bool, int]:
    """Step 0 — Rotation Correction:

    - Use PaddleOCR's angle classifier to detect document rotation.
    - If rotation is detected (90, 180, 270 degrees), correct it using
      PIL Image.rotate() with expand=True.
    - Must run BEFORE any other preprocessing step.
    """
    classifier = get_orientation_classifier()
    if classifier is None:
        return image, False, 0

    # Downscale a copy for orientation detection to prevent heavy multi-megapixel latency
    w, h = image.size
    max_side = max(w, h)
    if max_side > 1000:
        scale = 1000.0 / max_side
        thumb = image.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            resample=getattr(Image, "Resampling", Image).BILINEAR,
        )
    else:
        thumb = image

    img_np = np.array(thumb.convert("RGB"))

    try:
        results = list(classifier.predict(img_np))
        if not results:
            return image, False, 0

        pred = results[0]
        label = None
        if isinstance(pred, dict):
            if "label_names" in pred and pred["label_names"]:
                label = pred["label_names"][0]
            elif "label" in pred:
                label = pred["label"]
        elif hasattr(pred, "label_names") and pred.label_names:
            label = pred.label_names[0]

        if label is not None:
            angle = int(label)
            if angle in (90, 180, 270):
                image = image.rotate(angle, expand=True)
                return image, True, angle
    except Exception:
        # If angle classification encounters an error, preserve original image
        pass

    return image, False, 0


def cap_resolution(image: Image.Image, max_edge: int = 1800) -> Image.Image:
    """Step 1 — Resolution Cap:

    - If the longest edge of the image exceeds 1800px, resize proportionally to fit within 1800px.
    - Use PIL Image.LANCZOS for resampling.
    """
    width, height = image.size
    longest_edge = max(width, height)

    if longest_edge > max_edge:
        scale = float(max_edge) / float(longest_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((new_width, new_height), resample=resample_filter)

    return image


def gray_world_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Step 2 — Gray-World Color Balancing:

    - Convert image to OpenCV LAB color space.
    - Compute mean of A and B channels.
    - Subtract the deviation from 128 scaled by luminance for each channel.
    - Convert back to BGR. This removes the yellow cast from carbon copy documents.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    mean_a = float(np.mean(a_channel))
    mean_b = float(np.mean(b_channel))

    dev_a = mean_a - 128.0
    dev_b = mean_b - 128.0

    luminance_scale = l_channel / 255.0

    a_balanced = a_channel - (dev_a * luminance_scale)
    b_balanced = b_channel - (dev_b * luminance_scale)

    lab[:, :, 0] = np.clip(l_channel, 0.0, 255.0)
    lab[:, :, 1] = np.clip(a_balanced, 0.0, 255.0)
    lab[:, :, 2] = np.clip(b_balanced, 0.0, 255.0)

    balanced_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return balanced_bgr


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Step 3 — CLAHE:

    - Convert image to grayscale.
    - Apply CLAHE with clipLimit=3.0 and tileGridSize=(8,8).
    - Contrast Limited Adaptive Histogram Equalization improves local contrast
      in handwritten regions that are suppressed by nearby dark stamps.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    return clahe_gray


def apply_adaptive_threshold(gray_img: np.ndarray) -> Image.Image:
    """Step 4 — Adaptive Thresholding:

    - Apply Gaussian adaptive thresholding (NOT Otsu).
    - blockSize=31, C=10.
    - Return as PIL Image (RGB mode, converted from grayscale).
    """
    thresh = cv2.adaptiveThreshold(
        gray_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(thresh_rgb)


def preprocess(
    image_input: Union[str, Path, Image.Image, np.ndarray, Any],
    return_meta: bool = True,
) -> Union[PIL.Image, Tuple[PIL.Image, Dict[str, Any]]]:
    """Preprocess a document image for OCR through a 5-step enhancement pipeline.

    Steps executed in order:
    1. Step 0: Rotation Correction using PaddleOCR angle classifier
    2. Step 1: Resolution Cap to 1800px longest edge using LANCZOS
    3. Step 2: Gray-World Color Balancing in LAB color space
    4. Step 3: CLAHE contrast equalization (clipLimit=3.0, tileGridSize=(8, 8))
    5. Step 4: Gaussian adaptive thresholding (blockSize=31, C=10)

    Args:
        image_input: File path (str or Path) or PIL Image (or numpy array).
        return_meta: If True, returns (preprocessed_image, preprocess_meta).
                     If False, returns preprocessed_image.

    Returns:
        preprocessed_image, preprocess_meta (if return_meta=True)
    """
    # Reject PDF path or inspect input type
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input).strip()
        if path_str.lower().endswith(".pdf"):
            raise ValueError(
                "PDF file input detected. Please convert PDF pages to images first (e.g. using pdf2image) before preprocessing."
            )
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")
        image = Image.open(path_str)
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        # Check if caller passed a PDF page object (from fitz, pypdfium2, pypdf, pdf2image, etc.)
        type_str = type(image_input).__name__.lower()
        module_str = getattr(type(image_input), "__module__", "").lower()
        if "pdf" in type_str or "pdf" in module_str or "page" in type_str:
            raise ValueError(
                "PDF page object detected. Please convert PDF pages to an image first (e.g. using pdf2image) before preprocessing."
            )
        raise ValueError(
            f"Unsupported input type '{type(image_input).__name__}'. Expected an image file path (str) or a PIL Image."
        )

    # Check for PDF image metadata if opened via Pillow PDF parser
    if getattr(image, "format", None) == "PDF":
        raise ValueError(
            "PDF file input detected. Please convert PDF pages to images first (e.g. using pdf2image) before preprocessing."
        )

    # Ensure RGB representation
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 0 — Rotation Correction (Runs before any other preprocessing step)
    image, rotation_corrected, rotation_angle = correct_rotation(image)

    # Step 1 — Resolution Cap (max longest edge 1800px)
    image = cap_resolution(image, max_edge=1800)

    # Convert PIL Image (RGB) to OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 2 — Gray-World Color Balancing
    balanced_bgr = gray_world_balance(img_bgr)

    # Step 3 — CLAHE
    clahe_gray = apply_clahe(balanced_bgr)

    # Step 4 — Adaptive Thresholding (returns PIL Image in RGB mode)
    final_image = apply_adaptive_threshold(clahe_gray)

    preprocess_meta = {
        "rotation_corrected": rotation_corrected,
        "rotation_angle": rotation_angle,
    }

    if return_meta:
        return final_image, preprocess_meta
    return final_image


def detect_and_mask_stamps(pil_image: Image.Image) -> Dict[str, Any]:
    """Isolate physical rubber stamps and ink from document content using HSV color analysis.

    1. Convert PIL Image to OpenCV BGR, then to HSV.
    2. Compute background hue:
       - Filter pixels with Saturation > 15 AND Value > 50 (excludes pure black/white pixels).
       - Compute statistical mode of Hue channel for these pixels (paper background_hue).
    3. Build Color Mask (stamps and colored ink):
       - Circular hue distance from background_hue: min(abs(h - bg), 180 - abs(h - bg)).
       - Select pixels where circular_hue_distance > 35 AND Value > 30.
    4. Build Dark Mask (black ink):
       - Select pixels where Value < 60 AND Saturation < 80.
    5. Find contours on combined mask (Color Mask OR Dark Mask) with area > 500px.
    6. Crop each stamp region from the original image with 10px padding.

    Returns:
        dict:
            stamp_crops: List[PIL.Image]
            stamp_mask: np.ndarray (binary mask of stamp regions)
            background_hue: int
            stamp_count: int
    """
    # Work on a copy to avoid mutating original
    orig_image = pil_image.convert("RGB")
    w_img, h_img = orig_image.size

    img_rgb = np.array(orig_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h_channel = img_hsv[:, :, 0].astype(np.int32)
    s_channel = img_hsv[:, :, 1].astype(np.int32)
    v_channel = img_hsv[:, :, 2].astype(np.int32)

    # 2. Compute background hue
    bg_filter = (s_channel > 15) & (v_channel > 50)
    filtered_h = h_channel[bg_filter]
    if filtered_h.size > 0:
        counts = np.bincount(filtered_h, minlength=180)
        background_hue = int(np.argmax(counts))
    else:
        background_hue = 0

    # 3. Build Color Mask (stamps and colored ink)
    diff = np.abs(h_channel - background_hue)
    circular_hue_distance = np.minimum(diff, 180 - diff)
    color_mask = (circular_hue_distance > 35) & (v_channel > 30)

    # 4. Build Dark Mask (black ink)
    dark_mask = (v_channel < 60) & (s_channel < 80)

    # 5. Combined Mask and Contours
    combined_mask = np.uint8((color_mask | dark_mask) * 255)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stamp_contours = [c for c in contours if cv2.contourArea(c) > 500]

    # Create clean binary mask of isolated stamp regions
    stamp_mask = np.zeros_like(combined_mask)
    if stamp_contours:
        cv2.drawContours(stamp_mask, stamp_contours, -1, 255, -1)

    # 6. Crop each stamp region from original image with 10px padding
    stamp_crops: List[Image.Image] = []
    for cnt in stamp_contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(w_img, x + w_box + 10)
        y2 = min(h_img, y + h_box + 10)
        crop = orig_image.crop((x1, y1, x2, y2))
        stamp_crops.append(crop)

    return {
        "stamp_crops": stamp_crops,
        "stamp_mask": stamp_mask,
        "background_hue": background_hue,
        "stamp_count": len(stamp_crops),
    }

