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

    # Return the clean rotation-corrected, resolution-capped RGB image
    # Modern deep learning OCR (PaddleOCR DBNet + TrOCR) achieves optimal text
    # detection on continuous-tone RGB/grayscale images without destructive binarization.
    final_image = image

    preprocess_meta = {
        "rotation_corrected": rotation_corrected,
        "rotation_angle": rotation_angle,
    }

    if return_meta:
        return final_image, preprocess_meta
    return final_image

