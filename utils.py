import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(file):
    """Convert uploaded file to OpenCV image"""
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            logger.error("Failed to decode image")
        return img
    except Exception as e:
        logger.error(f"load_image error: {e}")
        return None


def resize_image(img, width=800):
    """Standardize image size while preserving aspect ratio"""
    try:
        h, w = img.shape[:2]
        scale = width / w
        new_h = int(h * scale)
        resized = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized
    except Exception as e:
        logger.error(f"resize_image error: {e}")
        return img


def convert_gray(img):
    """Convert to grayscale"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    except Exception as e:
        logger.error(f"convert_gray error: {e}")
        return img


def compute_image_stats(image):
    """Compute general image statistics"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "megapixels": round((w * h) / 1_000_000, 2)
        }
    except Exception as e:
        logger.error(f"compute_image_stats error: {e}")
        return {}


def overlay_heatmap(image, mask):
    """Overlay a green heatmap on the original image"""
    try:
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask  # green channel
        blended = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        return blended
    except Exception as e:
        logger.error(f"overlay_heatmap error: {e}")
        return image