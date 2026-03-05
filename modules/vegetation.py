import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def vegetation_mask(image):
    """
    Generate vegetation mask using HSV color thresholding.
    Returns: (mask, stats_dict)
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Broad green range
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Compute vegetation stats
        total_pixels = mask.shape[0] * mask.shape[1]
        vegetation_pixels = int(np.sum(mask > 0))
        coverage_pct = round((vegetation_pixels / total_pixels) * 100, 2)

        # Vegetation density map (normalized)
        blur_mask = cv2.GaussianBlur(mask, (21, 21), 0)
        density_colored = cv2.applyColorMap(blur_mask, cv2.COLORMAP_GREEN)

        # Contours for region count
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]

        stats = {
            "coverage_pct": coverage_pct,
            "vegetation_pixels": vegetation_pixels,
            "total_pixels": total_pixels,
            "region_count": len(significant_contours),
        }

        return mask, density_colored, stats

    except Exception as e:
        logger.error(f"vegetation_mask error: {e}")
        h, w = image.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, image.copy(), {"coverage_pct": 0, "vegetation_pixels": 0, "total_pixels": h*w, "region_count": 0}