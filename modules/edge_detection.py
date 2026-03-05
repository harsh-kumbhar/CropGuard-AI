import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_edges(image):
    """
    Multi-stage edge detection with stats.
    Returns: (edge_image_colored, stats_dict)
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Gaussian blur
        blur = cv2.GaussianBlur(denoised, (5, 5), 0)

        # Auto Canny thresholds using Otsu
        otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = int(max(0, 0.5 * otsu_thresh))
        high = int(min(255, 1.5 * otsu_thresh))

        edges = cv2.Canny(blur, low, high)

        # Dilate edges slightly for visibility
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Color the edges for better display
        edges_colored = cv2.applyColorMap(edges_dilated, cv2.COLORMAP_COOL)

        # Edge stats
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = int(np.sum(edges > 0))
        edge_density = round((edge_pixels / total_pixels) * 100, 2)

        # Laplacian sharpness score
        laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

        stats = {
            "edge_density_pct": edge_density,
            "edge_pixels": edge_pixels,
            "sharpness_score": laplacian_var,
            "canny_low": low,
            "canny_high": high,
        }

        return edges_colored, stats

    except Exception as e:
        logger.error(f"detect_edges error: {e}")
        return image.copy(), {"edge_density_pct": 0, "edge_pixels": 0, "sharpness_score": 0, "canny_low": 50, "canny_high": 150}