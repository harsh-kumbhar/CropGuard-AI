import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_stems(image):
    """
    Detect crop stems using Hough Line Transform with detailed metrics.
    Returns: (annotated_image, avg_angle, stats_dict)
    """
    debug = image.copy()

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        # Morphological operation to enhance vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

        lines = cv2.HoughLinesP(
            enhanced,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=60,
            maxLineGap=15
        )

        angles = []
        lengths = []
        vertical_count = 0
        tilted_count = 0

        if lines is None:
            return debug, None, {"line_count": 0, "avg_angle": None, "vertical_pct": 0, "tilt_variance": 0}

        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle > 90:
                angle = 180 - angle

            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angles.append(angle)
            lengths.append(length)

            # Color coding: green = vertical/healthy, red = tilted
            if angle > 60:
                color = (0, 255, 80)   # green
                vertical_count += 1
            elif angle > 35:
                color = (0, 200, 255)  # yellow
                tilted_count += 1
            else:
                color = (0, 60, 255)   # red
                tilted_count += 1

            thickness = max(1, int(length / 60))
            cv2.line(debug, (x1, y1), (x2, y2), color, thickness)

        if len(angles) == 0:
            return debug, None, {"line_count": 0, "avg_angle": None, "vertical_pct": 0, "tilt_variance": 0}

        angles_arr = np.array(angles)
        avg_angle = float(np.mean(angles_arr))
        tilt_variance = float(np.var(angles_arr))
        vertical_pct = round((vertical_count / len(angles)) * 100, 2)
        avg_length = round(float(np.mean(lengths)), 2)

        # Draw angle legend
        h, w = debug.shape[:2]
        cv2.putText(debug, f"Avg Angle: {avg_angle:.1f}°", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug, f"Lines: {len(angles)}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        stats = {
            "line_count": len(angles),
            "avg_angle": round(avg_angle, 2),
            "vertical_pct": vertical_pct,
            "tilt_variance": round(tilt_variance, 2),
            "avg_line_length": avg_length,
            "vertical_count": vertical_count,
            "tilted_count": tilted_count,
        }

        return debug, avg_angle, stats

    except Exception as e:
        logger.error(f"detect_stems error: {e}")
        return debug, None, {"line_count": 0, "avg_angle": None, "vertical_pct": 0, "tilt_variance": 0}