import numpy as np
import logging
from modules.vegetation import vegetation_mask
from modules.edge_detection import detect_edges
from modules.stem_detection import detect_stems
from utils import compute_image_stats, overlay_heatmap

logger = logging.getLogger(__name__)


def compute_confidence(angle, veg_coverage, edge_density, line_count, tilt_variance):
    """Dynamically compute confidence score based on multiple signals"""
    score = 50.0

    # More lines = more reliable
    if line_count >= 20:
        score += 15
    elif line_count >= 10:
        score += 8
    elif line_count < 5:
        score -= 10

    # Good vegetation coverage helps
    if veg_coverage >= 30:
        score += 10
    elif veg_coverage < 10:
        score -= 8

    # Good edge density helps
    if 5 <= edge_density <= 30:
        score += 10
    elif edge_density < 2:
        score -= 10

    # Low tilt variance = more consistent = more confident
    if tilt_variance < 200:
        score += 10
    elif tilt_variance > 800:
        score -= 8

    # Clear angle boundaries boost confidence
    if angle > 75 or angle < 25:
        score += 5

    return int(min(max(score, 10), 98))


def analyze_crop(image):
    """
    Full crop lodging analysis pipeline.
    Returns all images, result label, suggestion, confidence, and detailed metrics.
    """
    # --- Run all modules ---
    veg_mask, veg_density, veg_stats = vegetation_mask(image)
    edges_colored, edge_stats = detect_edges(image)
    stem_img, angle, stem_stats = detect_stems(image)
    img_stats = compute_image_stats(image)
    veg_overlay = overlay_heatmap(image, veg_mask)

    # --- Handle no detection ---
    if angle is None:
        result = "Unable to Detect"
        suggestion = "No clear stem structures found. Try a closer, well-lit image with visible crop rows."
        confidence = 0
        all_metrics = {
            "image": img_stats,
            "vegetation": veg_stats,
            "edges": edge_stats,
            "stems": stem_stats,
            "result": {
                "label": result,
                "confidence": confidence,
                "avg_angle": None,
                "risk_score": None
            }
        }
        return veg_mask, veg_density, veg_overlay, edges_colored, stem_img, result, suggestion, confidence, all_metrics

    # --- Classification ---
    coverage = veg_stats.get("coverage_pct", 0)
    edge_density = edge_stats.get("edge_density_pct", 0)
    line_count = stem_stats.get("line_count", 0)
    tilt_variance = stem_stats.get("tilt_variance", 0)
    vertical_pct = stem_stats.get("vertical_pct", 0)

    confidence = compute_confidence(angle, coverage, edge_density, line_count, tilt_variance)

    if angle > 70:
        result = "✅ Healthy Crop"
        suggestion = (
            f"Crop stems are predominantly vertical (avg {angle:.1f}°). "
            f"{vertical_pct:.0f}% of detected stems are upright. "
            "No intervention required. Continue standard monitoring."
        )
        risk_score = max(0, 100 - int(angle))

    elif angle > 40:
        result = "⚠️ Moderate Lodging"
        suggestion = (
            f"Partial stem bending detected (avg {angle:.1f}°). "
            f"Only {vertical_pct:.0f}% of stems remain vertical. "
            "Increase field monitoring frequency. Consider soil aeration and wind protection measures."
        )
        risk_score = int((70 - angle) * 2.5 + 30)

    else:
        result = "🚨 Severe Lodging"
        suggestion = (
            f"Significant stem collapse detected (avg {angle:.1f}°). "
            f"Only {vertical_pct:.0f}% of stems are upright. "
            "Immediate action recommended: mechanical support, drainage improvement, and crop assessment for harvest viability."
        )
        risk_score = min(100, int((40 - angle) * 3 + 70))

    all_metrics = {
        "image": img_stats,
        "vegetation": veg_stats,
        "edges": edge_stats,
        "stems": stem_stats,
        "result": {
            "label": result,
            "confidence": confidence,
            "avg_angle": round(angle, 2),
            "risk_score": risk_score
        }
    }

    return veg_mask, veg_density, veg_overlay, edges_colored, stem_img, result, suggestion, confidence, all_metrics