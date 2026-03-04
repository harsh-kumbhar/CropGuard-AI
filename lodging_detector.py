import cv2
import numpy as np

# ── Load ─────────────────────────────────────────────────────
image = cv2.imread("test_images/lodged_crop.jpg")
image = cv2.resize(image, (640, 480))
h, w  = image.shape[:2]

# ============================================================
#  STEP 1 — Stem zone: lower 50% of image, central 60% width
#  Leaves spread wide → we narrow horizontally to avoid them.
#  The main stem runs through the vertical centre of the plant.
# ============================================================
x1_roi = int(w * 0.20)
x2_roi = int(w * 0.80)
y1_roi = int(h * 0.40)
y2_roi = int(h * 0.95)
stem_zone = image[y1_roi:y2_roi, x1_roi:x2_roi]
sz_h, sz_w = stem_zone.shape[:2]

# ============================================================
#  STEP 2 — Stem-specific color mask
#  Stems are darker, less-saturated green compared to bright leaves.
#  Also include brown/yellow for dry bent stems.
# ============================================================
hsv = cv2.cvtColor(stem_zone, cv2.COLOR_BGR2HSV)

# Green stems (lower brightness/saturation than leaves)
mask_green = cv2.inRange(hsv, np.array([30, 30, 30]),  np.array([85, 255, 180]))
# Brown/dry bent stems
mask_brown = cv2.inRange(hsv, np.array([10, 25, 30]),  np.array([30, 200, 180]))
mask_stem  = cv2.bitwise_or(mask_green, mask_brown)

k = np.ones((4, 4), np.uint8)
mask_stem = cv2.morphologyEx(mask_stem, cv2.MORPH_OPEN,  k)
mask_stem = cv2.morphologyEx(mask_stem, cv2.MORPH_CLOSE, k)

# ============================================================
#  STEP 3 — Find the CENTRAL VERTICAL STRIP where the stem lives
#  The main stem of a single plant is in the middle 40% of the ROI.
#  This removes side leaves entirely.
# ============================================================
strip_x1 = int(sz_w * 0.30)
strip_x2 = int(sz_w * 0.70)
stem_strip_mask = np.zeros_like(mask_stem)
stem_strip_mask[:, strip_x1:strip_x2] = mask_stem[:, strip_x1:strip_x2]

# ============================================================
#  STEP 4 — Edge detection on stem strip only
# ============================================================
gray         = cv2.cvtColor(stem_zone, cv2.COLOR_BGR2GRAY)
gray_masked  = cv2.bitwise_and(gray, gray, mask=stem_strip_mask)
gray_blur    = cv2.GaussianBlur(gray_masked, (3, 3), 0)
edges        = cv2.Canny(gray_blur, 40, 120)

# ============================================================
#  STEP 5 — Hough lines with strict minimum length
#  Short lines = leaf edges and noise. Long lines = stems.
# ============================================================
lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                         threshold=20,
                         minLineLength=40,   # only long lines = real stems
                         maxLineGap=15)

debug = stem_zone.copy()
# Show the central strip boundary
cv2.rectangle(debug, (strip_x1, 0), (strip_x2, sz_h), (255,255,0), 1)

if lines is None or len(lines) == 0:
    # Fallback: fit ellipse on largest contour in stem strip
    contours, _ = cv2.findContours(stem_strip_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if contours and len(contours[0]) >= 5:
        ell   = cv2.fitEllipse(contours[0])
        angle = ell[2]
        if angle > 90: angle = 180 - angle
        dominant_angle = angle
        angle_std      = 0
        n_lines        = 0
        cv2.ellipse(debug, ell, (0,255,255), 2)
        print("No lines found — using ellipse fallback")
    else:
        print("Cannot determine stem angle — no stem detected")
        exit()
else:
    # ── Weight long lines more; discard very short ones ──────
    all_angles, all_weights = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang > 90: ang = 180 - ang
        length = np.hypot(x2 - x1, y2 - y1)
        all_angles.append(ang)
        all_weights.append(length)
        color = (0, 255, 0) if ang > 50 else (0, 0, 255)
        cv2.line(debug, (x1,y1),(x2,y2), color, 2)

    dominant_angle = np.average(all_angles, weights=all_weights)
    angle_std      = np.std(all_angles)
    n_lines        = len(lines)

# ============================================================
#  STEP 6 — Decision
#  Healthy vertical stem ≥ 65°  (near 90° = truly upright)
#  Lodged stem           ≤ 45°  (leaning heavily)
#  Borderline            45–65°
# ============================================================
if dominant_angle >= 65:
    result    = "Healthy Crop (Vertical)"
    color_r   = (0, 200, 0)
    confidence = min(100, int((dominant_angle - 55) * 7))
elif dominant_angle <= 45:
    result    = "Lodged Crop Detected"
    color_r   = (0, 0, 255)
    confidence = min(100, int((55 - dominant_angle) * 7))
else:
    result    = "Borderline — Manual Check Advised"
    color_r   = (0, 165, 255)
    confidence = int(abs(dominant_angle - 55) * 5)

# ── Print ─────────────────────────────────────────────────────
print(f"\n--- Detection Results ---")
print(f"Dominant stem angle  : {round(dominant_angle, 2)} deg")
print(f"Angle std deviation  : {round(angle_std, 2)} deg")
print(f"Lines detected       : {n_lines}")
print(f"Result               : {result}  [confidence ~{confidence}%]")
print(f"-------------------------")

# ── Display ───────────────────────────────────────────────────
orig_disp = image.copy()
cv2.rectangle(orig_disp, (x1_roi, y1_roi), (x2_roi, y2_roi), (0,255,255), 2)
# also show the narrow stem strip on original
cv2.rectangle(orig_disp,
              (x1_roi + strip_x1, y1_roi),
              (x1_roi + strip_x2, y2_roi),
              (255,255,0), 1)
cv2.putText(orig_disp,
            f"Stem: {round(dominant_angle,1)} deg  |  {result}",
            (10, y1_roi - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_r, 2)

cv2.imshow("Original  (cyan=ROI  yellow=stem strip)", orig_disp)
cv2.imshow("Stem Strip Mask", stem_strip_mask)
cv2.imshow("Stem Lines  (green=vertical  red=horizontal)", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()