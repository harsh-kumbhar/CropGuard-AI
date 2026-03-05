from modules.vegetation import vegetation_mask
from modules.edge_detection import detect_edges
from modules.stem_detection import detect_stems


def analyze_crop(image):

    veg = vegetation_mask(image)

    edges = detect_edges(image)

    stem_img, angle = detect_stems(image)

    if angle is None:

        result = "Unable to detect crop structure"
        suggestion = "Try uploading a clearer image"

        return veg, edges, stem_img, result, suggestion, 0


    if angle > 70:

        result = "Healthy Crop"

        suggestion = "Crop stems are mostly vertical. No action required."

        confidence = 85


    elif angle > 40:

        result = "Moderate Lodging"

        suggestion = "Some bending detected. Monitor crop condition."

        confidence = 70


    else:

        result = "Severe Lodging Detected"

        suggestion = "Crop stems are significantly tilted. Consider support measures."

        confidence = 90


    return veg, edges, stem_img, result, suggestion, confidence