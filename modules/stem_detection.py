import cv2
import numpy as np


def detect_stems(image):

    debug = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 120)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=60,
        minLineLength=60,
        maxLineGap=10
    )

    angles = []

    if lines is None:
        return debug, angles

    for line in lines:

        x1,y1,x2,y2 = line[0]

        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

        if angle > 90:
            angle = 180 - angle

        angles.append(angle)

        cv2.line(debug, (x1,y1), (x2,y2), (0,255,0), 2)

    return debug, angles