import cv2
import numpy as np


def vegetation_mask(image):

    try:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    except:
        return image