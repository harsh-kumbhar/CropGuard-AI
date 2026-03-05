import cv2
import numpy as np


def load_image(file):
    """
    Convert uploaded file to OpenCV image
    """
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        return img
    except:
        return None


def resize_image(img, width=640):
    """
    Standardize image size
    """
    try:
        h, w = img.shape[:2]

        scale = width / w
        new_h = int(h * scale)

        resized = cv2.resize(img, (width, new_h))

        return resized
    except:
        return img


def convert_gray(img):

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    except:
        return img