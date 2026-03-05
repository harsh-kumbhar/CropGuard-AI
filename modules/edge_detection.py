import cv2


def detect_edges(image):

    try:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, 50, 150)

        return edges

    except:
        return image