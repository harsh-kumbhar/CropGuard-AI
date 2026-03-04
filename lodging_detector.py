import cv2
import numpy as np
import math

# Load image
image = cv2.imread("test_images/images (14).jpg")

# Resize image
image = cv2.resize(image, (640,480))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blur = cv2.GaussianBlur(gray,(5,5),0)

# Edge detection
edges = cv2.Canny(blur,50,150)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=50,maxLineGap=10)

angles = []

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]

        angle = abs(math.degrees(math.atan2((y2-y1),(x2-x1))))
        angles.append(angle)

        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

vertical = 0
tilted = 0

for angle in angles:

    if 80 <= angle <= 100:
        vertical += 1
    else:
        tilted += 1

total = vertical + tilted

if total == 0:
    result = "No crop lines detected"
else:
    tilt_ratio = tilted / total

    if tilt_ratio > 0.4:
        result = "Lodged Crop Detected"
    else:
        result = "Crop Healthy"

print("Vertical lines:", vertical)
print("Tilted lines:", tilted)
print("Result:", result)

cv2.imshow("Detected Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()