import cv2
import numpy as np

def measure_object_size(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(thresh, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("error: No contours found.")
        return

    # max contour
    largest_contour = max(contours, key=cv2.contourArea)

    # min rectangle
    rect = cv2.minAreaRect(largest_contour)
    (x, y), (w, h), angle = rect

    area = w*h

    print(f"size (in pixels):\nWidth = {w:.2f}\nHeight = {h:.2f}\narea = {area:.2f}")
    return area

measure_object_size("big.jpg")
