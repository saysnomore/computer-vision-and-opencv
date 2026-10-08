import cv2
import numpy as np

def measure_object(image_path, known_ref_mm=24.0):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width_px = rect[1][0]
    height_px = rect[1][1]
    
    pixels_per_mm = width_px / known_ref_mm
    print(f"Calibration Scale: {pixels_per_mm:.2f} pixels/mm")

    target_mm = 30.0
    tolerance_mm = 2.0
    measured_mm = width_px / pixels_per_mm

    diff = abs(measured_mm - target_mm)
    status = "PASS" if diff <= tolerance_mm else "DEFECTIVE"
    print(f"Measured: {measured_mm:.2f}mm | Status: {status}")

# Run test (ensure you have a test image named reference.jpg)
# measure_object("reference.jpg")
