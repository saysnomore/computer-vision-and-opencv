import cv2
import numpy as np

def classical_grid_inspection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Blister outline not found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    ROWS, COLS = 5, 6
    cell_w = w // COLS
    cell_h = h // ROWS

    output = image.copy()
    print("Running classical variance check across 30 cavities...")

    for r in range(ROWS):
        for c in range(COLS):
            cx1 = x + (c * cell_w) + int(cell_w * 0.15)
            cy1 = y + (r * cell_h) + int(cell_h * 0.15)
            cx2 = cx1 + int(cell_w * 0.7)
            cy2 = cy1 + int(cell_h * 0.7)

            cell_crop = gray[cy1:cy2, cx1:cx2]
            if cell_crop.size == 0:
                continue

            variance = np.var(cell_crop)
            
            if variance > 1000:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(output, (cx1, cy1), (cx2, cy2), color, 2)

    cv2.imshow("Classical Variance Inspection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
