import cv2
import os
import glob

os.makedirs("dataset/staging", exist_ok=True)
image_files = glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png")

count = 0
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    ROWS, COLS = 5, 6
    cell_w = w // COLS
    cell_h = h // ROWS

    for r in range(ROWS):
        for c in range(COLS):
            cx1 = x + (c * cell_w) + int(cell_w * 0.15)
            cy1 = y + (r * cell_h) + int(cell_h * 0.15)
            cx2 = cx1 + int(cell_w * 0.7)
            cy2 = cy1 + int(cell_h * 0.7)

            cell_crop = image[cy1:cy2, cx1:cx2]
            if cell_crop.size == 0:
                continue

            crop_filename = f"dataset/staging/img_{os.path.basename(img_path).split('.')[0]}_r{r}_c{c}.jpg"
            cv2.imwrite(crop_filename, cell_crop)
            count += 1

print(f"SUCCESS: Automatically sliced total cell crops inside 'dataset/staging'! Total: {count}")
