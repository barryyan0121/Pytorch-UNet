import cv2 as cv
import numpy as np

roi_w, roi_h = 300, 300
kernel = np.ones((3, 3), dtype=np.uint8)

img = cv.imread('output.jpg', 0)
# closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
color_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
contours, hierarchy = cv.findContours(img, 1, 2)


cnt = np.concatenate(contours)
# cv.drawContours(color_img, contours, -1, (0, 255, 0), 1)
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

x1, y1 = x + w // 4, y - roi_h // 2
cv.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 3)

y1 += h
y2 = y1 + roi_h
cv.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 3)

x1, y1 = x + 3 * w // 4, y - roi_h // 2
cv.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 3)

y1 += h
cv.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 3)

x1, y1 = x + w - roi_w // 2, y + h // 2 - roi_h // 2
cv.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 3)

cv.imshow("", color_img)
cv.waitKey(0)
cv.destroyAllWindows()
