import cv2 as cv
import numpy as np
import os
list_of_files = os.listdir('im/')
img = cv.imread(f'im/{list_of_files[-1]}')
print(img.shape)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img = cv.medianBlur(img, 3)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.dilate(img, np.ones((1,2), np.uint8), iterations=1)
img = cv.erode(img, np.ones((2,1), np.uint8), iterations=1)
cv.imshow('img', img)
cv.imwrite('cleaned.png', img)
cv.waitKey(0)
cv.destroyAllWindows()