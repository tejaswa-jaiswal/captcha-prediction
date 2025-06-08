import cv2 as cv
import numpy as np
from keras.models import load_model
from PIL import Image

img = cv.imread('img.png')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img = cv.medianBlur(img, 3)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.dilate(img, np.ones((1,2), np.uint8), iterations=1)


cv.imwrite('cleaned.png', img)

cv.destroyAllWindows()

import cv2 as cv
import numpy as np

img = cv.imread('cleaned.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((2, 2), np.uint8)
thresh = cv.dilate(thresh, kernel, iterations=1)


contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Filter small contours
MIN_AREA = 100
filtered_boxes = []
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    if w * h > MIN_AREA:
        filtered_boxes.append((x, y, w, h))

# Sort left to right
bounding_boxes = sorted(filtered_boxes, key=lambda b: b[0])

# Save digits
for i, (x, y, w, h) in enumerate(bounding_boxes):
    digit = thresh[y:y+h, x:x+w]

    # Make square by padding
    h_pad, w_pad = digit.shape
    size = max(h_pad, w_pad)
    top = (size - h_pad) // 2
    bottom = size - h_pad - top
    left = (size - w_pad) // 2
    right = size - w_pad - left

    digit_padded = cv.copyMakeBorder(digit, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    

    # Resize to 28x28
    digit_resized = cv.resize(digit_padded, (28, 28), interpolation=cv.INTER_AREA)
    digit_resized = cv.erode(digit_resized, np.ones((2,2), np.uint8), iterations=1)
    
    
    
    
    cv.imwrite(f'digit_{i+1}.png', digit_resized)
    


cv.destroyAllWindows()

model = load_model('my_model.keras')

s = ""
for i in range(1, 6):
   
    img = Image.open(f'digit_{i}.png').convert('L').resize((28, 28))
    img_array = np.array(img)

    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0 
    img_array = img_array.reshape(1, 28, 28, 1)  
 
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)
    s += str(pred_label)

print("Predicted sequence:", s)