import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('segtest.PNG',0)
(thresh, bin_img) = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
canny = cv.Canny(img,0,200)
contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

c_array = []
for (i,c) in enumerate(contours):
    x,y,w,h= cv.boundingRect(c)
    
    if w > 10:
        cropped_contour= img[y:y+h, x:x+w]
        c_array.append(cropped_contour)

# Show images
for i in range(len(c_array)):
    cv.imshow('Image',c_array[i])
    cv.waitKey(0)
    cv.destroyAllWindows()

images = [img, bin_img]
for i in range(len(images)):
    plt.subplot(len(images),1,i+1),plt.imshow(images[i],'gray')
plt.show()
