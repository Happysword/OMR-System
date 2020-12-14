import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.data import page
from skimage import exposure
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
img = cv.imread('book.PNG',0)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,21,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,21,2)

# th2 = cv.medianBlur(th2,3)
# th3 = cv.medianBlur(th3,3)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# binary_global = img > threshold_otsu(img)

# window_size = 11
# thresh_niblack = threshold_niblack(img, window_size=window_size, k=0.8)
# thresh_sauvola = threshold_sauvola(img, window_size=window_size)

# binary_niblack = img > thresh_niblack
# binary_sauvola = img > thresh_sauvola

# plt.figure(figsize=(8, 7))
# plt.subplot(2, 2, 1)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.title('Original')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.title('Global Threshold')
# plt.imshow(binary_global, cmap=plt.cm.gray)
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(binary_niblack, cmap=plt.cm.gray)
# plt.title('Niblack Threshold')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(binary_sauvola, cmap=plt.cm.gray)
# plt.title('Sauvola Threshold')
# plt.axis('off')

# plt.show()