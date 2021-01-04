import cv2
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

def AdaptiveThresholding(img,method=0):

    w,h = img.shape[:2]
    blockSize = max(w,h)
    blockSize = int(blockSize *0.03)
    
    if blockSize % 2 == 0:
        blockSize += 1

    print("blockSize " + str(blockSize))

    if(method == 0):
        outputImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,8)
    elif(method == 1):
        outputImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,8)
    elif(method == 2):
        outputImg = img > threshold_niblack(img, blockSize, k=0.8)
    elif(method == 3):
        outputImg = img > threshold_sauvola(img, blockSize) 
    
    return outputImg

def GlobalThresholding(img):
    
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th2