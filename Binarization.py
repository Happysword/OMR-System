import cv2
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

def AdaptiveThresholding(img,method=0,block_size=11):
    
    if(method == 0):
        outputImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,8)
    elif(method == 1):
        outputImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,8)
    elif(method == 2):
        outputImg = img > threshold_niblack(img, block_size, k=0.8)
    elif(method == 3):
        outputImg = img > threshold_sauvola(img, block_size)         
    
    return outputImg

def GlobalThresholding(img):
    
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th2