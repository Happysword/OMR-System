import skimage
from commonfunctions import *
import numpy as np
import os
import Binarization as binarization
import cv2
from skimage.transform import hough_line, hough_line_peaks
import itertools 

img = cv2.imread('./Images/music3.PNG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#invert the image
gray_inv = cv2.bitwise_not(gray)
gray_inv_binarized = cv2.adaptiveThreshold(gray_inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

gray_inv_binarized = np.copy(gray_inv_binarized)

#Get the lines alone to measure space between lines (should get all staff info from other class)
horizontal = np.copy(gray_inv_binarized)
vertical = np.copy(gray_inv_binarized)

cols = horizontal.shape[1]
horizontal_size = cols // 30

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

#Apply RLE to get space height (should get all staff info from other class)
rle = list()
for y in range(0,horizontal.shape[1]):
    count = 1
    n = 0
    rle_in = list()
    for x in range(1,horizontal.shape[0]):
        if(horizontal[x][y] == horizontal[x-1][y]):
            count+=1
        else :
            rle_in.append(count)
            count = 1
    rle_in.append(count)
    rle.append(np.array(rle_in))
rle = np.array(rle)

rle = sorted(rle, key=len)
spaceWidth = rle[-5][2]

# Creating circle SE with size of note circle (still can't detect hallow circle) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(spaceWidth*0.9),int(spaceWidth*0.9)))    
erosion = cv2.erode(gray_inv_binarized,kernel,iterations = 1)
dilate = cv2.dilate(erosion,kernel,iterations = 1)


show_images([img,dilate])