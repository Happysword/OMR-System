import cv2
import Binarization as binarization
from DisplayImages import *
from skimage import exposure

originalImage = cv2.imread('Images/music2.png',cv2.IMREAD_GRAYSCALE)

# Binarization Step

thre1 = binarization.AdaptiveThresholding(originalImage,0,11) # sometimes make holes in the note
thre2 = binarization.AdaptiveThresholding(originalImage,1,11)
thre3 = binarization.AdaptiveThresholding(originalImage,2,11)
thre4 = binarization.AdaptiveThresholding(originalImage,3,11) # give good results
thre5 = binarization.GlobalThresholding(originalImage)

show_images([thre1,thre2,thre3,thre4,thre5],['ADAPTIVE_MEAN','ADAPTIVE_GAUSSIAN','niblack','sauvola','Otsu'])