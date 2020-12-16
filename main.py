import cv2
import Binarization as binarization
from commonfunctions import *
from skimage import exposure

originalImage = cv2.imread('Images/music2.png',cv2.IMREAD_GRAYSCALE)

# Binarization Step

thre1 = binarization.AdaptiveThresholding(originalImage,0,21) # sometimes make holes in the note
thre2 = binarization.AdaptiveThresholding(originalImage,1,21)
thre3 = binarization.AdaptiveThresholding(originalImage,2,21)
thre4 = binarization.AdaptiveThresholding(originalImage,3,21) # give good results
thre5 = binarization.GlobalThresholding(originalImage)

show_images([thre1,thre2,thre3,thre4,thre5],['ADAPTIVE_MEAN','ADAPTIVE_GAUSSIAN','niblack','sauvola','Otsu'])

# titles = ['ADAPTIVE_MEAN','ADAPTIVE_GAUSSIAN','niblack','sauvola','Otsu']
# images = [thre1, thre2, thre3, thre4, thre5]
# for i in range(5):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()