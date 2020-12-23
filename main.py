import cv2
import Binarization as binarization
from commonfunctions import *
from skimage import exposure
from fix_orientation import fix_orientation
from staff import *
from segment import *

originalImage = cv2.imread('Images/music1.png',cv2.IMREAD_GRAYSCALE)

# Binarization Step

thre4 = binarization.AdaptiveThresholding(originalImage,3,21) # give good results

# Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
fixed_orientation = fix_orientation(thre4)
show_images([fixed_orientation])

segmented_staffs_array = segment_staff(fixed_orientation)

# Getting Staff features

Staffs = []
for segment in segmented_staffs_array:
    Staffs.append(Staff(np.uint8(segment)))
for i in Staffs:
    show_images([i.lines,i.notes],["Detected Lines", "Detected notes"])

symbols = []      
for staff in Staffs:
    symbols = symbols + segment_symbols(255-staff.notes)
    print(segment_symbols(staff.notes))

print(symbols)
for i in symbols:
    cv.imshow("Image",i)
    cv.waitKey(0)
