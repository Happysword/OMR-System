import cv2
import Binarization as binarization
from commonfunctions import *
from skimage import exposure
from fix_orientation import fix_orientation
from staff import *
from segment import *
from NotesDetection import *

originalImage = cv2.imread('Images/music3.png',cv2.IMREAD_GRAYSCALE)

# Binarization Step

thre4 = binarization.AdaptiveThresholding(originalImage,3,21) # give good results

# Fixing Orientation Step (Fixing Rotation and Perspective and Crop)
fixed_orientation = fix_orientation(thre4)
# show_images([fixed_orientation])

segmented_staffs_array = segment_staff(thre4*255)

# Getting Staff features

Staffs = []
for segment in segmented_staffs_array:
    Staffs.append(Staff(np.uint8(segment)))
for i in Staffs:
    show_images([i.lines,i.notes],["Detected Lines", "Detected notes"])

symbols = []      
for staff in Staffs:
    temp = segment_symbols(staff.notes)
    symbols = symbols + temp
    show_images(temp)
    # print(segment_symbols(staff.notes))
    # print(staff.positions)
    # notePoints,notesNames = NotesPositions(thre4,staff.positions,staff.space)
    # print(notesNames)



