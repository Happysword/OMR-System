from commonfunctions import *
import cv2
import os
img = io.imread('./images/note.png', as_gray=True)
img = img >0.95

#Separate notes without line removal 
remove = img.shape[0] - img.sum(axis = 0)

img2 = np.copy(img)
img2[:,(remove < 11)] = 1
# show_images([img, img2])



#Remove line Second method
rle = list()
for y in range(0,img.shape[1]):
    count = 1
    n = 0
    rle_in = list()
    for x in range(1,img.shape[0]):
        if(img[x][y] == img[x-1][y]):
            count+=1
        else :
            rle_in.append(count)
            count = 1
    rle_in.append(count)
    rle.append(np.array(rle_in))
rle = np.array(rle)
img3 = np.copy(img)
for i in range(img.shape[1]):
    for j in range(1,len(rle[i]),2):
        if rle[i][j] < 4:
            s = sum(rle[i][:j+1])
            img3[s - rle[i][j] - 1:s, i] = True
# show_images([img, img3])


img = cv2.imread('Images/music1.png',cv2.IMREAD_GRAYSCALE)
img_inv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8)
# print(img_inv)
img_inv = 1 - (img_inv // 255)
# img_inv = 1-img
# show_images([img_inv])

horizontal = np.uint8(np.copy(img_inv))
vertical = np.uint8(np.copy(img_inv))


 # Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = cols // 30

# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)


# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = 4 # should be rows // 30

# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

show_images([img_inv, horizontal, vertical],["Original","Staff lines","Symbols"])

#Enhance Output (NOT WORKING WELL)
'''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
'''
vertical = 1-vertical
final = np.copy(vertical)
# Step 1
# edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
edges = cv2.Canny(vertical,0.8,0.2)
show_images([edges])
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)
show_images([edges])
# Step 3
smooth = np.copy(vertical)
# Step 4
smooth = cv2.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]
# Show final result
show_images([final, vertical],["Before Enhancement","After Enhancement"])
