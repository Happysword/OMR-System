from commonfunctions import *
import os
img = io.imread('./images/note.png', as_gray=True)
img = img >0.95

#Separate notes without line removal 
remove = img.shape[0] - img.sum(axis = 0)

img2 = np.copy(img)
img2[:,(remove < 11)] = 1
show_images([img, img2])



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
show_images([img, img3])
