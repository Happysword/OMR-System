from debug_utils import *
import cv2
import os
import operator

class Staff:
    def __init__(self,image):
        self.image = image
        self.thickness = 0
        self.space = 0
        self.positions = np.zeros(7, dtype=np.int16)
        self.lines = np.zeros(self.image.shape, dtype=np.uint8)
        self.notes = np.zeros(self.image.shape, dtype=np.uint8)
        self.__get_staff_specs()
        self.__get_staff_positions()

    # Get staff specifications like thickness, space then extract the music notes from the image (Remove staff lines)
    def __get_staff_specs(self):
        img = np.uint8(self.image)

        if np.max(img) == 255:
            img = img // 255
        
        img_inv = 1 - img

        horizontal = np.uint8(np.copy(img_inv))

        # Calculate Run length of each column then take the most frequently number of white to be line thickness 
        # and most frequently number of black to be the space between lines
        rle = list()
        rle_total = list()
        for y in range(horizontal.shape[1]):
            count = 1
            rle_in = list()
            for x in range(1, horizontal.shape[0]):
                if(horizontal[x][y] == horizontal[x-1][y]):
                    count+=1  
                else :
                    rle_in.append(count)
                    count = 1
            rle_in.append(count)
            if len(rle_in) > 8:
                rle.append(np.array(rle_in))
            rle_total.append(np.array(rle_in))

        rle = np.array(rle)
        rle_total = np.array(rle_total)

        height = 0
        space = 0
        
        for i, col in enumerate(rle):
            line_freq = dict()
            space_freq = dict()
            start = 0
            
            if horizontal[0][i] == 0:
                start = 1
            
            for x in range(start, len(col), 2):
                if col[x] not in line_freq.keys():
                    line_freq[col[x]] = 0
                line_freq[col[x]] += 1

            for x in range(1 - start, len(col),2):
                if col[x] not in space_freq.keys():
                    space_freq[col[x]] = 0
                space_freq[col[x]] += 1
            
            height += max(line_freq.items(), key=operator.itemgetter(1))[0]
            space += max(space_freq.items(), key=operator.itemgetter(1))[0]

        if(len(rle)):
            self.thickness = int(np.round(height / len(rle)))
            self.space = int(np.round(space / len(rle)))
        else:
            self.thickness = 1
            self.space = (horizontal.shape[0] // 100) + 2
        
        Tlen = min(2*self.thickness, self.thickness+self.space)

        # Remove music notes from the original image which results in staff lines image
        for i, col in enumerate(rle_total):
            start = 0
            if horizontal[0][i] == 0:
                start = 1
            for x in range(start, len(col), 2):
                if col[x] >= Tlen:
                    horizontal[sum(col[:x]):sum(col[:x+1]), i] = 0

        self.lines = horizontal

        # Extract music notes by subtracting the staff lines image from the original image
        self.notes = img_inv - self.lines
        self.notes *= 255
   
    # Get staff lines positions by calculating horizontal histogram and take values greater than (image width / 4)   
    def __get_staff_positions(self):
        
        lineVotes = np.zeros(self.lines.shape[0])
        for x in range(self.lines.shape[0]):
            if sum(self.lines[x]) >= (self.lines.shape[1] // 4):
                lineVotes[x] = 1
                
        i = 1
        x = 0
        while x  < len(lineVotes):
            if(lineVotes[x] == 1 and i < 6):
                self.positions[i] = x + (self.thickness // 2)
                i+=1
                x += self.space // 2
            x += 1

        for i in [2,3,4,5]:
            if(self.positions[i] == 0 or self.positions[i]-self.positions[i-1] > 2*self.space):
                self.positions[i] = self.positions[i-1] + (self.space+self.thickness)

        self.positions[self.positions < 0] = 0
        self.positions[self.positions > self.lines.shape[0]-1] = self.lines.shape[0]-1
        self.positions[0] = max(self.positions[1] - (self.space+self.thickness), 0)
        self.positions[6] = min(self.positions[5] + (self.space+self.thickness), self.lines.shape[0]-1)

        debug_print(self.positions)
        self.lines[self.positions] = 1