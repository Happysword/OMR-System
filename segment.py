import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from commonfunctions import show_images
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def segment_staff(img):
    #Takes a Uint8 grey image and returns an array of images of staffs
    #Only works on Non-rotated images
    inverted_img = (255 - img) / 255
    
    #Dilate the image
    
    struct_element = cv.getStructuringElement(cv.MORPH_RECT,(80,6)) #Should check for better structuring elements
    dilated_img = cv.dilate(inverted_img,struct_element,iterations=5)
    # show_images([dilated_img]) #print dilated image

    #Calculate the Horizontal histogram 
    hist_hor = np.sum(dilated_img,axis=1)
    # plt.plot(hist_hor)
    # plt.hlines(threshold,-50,len(hist_hor)+50,colors="0.5",linestyles="dashed")
    # plt.show()


    #Calculate the threshold of Segmentation
    # Try every different values of thresholds and choose the one that has lowest sigma and wider average distance
    sorted_hist_hor = sorted(hist_hor)
    percentage_taken = 0.03
    iteration_values = []
    for iteration in range(1,31) :
        
        #Calculate the Threshold with a different ratio everytime
        current_max_len = int( len(sorted_hist_hor) * percentage_taken * iteration)
        averaged_partition = sorted_hist_hor[0:current_max_len] # Percentage Average Taken
        threshold = round(np.average(averaged_partition))

        #calculate the width of the segments using the current threshold
        start_cut = 0
        staff_width = []
        less_flag = 0
        for i in range(len(hist_hor)):
            if less_flag == 0 and hist_hor[i] < threshold:
                less_flag = 1
            elif less_flag == 1 and hist_hor[i] > threshold:
                start_cut = i
                less_flag = 2
            elif less_flag == 2 and hist_hor[i] < threshold:
                staff_width.append( i-start_cut )
                less_flag = 0
        
        # If there was atleast one segment we add them to be compared later 
        if len(staff_width) > 0:
            sigma = np.std(staff_width)
            average_width = np.average(staff_width)
            iteration_values.append( (sigma,average_width,len(staff_width),threshold) )
    
    #Criteria for a good minimum sigma
    all_sigmas = [itr[0]for itr in iteration_values]
    min_sigma = 3 if np.median(all_sigmas) <= 3 else np.median(all_sigmas)
    

    #Initial value and base case if no segments found
    best_iteration = (0,0,0,0)
    max_segments = 0

    #Comparing between the sigmas and average width found in the above loop to find the best threshold
    for iteration in iteration_values:
        if iteration[0] < min_sigma and iteration[2] > max_segments:
            max_segments = iteration[2]
            best_iteration = iteration
        elif iteration[0] < min_sigma and iteration[1] > best_iteration[1]:
            best_iteration = iteration

    #Find the boundaries with best threshold
    threshold = best_iteration[3]
    # plt.plot(hist_hor)
    # plt.hlines(threshold,-50,len(hist_hor)+50,colors="0.5",linestyles="dashed")
    # plt.show()

    start_cut = 0
    staff_indices = []
    less_flag = 0
    for i in range(len(hist_hor)):
        if less_flag == 0 and hist_hor[i] < threshold:
            less_flag = 1
        elif less_flag == 1 and hist_hor[i] > threshold:
            start_cut = i
            less_flag = 2
        elif less_flag == 2 and hist_hor[i] < threshold:
            staff_indices.append( (start_cut,i) )
            less_flag = 0
    
    #Segment the Optimal Images and return them
    segmented_staffs = []
    for i in staff_indices:
        segmented_staffs.append(img[i[0]:i[1]])

    return segmented_staffs



def segment_symbols(img, width = 16, height = 32):
    #Takes an image and returns an array of symbols with size (width x height)

    #Find the Contours and and sort them According to their x values
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boundingBoxes = [cv.boundingRect(c) for c in contours] 
    (sorted_contours,sorted_bounding_rect) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][0]))

    
    #Use the sorted_bounding_rect to merge the overlapping Symbols
    new_sorted_bounding_rect = []
    i = 0
    while i < len(sorted_bounding_rect):
        x_min = sorted_bounding_rect[i][0]
        x_max = sorted_bounding_rect[i][0] + sorted_bounding_rect[i][2]
        y_min = sorted_bounding_rect[i][1]
        y_max = sorted_bounding_rect[i][1] + sorted_bounding_rect[i][3]
        for j in range(i+1,len(sorted_bounding_rect)):
            if sorted_bounding_rect[j][0] <= x_max + 3:
                x_max = max(x_max,sorted_bounding_rect[j][0]+sorted_bounding_rect[j][2])
                y_max = max(y_max,sorted_bounding_rect[j][1]+sorted_bounding_rect[j][3])
                y_min = min(y_min,sorted_bounding_rect[j][1])
            else:
                i = j-1
                new_sorted_bounding_rect.append((x_min,x_max,y_min,y_max))
                break
        if i == len(sorted_bounding_rect)-1 and sorted_bounding_rect[i][0] > new_sorted_bounding_rect[-1][0] + new_sorted_bounding_rect[-1][2] + 3:
            new_sorted_bounding_rect.append((x_min,x_max,y_min,y_max))
            break
        i +=1
    #Segment every Symbol and Put it as an image in the contours array
    contours_array = []
    for (i,c) in enumerate(new_sorted_bounding_rect):
        x_min ,x_max,y_min,y_max = new_sorted_bounding_rect[i]
        if x_max-x_min > 10:  #Should find a way to get the size w to neglect
            cropped_contour= img[y_min:y_max, x_min:x_max]
            contours_array.append(cropped_contour)

    return contours_array
