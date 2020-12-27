import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from commonfunctions import show_images
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


###EXPERIMENT RESULTS
# 1- Taking the derivatives leads me to some maxima and minima that need to be known but has the advantage of taking the exact image
# 2- Taking the original image gets the maximas quite easily after smoothing but still needs some sort of localizing the maximum and it also picks more than one point at times
# 3- Curve fitting is a no go as the curve would be fit too well and there is alot of parameters to consider
# 4- using wavelet transform find_peaks_cwt doesnt do anything
# 5- Taking multiple derivatives doesn't do anything as well


def func(x, a, b):
    return a * np.sin(b * x)

def segment_staff(img):
    #Takes a Uint8 grey image and returns an array of images of staffs
    #Only works on Non-rotated images
    inverted_img = (255 - img) / 255
    
    # #Sobel horizontal
    # sobely = cv.Sobel(inverted_img,cv.CV_64F,0,1,ksize=5) >= 1
    # show_images([sobely])

    #Dilate the image
    struct_element = cv.getStructuringElement(cv.MORPH_RECT,(7,7)) #Should check for better structuring elements
    dilated_img = cv.dilate(inverted_img,struct_element,iterations=5)
    show_images([dilated_img]) #print dilated image

    #Calculate the Horizontal histogram 
    hist_hor = np.sum(dilated_img,axis=1)
    plt.plot(hist_hor)
    plt.show()

    #Calculate the threshold of Segmentation
    sorted_hist_hor = sorted(hist_hor)
    averaged_partition = sorted_hist_hor[0:int(len(sorted_hist_hor)/3)] # Average on the first 33.3% -> to be improved
    threshold = round(np.average(averaged_partition))
    print(threshold)
   
    #Segment the images
    start_cut = 0
    segmented_staffs = []
    less_flag = 0
    for i in range(len(hist_hor)):
        if less_flag == 0 and hist_hor[i] < threshold:
            less_flag = 1
        elif less_flag == 1 and hist_hor[i] > threshold:
            start_cut = i
            less_flag = 2
        elif less_flag == 2 and hist_hor[i] < threshold:
            segmented_staffs.append(img[start_cut:i])
            less_flag = 0
    
    return segmented_staffs

def segment_staff_2(img):
    inverted_img = (255 - img) / 255
    
    #Dilate the image
    struct_element = cv.getStructuringElement(cv.MORPH_RECT,(5,5)) #Should check for better structuring elements
    dilated_img = cv.dilate(inverted_img,struct_element,iterations=5)
    show_images([dilated_img])

    #Calculate the Horizontal histogram 
    hist_hor = np.sum(dilated_img,axis=1)
    plt.plot(hist_hor)
    plt.show()
    
    # #Derivative of the histogram
    # deriv = np.diff(hist_hor)
    # plt.plot(deriv)
    # plt.show() 

    # # #get the range of the filter set it to 3% of the size of the image
    # srange = int(len(hist_hor) * 0.03 / 2)
    # #Smoothing the curve as much as possible
    # for i in range(srange,len(hist_hor)-srange):
    #     hist_hor[i] = np.sum(hist_hor[i-srange:i+srange])/(srange*2)
    
    # #Derivative of the histogram
    # deriv = np.diff(hist_hor)
    # plt.plot(deriv)
    # plt.show() 

    # #Fit the Sine curve
    # xdata = range(0,len(deriv))
    # popt, pcov = curve_fit(func, xdata, deriv)
    # plt.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

    #Find Peaks
    deriv = hist_hor
    peaks, _ = find_peaks(deriv, height=0)
    negpeaks, _ = find_peaks(deriv*-1,height=0)
    plt.plot(deriv)
    plt.plot(peaks, deriv[peaks], "x")
    plt.plot(negpeaks, deriv[negpeaks], "o")
    plt.show() 

    return

def segment_symbols(img, width = 16, height = 32):
    #Takes an image and returns an array of symbols with size (width x height)
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0]) # TO be optimized

    rectangles = []
    for (i,obj) in enumerate(sorted_contours):
        rectangles.append(cv.boundingRect(obj))

    new_rectangles = []
    i = 0
    while i < len(rectangles)-1:
        x_min = rectangles[i][0]
        x_max = rectangles[i][0] + rectangles[i][2]
        for j in range(i+1,len(rectangles)):
            if rectangles[j][0] <= x_max + 3:
                x_max = max(x_max,rectangles[j][0]+rectangles[j][2])
            else:
                i = j-1
                new_rectangles.append((x_min,x_max))
                break
        i +=1

    contours_array = []
    for (i,c) in enumerate(new_rectangles):
        x_min ,x_max = new_rectangles[i]
        
        if x_max-x_min > 10:  #Should find a way to get the size w to neglect
            cropped_contour= img[:, x_min:x_max]
            contours_array.append(cropped_contour)
    return contours_array


### Test of segment_staff
# img = cv.imread("Images\music3.png",0)
# c_array = segment_staff(img)
# show_images(c_array)

#### Test of segment_symbols
# img = cv.imread("Images\segtest.png",0)

# c_array = segment_symbols(img)
# # Show images
# show_images(c_array)

# cv.imshow("img",img)
# cv.waitKey(0)