<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Happysword/OMR-System">
    <img src="https://image.freepik.com/free-vector/note-music-logo-design_93835-645.jpg" alt="Logo" width="160" height="160">
  </a>

  <h3 align="center">OMR System</h3>

  <p align="center">
    A sheet music reader that converts sheet music to a machine-readable version! 
    <br />


## Methodology

### 1 - Fixing orientation and Skew and biniarization
#### Input Image
![alt text](https://i.ibb.co/2dzrbMp/21.jpg)
#### Fixed Rotation Image
![alt text](https://i.ibb.co/vPnzdBJ/Whats-App-Image-2021-01-10-at-10-15-17-PM.jpg)
#### Binarized Image
![alt text](https://i.ibb.co/L5gz07K/Whats-App-Image-2021-01-10-at-10-15-17-PM.jpg)

We Binarize the Image using Savoula with a block size relative to the size
of the image then we dilate it to get the bounding rectangle around the
image then we crop the image and we start to fix the orientation by rotating
using the angle from Hough line transform and we fix the skew by using
four-point perspective.

### 2 - Segmentation of Staffs , Detecting Staff Notes and Line Removal
#### Segmented Image
![alt text](https://i.ibb.co/QnjDFCS/Whats-App-Image-2021-01-10-at-10-15-17-PM.jpg)

We first dilate the image so that each staff is completely connected and
then we calculate the row histogram form which using an iterative way we
try different thresholds and calculate some parameters like the average
width and standard deviation of widths and then choose the parameters
that are best for the threshold and then we segment.

We calculate the thickness and space between lines using a column run
length and histogram then we detect the staff line Positions by using the
histogram and we remove them from the original image.

### 3 - Notes Positions Detection
#### Position of notes
![alt text](https://i.ibb.co/YBvpX8p/Whats-App-Image-2021-01-10-at-10-15-17-PM.jpg)

We use a structuring element that is in the shape of an ellipse using the
same height as the detected from the Lines, then to detect hollow notes
we use skeletonization and then filling the outer and opening to and then
we use the same structure element again.

### 4 - Notes Segmentation and Classifier Prediction

Lastly, we segment the notes using their bounding rectangle and pass
them to the Classifier to get their value and translate them to the
required output and write them to file



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

## Built With

* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Scikit-Image](https://scikit-image.org/)

