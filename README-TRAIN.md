# OMR Project Team 4
## &emsp;We have used <b>SVM</b> in the classifier with linear kernel as it's fast and very accurate in our case. 
<br>

# 1. How to train the classifier?
## &emsp;1. You have to create a folder of the dataset containing folders for each symbol as each folder name is the label of the symbol.
### &emsp;&emsp;- You can find our naming convention <a href="https://docs.google.com/spreadsheets/d/1XqSvS_hDt0-i6hHHee-wwU1KsJoMRWFWlnPQF_Axy30/edit?usp=sharing"> *here* </a>.
## &emsp;2. Each image should be binarized and inverted (symbols = white, background = black).
## &emsp;3. Then run features.py file as follow:
``````````````````
    python features.py [Dataset path]
``````````````````
<br>

# 2. The Dataset that we have used:
## &emsp;- We have generated and collected our own dataset.
## &emsp;- Our data set is a collection of printed and handwritten music notes.
## &emsp;- You can find it <a href="https://drive.google.com/drive/folders/1pKRwwhgKUzjncaZcCojwe84l8AA-QIM_?usp=sharing"> *here* </a>.

<br>

# 3. How much time does it take to train The classifier?
## &emsp;- For a dataset with size about 48K images (41 MB), It takes about 5-6 minutes.

<br>

# 4. The Hardware that we used for training:
## &emsp;- CPU: core i7-8750H
## &emsp;- RAM: 16 GB
