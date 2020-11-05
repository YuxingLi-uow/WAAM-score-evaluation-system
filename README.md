# WAAM-score-evaluation-system
A score system is used to extract current and voltage signals features in WAAM process, and then a segmented function is designed to evaluate welding score and provide information which indicating the existence of defects. 

# How to run
## GUI
1. Run labelRawdata.py
2. Click 'open file' button and select the current and voltage signal csv file. Some template data csv file could be found in '/RawData'. 
   Note: Make sure there is 5 rows head before your data. 
3. Check the feature extracted. 
4. Close the window by clicking the 'X' on right top corner if you don't want to the feature used to update your SGD prediction model. If this data is good enough to update the prediction model, follow step 5.
5. You can give labels to the bad range of data by both table and chart in GUI. 
6. Click 'Done' button on the right bottom and this will save the feature to the '/LabelData' and could be used for updating prediction model pretrained.

## Incremental learning model
Unfinished...

# Environment
python 3.6.5

# Contact
liyuxing1210@gmail.com or yl452@uowmail.edu.au 
Any advice and problem about this system is appreciated. 
