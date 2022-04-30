The file name is collaborative_filtering.py.  There are 3 total arguments that need to be entered into the 
command line.

1.) The file name.
2.) The path to the training text file.
3.) The path to the testing text file.

For example, in my case, I would enter:
python collaborative_filtering.py "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TrainingRatings.txt" "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TestingRatings.txt"

When you run the file, it will only process 1000 rows of testing data.  This is to save time.  The result printed at the end will only be for 
the 1000 rows which were processed.  TO adjust this threshold, change line 255 of the python file.