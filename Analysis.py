import csv
import pandas
import os

from Arguments import *

def analytics():
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        pass
    
    if not os.path.exists("Correction.csv"):

        # Retrieve all csv files in datasets
        numFiles = []
        fileNames = os.listdir(DATASETS_DIR)
        for fileNames in fileNames:
            if fileNames.endswith(".csv"):
                numFiles.append(fileNames)

        # Retrieve labels from each file and append to new file
        df = pandas.DataFrame()
        for i in range(len(numFiles)):
            ds = pandas.read_csv(DATASETS_DIR+numFiles[i])
            y_true = ds.iloc[:,-1]
            #print(numFiles[i])
            y_true.name = numFiles[i]
            df[i] = y_true.to_frame()
            #print(df)

        df.to_csv("Correction.csv")

    else:
        print("Correction file already exists.")


if __name__ == '__main__':
    analytics()

