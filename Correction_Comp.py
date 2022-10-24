import csv
import pandas
import os

from Arguments import *

def analytics():
    
    if not os.path.exists("Analysis.csv"):

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
            y_true = None
            tr = None
            if numFiles[i][0].isdigit():
                y_true = ds.iloc[:,-2]
                tr = ds.iloc[:,-1]
            else:
                y_true = ds.iloc[:,-1]
            #df = df.append(y_true)
            #df = df.append(tr)
            df[numFiles[i]] = y_true.to_frame()
            if numFiles[i][0].isdigit():
                df["Tr" + str(i+1)] = tr.to_frame()

        # Calculate % of corrected labels
        mainFile = df.columns.get_loc(DATASETS[0])
        for (index, row) in df.iterrows():
            counter = 0
            numCols = 0
            i = 0
            j = 0
            while i < mainFile:
                if row[i+1] == 1:
                    numCols += 1
                i += 2
            while j < mainFile:
                if row[j] != row[mainFile]:
                    if row[j+1] == 1:
                        counter += 1
                #numCols += 1
                if numCols != 0:
                    change = counter/numCols
                else:
                    change = 0
                j += 2
            df.loc[index, '#changes'] = counter
            df.loc[index, 'potential_changes'] = numCols
            df.loc[index, 'change%'] = change

        df.to_csv("Analysis.csv")

    else:
        print("Analysis file already exists.")


if __name__ == '__main__':
    analytics()

