import csv
import pandas
import os

from Arguments import *
from stgp.SimpleThresholdClassifier import SimpleThresholdClassifier
from stgp.Individual import Individual
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


def validation():
    fileNames = os.listdir(OUTPUT_DIR)

    if len(fileNames) == 0:
        print("Output file does not exist.")
        exit()

    # Obtain final model of each run
    ds = pandas.read_csv(OUTPUT_DIR + fileNames[0])
    models = ds.loc[ds['Attribute'] == "Final_Model", '0']
    models = models.tolist()

    # Obtain the number of corrections of each run
    columns = list(ds)
    columns.remove('Attribute')
    columns.remove('Run')
    columns.pop()

    totals = ds.loc[ds['Attribute'] == 'Num-Corrections', columns].sum(axis=1)
    totals = totals.tolist()

    # Obtain csv files of corrected datasets
    numFiles = []
    correctedFiles = os.listdir(DATASETS_DIR)
    for fileName in correctedFiles:
        if fileName.endswith(".csv") and fileName[0].isdigit():
            numFiles.append(fileName)

    # Original dataset data
    main_ds = pandas.read_csv(DATASETS_DIR + DATASETS[0])
    tag = main_ds.iloc[:, -1]  # Y original
    feat = main_ds.drop(columns=['Y'])  # Tr original

    # Corrected dataset data
    labels = []  # Y corrections
    trX_corrected = []  # Tr Features corrections
    trY_corrected = []  # Tr Y corrections
    teX_corrected = []  # Te Features corrections
    teY_corrected = []  # Te Y corrections

    for i in range(len(numFiles)):
        trX_df = pandas.DataFrame()
        trY_df = pandas.DataFrame()
        teX_df = pandas.DataFrame()
        teY_df = pandas.DataFrame()
        ds = pandas.read_csv(DATASETS_DIR + numFiles[i])
        y_true = ds.iloc[:, -2]  # Labels
        tr = ds.iloc[:, -1]  # Training or not
        # Non-labels
        tempdata = ds.drop(columns=['Y', 'Tr'])
        data = tempdata.iloc[:, 1:14]  # Static, correct to collect all values
        data2 = ds[['Y']]
        for j in range(len(y_true)):
            # Check if it's a training or test sample
            if tr[j] == 1:
                trX_df = trX_df.append(data.iloc[[j]])
                trY_df = trY_df.append(data2.iloc[[j]])
            if tr[j] == 0:
                teX_df = teX_df.append(data.iloc[[j]])
                teY_df = teY_df.append(data2.iloc[[j]])
        trX_corrected.append(trX_df)
        trY_corrected.append(trY_df)
        teX_corrected.append(teX_df)
        teY_corrected.append(teY_df)
        labels.append(y_true)

    model = Individual(OPERATORS, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'], 6)  # Static, Brasil
    #model = Individual(OPERATORS, ['X0', 'X1'], 6)  # Static, Synthetic
    model.fitnessType = "Accuracy"

    if model.model_name == "SimpleThresholdClassifier":
        model.model = SimpleThresholdClassifier()

    # Store results and print them
    results = []
    for i in range(len(models)):
        model.fromString(models[i])
        raw = model.predict(trX_corrected[i], classOutput=False)
        rawList = raw.tolist()
        pred = model.predict(trX_corrected[i])
        print("\nGT/Predictions")
        print(trY_corrected[i]['Y'].tolist())
        file = open('Raw Values.csv', 'w+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerow(rawList)
        acc = accuracy_score(trY_corrected[i]['Y'].tolist(), pred)
        print("Accuracy: " + str(acc))

    # Scatter plot of original dataset
    #main_ds.plot.scatter(x='X0', y='X1', s=50, c='Y', cmap='coolwarm', label='original')
    # Scatter plot of corrected datasets
    if len(numFiles) == len(totals) and 0:
        for i in range(len(numFiles)):
            df = pandas.DataFrame()
            ds = pandas.read_csv(DATASETS_DIR + numFiles[i])
            ds.plot.scatter(x='X0', y='X1', s=50, alpha=0.5, c='Y', cmap='coolwarm', label="corrected")
            if totals[i] != 0:
                ds.plot.scatter(x='X0', y='X1', s=50, alpha=0.5, c='Y', cmap='coolwarm', label="corrected")
        plt.show()
    else:
        print("Incorrect nº of files in datasets")

    return results

def writeToFile(data):
    file = open('Raw Values.csv', 'w+', newline='')
    print(data)

    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(data)

if __name__ == '__main__':
    validation()
