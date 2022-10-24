import csv
import pandas
import os

from Arguments import *
from stgp.SimpleThresholdClassifier import SimpleThresholdClassifier
from stgp.Individual import Individual
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


def attribution():
    fileNames = os.listdir(OUTPUT_DIR)

    if len(fileNames) == 0:
        print("Output file does not exist.")
        exit()

    # Obtain final model of each run
    ds = pandas.read_csv(OUTPUT_DIR + fileNames[0])
    models = ds.loc[ds['Attribute'] == "Final_Model", '0']
    models = models.tolist()

    # Original dataset data
    main_ds = pandas.read_csv(DATASETS_DIR + DATASETS[0])
    label = main_ds.iloc[:, -1]  # Y original
    feat = main_ds.drop(columns=['Y'])  # Tr original

    # Retrieve all csv file names in datasets
    numFiles = []
    fileNames = os.listdir(DATASETS_DIR)
    for fileNames in fileNames:
        if fileNames.endswith(".csv"):
            numFiles.append(fileNames)

    model = Individual(OPERATORS, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'], 6)  # Static, Brasil
    #model = Individual(OPERATORS, ['B1', 'B2', 'B3'], 6)  # Static, Protein
    #model = Individual(OPERATORS, ['X0', 'X1'], 6)  # Static, Synthetic
    model.fitnessType = "Accuracy"

    if model.model_name == "SimpleThresholdClassifier":
        model.model = SimpleThresholdClassifier()

    # Store prediction results and print them
    results = []
    for i in range(len(models)):
        model.fromString(models[i])
        print(model)
        exit()
        pred = model.predict(feat)
        results.append(pred)
        #print(label.tolist())
        #acc = accuracy_score(label.tolist(), pred)
        #print("Accuracy: " + str(acc))

    # Join prediction values on one dataset
    df = pandas.DataFrame()
    for i in range(len(numFiles)-1):
        df['Model ' + str(i)] = results[i]

    if not os.path.exists("Predictions.csv"):

        # Calculate number of 1 and -1 predictions for each sample
        for (index, row) in df.iterrows():
            negatives = 0
            positives = 0
            for i in range(len(row)):
                if row[i] == -1:
                    negatives += 1
                elif row[i] == 1:
                    positives += 1
            if label[index] == 1:
                changed = negatives/(len(numFiles)-1)
            else:
                changed = positives/(len(numFiles)-1)
            df.loc[index, '1'] = positives
            df.loc[index, '-1'] = negatives
            df.loc[index, 'change%'] = changed
            df.loc[index, 'ID'] = index

        df.to_csv("Predictions.csv")

    else:
        print("Predictions file already exists.")


if __name__ == '__main__':
    attribution()
