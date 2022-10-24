import pandas

from stgp.STGP import STGP
from Arguments import *
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import csv

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#




def openAndSplitDatasets(which,seed):
	if VERBOSE:
		print( "> Opening: ", which )
	
	# Open dataset
	ds = pandas.read_csv(DATASETS_DIR+which)

	# Read header
	class_header = ds.columns[-1]

	return train_test_split(ds.drop(columns=[class_header]), ds[class_header], train_size=TRAIN_FRACTION, random_state=seed, stratify=ds[class_header])

def runNormal(r,dataset):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print()

	Tr_X, Te_X, Tr_Y, Te_Y = openAndSplitDatasets(dataset,r)

	# Extension of the dataset
	if SSUP:
		dt = []
		for index, row in Tr_X.iterrows():
			dt.append(row)
		for index, row in Tr_X.iterrows():
			dt.append(row)
		Tr_X = pandas.DataFrame(data=dt)

		ind = []
		val = []
		for index, value in Tr_Y.iteritems():
			ind.append(index)
			val.append(value)
		for index, value in Tr_Y.iteritems():
			ind.append(index)
			val.append(123)
		s = pandas.Series(val, index=ind)
		Tr_Y = s

	pandas.set_option('display.max_rows', None)
	pandas.set_option('display.max_columns', None)
	pandas.set_option('display.width', None)
	pandas.set_option('display.max_colwidth', None)

	# Train a model
	model = STGP(OPERATORS, MAX_DEPTH, POPULATION_SIZE, MAX_GENERATION, TOURNAMENT_SIZE, 
		ELITISM_SIZE, LIMIT_DEPTH, THREADS, VERBOSE)
	model.fit(Tr_X, Tr_Y, Te_X, Te_Y)


	# Obtain training results
	accuracy  = model.getAccuracyOverTime()
	rmse      = model.getRMSEOverTime()
	size      = model.getSizeOverTime()
	model_str = str(model.getBestIndividual())
	times     = model.getGenerationTimes()
	
	numCorr   = model.getNumCorrections()
	listCorr  = model.getListCorrections()

	tr_acc     = accuracy[0]
	te_acc     = accuracy[1]
	tr_rmse    = rmse[0]
	te_rmse    = rmse[1]


	if VERBOSE:
		print("> Ending run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print("  > Final model:", model_str)
		print("  > Training accuracy:", tr_acc[-1])
		print("  > Test accuracy:", te_acc[-1])
		print("  > Training RMSE:", tr_rmse[-1])
		print("  > Test RMSE:", te_rmse[-1])
		print()

	return (tr_acc,te_acc,
			tr_rmse,te_rmse,
			size, times, numCorr, listCorr,
			model_str)

def runKFold(r, dataset, split):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print()

	Tr_X, Te_X, Tr_Y, Te_Y = split

	# Extension of the dataset
	if SSUP:
		dt = []
		for index, row in Tr_X.iterrows():
			dt.append(row)
		for index, row in Tr_X.iterrows():
			dt.append(row)
		Tr_X = pandas.DataFrame(data=dt)

		ind = []
		val = []
		for index, value in Tr_Y.iteritems():
			ind.append(index)
			val.append(value)
		for index, value in Tr_Y.iteritems():
			ind.append(index)
			val.append(123)
		s = pandas.Series(val, index=ind)
		Tr_Y = s

	pandas.set_option('display.max_rows', None)
	pandas.set_option('display.max_columns', None)
	pandas.set_option('display.width', None)
	pandas.set_option('display.max_colwidth', None)

	# Train a model
	model = STGP(OPERATORS, MAX_DEPTH, POPULATION_SIZE, MAX_GENERATION, TOURNAMENT_SIZE,
		ELITISM_SIZE, LIMIT_DEPTH, THREADS, VERBOSE)
	model.fit(Tr_X, Tr_Y, Te_X, Te_Y)


	# Obtain training results
	accuracy  = model.getAccuracyOverTime()
	rmse      = model.getRMSEOverTime()
	size      = model.getSizeOverTime()
	model_str = str(model.getBestIndividual())
	times     = model.getGenerationTimes()

	numCorr   = model.getNumCorrections()
	listCorr  = model.getListCorrections()

	tr_acc     = accuracy[0]
	te_acc     = accuracy[1]
	tr_rmse    = rmse[0]
	te_rmse    = rmse[1]


	if VERBOSE:
		print("> Ending run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print("  > Final model:", model_str)
		print("  > Training accuracy:", tr_acc[-1])
		print("  > Test accuracy:", te_acc[-1])
		print("  > Training RMSE:", tr_rmse[-1])
		print("  > Test RMSE:", te_rmse[-1])
		print()

	return (tr_acc,te_acc,
			tr_rmse,te_rmse,
			size, times, numCorr, listCorr,
			model_str)

def callm3gp():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		outputFilename = OUTPUT_DIR+"stgp_"+dataset
		if not os.path.exists(outputFilename):
			results = []

			if not KFOLD:
				# Run the algorithm several times
				for r in range(RUNS):
					results.append(runNormal(r, dataset))
			else:
				# Apply 10-Fold data splitting to the dataset
				splits = getDataSplits(dataset, SPLITS)

				# Run the algorithm several times
				if RUNS != SPLITS:
					print("Error, number of runs different from number of data folds.")
					exit()
				else:
					for r in range(RUNS):
						results.append(runKFold(r, dataset, splits[r]))

			# Write output header
			file = open(outputFilename , "w")
			file.write("Attribute,Run,")
			for i in range(MAX_GENERATION):
				file.write(str(i)+",")
			file.write("\n")
		
			attributes = ["Training-Accuracy","Test-Accuracy",
						 "Training-RMSE", "Test-RMSE",
						 "Size", "Time", "Num-Corrections", "List-Corrections",
						 "Final_Model"]

			# Write attributes with value over time
			for ai in range(len(attributes)-1):
				for i in range(RUNS):
					if ai != len(attributes)-2:	
						file.write("\n"+attributes[ai]+","+str(i)+",")
						file.write( ",".join([str(val) for val in results[i][ai]]))
					else:
						writer = csv.writer(file)
						file.write("\n"+attributes[ai]+","+str(i)+",")
						writer.writerow([str(val) for val in results[i][ai]])

				file.write("\n")

			# Write the final models
			for i in range(len(results)):
				file.write("\n"+attributes[-1]+","+str(i)+",")
				file.write(results[i][-1])
			file.write("\n")

			# Write some parameters
			file.write("\n\nParameters")
			file.write("\nOperators,"+str(OPERATORS))
			file.write("\nMax Initial Depth,"+str(MAX_DEPTH))
			file.write("\nPopulation Size,"+str(POPULATION_SIZE))
			file.write("\nMax Generation,"+str(MAX_GENERATION))
			file.write("\nTournament Size,"+str(TOURNAMENT_SIZE))
			file.write("\nElitism Size,"+str(ELITISM_SIZE))
			file.write("\nDepth Limit,"+str(LIMIT_DEPTH))
			file.write("\nThreads,"+str(THREADS))


			file.close()
		else:
			print("Filename: " + outputFilename +" already exists.")

def getDataSplits(which, n):
	ds = pandas.read_csv(DATASETS_DIR+which)

	class_header = ds.columns[-1]
	x = ds.drop(columns=[class_header])
	y = ds[class_header]

	kf = KFold(n_splits=n, shuffle=True)

	splits = []
	for train_index, test_index in kf.split(x):
		temp = []
		x_train, x_test = x.iloc[train_index], x.iloc[test_index]
		temp.append(x_train)
		temp.append(x_test)
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		temp.append(y_train)
		temp.append(y_test)
		splits.append(temp)

	return splits

if __name__ == '__main__':
	callm3gp()
