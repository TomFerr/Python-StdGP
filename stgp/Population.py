from .Individual import Individual
from .GeneticOperators import getElite, getOffspring, discardDeep
import multiprocessing as mp
import time

import pandas as pd
from Arguments import *
#from itertools import chain
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pickle
import os

#
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#

class Population:
	operators = None
	max_initial_depth = None
	population_size = None
	max_generation = None
	tournament_size = None
	elitism_size = None
	limit_depth = None
	verbose = None
	threads = None
	terminals = None


	population = None
	bestIndividual = None
	currentGeneration = 0

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingRMSEOverTime = None
	testRMSEOverTime = None
	sizeOverTime = None

	generationTimes = None

	errorNumbers = []
	errorRecords = []
	runCounter = []
	popPredictions = []
	bestInds = []
	bestIndPreds = []
	labeledSamplesY = []
	labeledSamplesX = []

	do_labelling = False
	do_correction = True

	def __init__(self, Tr_x, Tr_y, Te_x, Te_y, operators, max_initial_depth, population_size,
		max_generation, tournament_size, elitism_size, limit_depth, threads, verbose):

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y

		self.terminals = list(Tr_x.columns)
		self.operators = operators
		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size
		self.limit_depth = limit_depth
		self.threads = threads
		self.verbose = verbose

		self.population = []

		while len(self.population) < self.population_size:
			ind = Individual(self.operators, self.terminals, self.max_initial_depth)
			ind.create()
			self.population.append(ind)

		self.bestIndividual = self.population[0]
		self.bestIndividual.fit(self.Tr_x, self.Tr_y)

		# Obtain the Tr set elements without missing labels (for accuracy calculations)
		if self.currentGeneration == 0:
			trLen = len(self.Tr_x)
			self.labeledSamplesX = self.Tr_x.head(int(trLen/2))
			self.labeledSamplesY = self.Tr_y.head(int(trLen/2))
			#print(self.bestIndividual.getAccuracy(self.labeledSamplesX, self.labeledSamplesY))

		if not self.Te_x is None:
			self.trainingAccuracyOverTime = []
			self.testAccuracyOverTime = []
			self.trainingRMSEOverTime = []
			self.testRMSEOverTime = []
			self.sizeOverTime = []
			self.generationTimes = []


	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation

		# For accuracy fitness
		#perfectTraining = self.bestIndividual.getFitness() == 1
		# For rmse fitness
		perfectTraining = self.bestIndividual.getFitness() == 0

		if perfectTraining:
			print("---PERFECT TRAINING!---")

		return genLimit or perfectTraining

	def openPickle(self, file):
		'''
		Gets an old plot and shows it, can be edited (but no data values)
		'''
		fnames = os.listdir("pickle")
		for i in fnames:
			print(i)
		pickle.load(open("pickle/" + fnames[file], 'rb'))
		plt.show()

	def train(self):
		'''
		Training loop for the algorithm.
		'''
		if self.verbose:
			print("> Running log:")

		while self.currentGeneration < self.max_generation:
			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			else:
				duration = 0
			self.currentGeneration += 1

			if not self.Te_x is None:
				self.trainingAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))  # Tr_x/labeledSamplesX
				self.testAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
				self.trainingRMSEOverTime.append(self.bestIndividual.getRMSE(self.Tr_x, self.Tr_y, pred="Tr"))  # Tr_x/labeledSamplesX
				self.testRMSEOverTime.append(self.bestIndividual.getRMSE(self.Te_x, self.Te_y, pred="Te"))
				self.sizeOverTime.append(self.bestIndividual.getSize())
				self.generationTimes.append(duration)

		if self.verbose:
			print()

	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()

		# Calculates the accuracy of the population using multiprocessing
		if self.threads > 1:
			with mp.Pool(processes= self.threads) as pool:
				model = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y) for ind in self.population] )
				for i in range(len(self.population)):
					self.population[i].model = model[i][0].model
					self.population[i].labelToInt = model[i][0].labelToInt
					self.population[i].intToLabel = model[i][0].intToLabel
					self.population[i].trainingPredictions = model[i][1]
					self.population[i].training_X = self.Tr_x
					self.population[i].training_Y = self.Tr_y
		else:
			[ind.fit(self.Tr_x, self.Tr_y) for ind in self.population]
			[ind.getFitness() for ind in self.population]

		# Sort the population from best to worse
		self.population.sort(reverse=True)

		# Update best individual
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]

		# Reset the population variables
		if self.currentGeneration == 0:
			self.runCounter.append(0)
			self.errorNumbers = []
			self.errorRecords = []
			self.popPredictions = []
			self.bestInds = []
			self.bestIndPreds = []

		# ===== ATTRIBUTION =====

		if self.do_labelling:
			self.labelAttribution()

		# ===== CORRECTION =====

		errorCounter = 0
		errorRepeats = 0
		errorList = []
		if self.do_correction:
			# Choose one or more of the correction criteria
			critCorrections = []
			#critCorrections.append(self.convergenceCriteria())
			#critCorrections.append(self.labelCriteria())
			#critCorrections.append(self.outlierCriteria())

			# Do necessary corrections based on criteria
			if len(critCorrections) == 0:
				print("No criteria for correction selected!")
			else:
				# Check what samples passed all criteria
				ids = []
				ind = self.Tr_y.index.tolist()
				for i in ind:
					counter = 0
					for j in critCorrections:
						if i in j:
							counter += 1
					if counter == len(critCorrections):
						ids.append(i)

				# Correct the above sample labels
				if len(ids) == 0:
					print("No samples passed the selected criteria!")
				else:
					for i in ids:
						# Counting nº of corrections
						errorCounter += 1
						# Register id of corrected label
						errorList.append(i)
						# Correct the label
						if self.Tr_y.loc[i] == -1:
							self.Tr_y.loc[i] = 1
						elif self.Tr_y.loc[i] == 1:
							self.Tr_y.loc[i] = -1
					self.errorNumbers.append(errorCounter)
					self.errorRecords.append(errorList)

				# Count the number of repeated corrections
				if self.currentGeneration != 0 and len(self.errorRecords) > 1:
					for i in range(len(errorList)):
						if errorList[i] in self.errorRecords[-2]:
							errorRepeats += 1

			# Corrected Datasets
			if self.currentGeneration == self.max_generation-1:

				ds = pd.read_csv(DATASETS_DIR+DATASETS[0])
				labelUpdate = self.Tr_y.to_frame()
				ds["Tr"] = "0"
				for i in range(len(labelUpdate)):
					ds.at[labelUpdate.index[i], 'Y'] = labelUpdate.iat[i, 0]
					ds.at[labelUpdate.index[i], 'Tr'] = "1"
				ds.to_csv(DATASETS_DIR+str(len(self.runCounter)-1)+DATASETS[0])

		# === PLOTTING AND GRAPHS ===

		# All BestInd predictions
		p = self.bestIndividual.getTrainingValuePredictions()
		#self.popPredictions.append(p)

		# All Pop predictions
		for i in self.population:
			p = i.getTrainingValuePredictions()
			#self.popPredictions.append(p)

		# Choose one of the plots
		#self.showPopEvolution("violin", 10)  # Uncomment popPredictions if using this one, violin/scatter & numGens
		#self.lastGenPlot("scatter", "BestInd")  # violin/scatter & BestInd/Pop

		# Visual representation of the current dataset (update for Brazil)
		if self.currentGeneration == 0:

			main_ds = pd.read_csv(DATASETS_DIR+DATASETS[0])
			#main_ds.plot.scatter(x='X0', y='X1', s=50, c='Y', cmap='coolwarm')
			#plt.show()

		# === END ===

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend(getElite(self.population, self.elitism_size))
		while len(newPopulation) < self.population_size:
			offspring = getOffspring(self.population, self.tournament_size)
			offspring = discardDeep(offspring, self.limit_depth)
			newPopulation.extend(offspring)
		self.population = newPopulation[:self.population_size]

		end = time.time()

		# Debug
		if self.verbose and self.currentGeneration % 1 == 0:
			if not self.Te_x is None:
				# Tr_x/labeledSamplesX
				print("   > Gen #" + str(
					self.currentGeneration) + ":  Tr-Acc: " + "%.6f" % self.bestIndividual.getAccuracy(self.Tr_x,
																									   self.Tr_y) + " // Te-Acc: " + "%.6f" % self.bestIndividual.getAccuracy(
					self.Te_x, self.Te_y) + " // Tr-Rmse: " + "%.6f" % self.bestIndividual.getRMSE(self.Tr_x,
																								   self.Tr_y) + " // Te-Rmse: " + "%.6f" % self.bestIndividual.getRMSE(
					self.Te_x, self.Te_y) + " // Time: " + str(end - begin))
				if self.do_correction:
					print("   > IDs: " + str(errorList))
					print("   > Corrections: " + str(errorCounter))
					print("   > Repeats: " + str(errorRepeats) + "\n")
			else:
				print("   > Gen #" + str(
					self.currentGeneration) + ":  Tr-Acc: " + "%.6f" % self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y))

	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)

	def getBestIndividual(self):
		return self.bestIndividual

	def getCurrentGeneration(self):
		return self.currentGeneration

	def getTrainingAccuracyOverTime(self):
		return self.trainingAccuracyOverTime

	def getTestAccuracyOverTime(self):
		return self.testAccuracyOverTime

	def getTrainingRMSEOverTime(self):
		return self.trainingRMSEOverTime

	def getTestRMSEOverTime(self):
		return self.testRMSEOverTime

	def getSizeOverTime(self):
		return self.sizeOverTime

	def getGenerationTimes(self):
		return self.generationTimes

	def getNumCorrections(self):
		return self.errorNumbers

	def getListCorrections(self):
		return self.errorRecords

	def labelCriteria(self):
		'''
		Label criteria for label correction
		'''
		# Get predictions of all individuals
		preds = []
		predIds = []
		for i in range(len(self.population)):
			x = self.population[i].getTrainingClassPredictions()
			preds.append(x)
			predIds.append(i)

		# Get list of unique individuals
		uniquePreds = []
		uniquePredIds = []
		for x in range(len(preds)):
			if preds[x] not in uniquePreds:
				uniquePreds.append(preds[x])
				uniquePredIds.append(x)

		# Check if the dataset needs correcting
		labelErrors = [0] * len(self.Tr_y)
		for i in range(len(self.population)):
			if i in uniquePredIds:  # Check only unique individuals
				proximity = self.population[i].getProximity()  # Check proximity to nearest label
				certainty = self.population[i].getCertainty()  # Check distance around labels
				for j in range(len(labelErrors)):
					labelErrors[j] += proximity[j]

		# Register samples that need label correction
		corrections = []
		for i in range(len(labelErrors)):
			if labelErrors[i] > len(uniquePredIds) * 0.15:  # % of the pop needed for correction
				corrections.append(self.Tr_y.index[i])
		return corrections

	def outlierCriteria(self):
		'''
		Outlier criteria for label correction
		'''
		biPreds = self.bestIndividual.trainingValuePredictions
		ind = self.Tr_y.index.tolist()
		val = self.Tr_y.values.tolist()
		c0 = []
		c0ind = []
		c1 = []
		c1ind = []
		for index, i in enumerate(biPreds):
			if val[index] == -1:
				c0.append(i)
				c0ind.append(ind[index])
			if val[index] == 1:
				c1.append(i)
				c1ind.append(ind[index])
		corrections = []

		q3, q1 = np.percentile(c0, [75, 25])
		iqr = q3 - q1
		lowLimit = q1 - (1.5 * iqr)
		highLimit = q3 + (1.5 * iqr)
		for index, i in enumerate(c0):
			if i < lowLimit:
				corrections.append(c0ind[index])
			elif i > highLimit:
				corrections.append(c0ind[index])

		q3, q1 = np.percentile(c1, [75, 25])
		iqr = q3 - q1
		lowLimit = q1 - 1.5 * iqr
		highLimit = q3 + 1.5 * iqr
		for index, i in enumerate(c1):
			if i < lowLimit:
				corrections.append(c1ind[index])
			elif i > highLimit:
				corrections.append(c1ind[index])
		return corrections

	def convergenceCriteria(self):
		'''
		Convergence criteria for label correction (simplified)
		'''
		# Collect unique best individuals
		change = False
		if self.currentGeneration != 0 and len(self.bestInds) == 0:
			self.bestInds.append(self.bestIndividual)
			self.bestIndPreds.append(self.bestIndividual.getTrainingValuePredictions())
			change = True
		else:
			if len(self.bestInds) > 0:
				if self.bestIndividual != self.bestInds[-1]:
					self.bestInds.append(self.bestIndividual)
					self.bestIndPreds.append(self.bestIndividual.getTrainingValuePredictions())
					change = True

		# Check if we have at least 1 BestInd and calculate the median
		medians = []
		if len(self.bestInds) >= 1 and change:
			numPreds = len(self.bestIndPreds[0])
			counter = 0
			while counter < numPreds:
				temp = [item[counter] for item in self.bestIndPreds]
				m = statistics.median(temp)
				medians.append(m)
				counter += 1

		# Compare the BI's predictions using the medians
		corrections = []
		ind = self.Tr_y.index.tolist()
		for index, i in enumerate(medians):
			# Certainty the label should be -1 but isn't
			if i < 0:
				if self.Tr_y.iloc[index] != -1:
					corrections.append(ind[index])
			# Certainty the label should be 1 but isn't
			elif i >= 0:
				if self.Tr_y.iloc[index] != 1:
					corrections.append(ind[index])
		return corrections

	def showPopEvolution(self, type1, genNum):
		'''
		Show entire population predictions every X generations
		'''
		if self.popPredictions:
			if self.currentGeneration % genNum == 0 or self.currentGeneration == self.max_generation-1:

				ds_len = len(self.Tr_y) + len(self.Te_y)
				temp = self.bestIndividual.training_X
				biIds = list(temp.index.values)

				i = 0
				finalList = []
				labelIDs = []
				while i < ds_len*TRAIN_FRACTION:
					tempList = []
					tempIDs = []
					for j in self.popPredictions:
						tempList.append(j[i])
						tempIDs.append(biIds[i])
					i += 1
					labelIDs.append(tempIDs)
					finalList.append(tempList)

				fig, ax = plt.subplots()
				plt.title("Generation " + str(self.currentGeneration), fontsize=20)
				plt.xlabel('Sample ID', fontsize=20)
				plt.ylabel('Prediction Value', fontsize=20)
				plt.xlim(left=-2, right=42)
				plt.ylim(top=3, bottom=-3)
				plt.grid(True, axis='both')
				if type1 == "scatter":
					for i in range(len(labelIDs)):
						t = np.arange(len(labelIDs[i]))
						ax.scatter(labelIDs[i], finalList[i], marker='o', c=t, cmap="Greens")
						filename = 'Scatter_Run' + str(len(self.runCounter)-1) + "_Generation" + str(self.currentGeneration)
						pickle.dump(ax, open("pickle/"+str(filename), 'wb'))
				if type1 == "violin":
					f = plt.violinplot(finalList, biIds, widths=0.75, showmeans=True)
					filename = 'Violin_Run' + str(len(self.runCounter)-1) + "_Generation" + str(self.currentGeneration)
					pickle.dump(f, open("pickle/"+str(filename), 'wb'))
				plt.show()

	def lastGenPlot(self, type1, type2):
		'''
		Data plots for last generation
		'''
		if self.currentGeneration == self.max_generation-1:

			ds_len = len(self.Tr_y) + len(self.Te_y)
			temp = self.bestIndividual.training_X
			biIds = list(temp.index.values)

			# Last BestInd predictions
			if type2 == "BestInd":
				p = self.bestIndividual.getTrainingValuePredictions()
				self.popPredictions.append(p)

			# Last Pop predictions
			if type2 == "Pop":
				for i in self.population:
					p = i.getTrainingValuePredictions()
					self.popPredictions.append(p)

			# Prediction pre-processing for plots
			i = 0
			finalList = []
			labelIDs = []
			while i < ds_len*TRAIN_FRACTION:
				tempList = []
				tempIDs = []
				for j in self.popPredictions:
					tempList.append(j[i])
					tempIDs.append(biIds[i])
				i += 1
				labelIDs.append(tempIDs)
				finalList.append(tempList)

			sampleIDs = biIds #* self.max_generation #* self.population_size

			fig, ax = plt.subplots()
			plt.title('Model Predictions', fontsize=20)
			plt.xlabel('Sample ID', fontsize=20)
			plt.ylabel('Prediction Value', fontsize=20)
			plt.xlim(left=-2, right=42)
			plt.ylim(top=3, bottom=-3)
			plt.grid(True, axis='both')
			if type1 == "scatter":
				for i in range(len(labelIDs)):
					t = np.arange(len(labelIDs[i]))
					ax.scatter(labelIDs[i], finalList[i], marker='o', c=t)
					filename = 'Scatter_Run' + str(len(self.runCounter)-1) + "_Generation" + str(self.currentGeneration)
					pickle.dump(ax, open("pickle/"+str(filename), 'wb'))
				plt.show()
			if type1 == "violin":
				ax.violinplot(finalList, sampleIDs, widths=0.75, showmeans=True)
				filename = 'Violin_Run' + str(len(self.runCounter)-1) + "_Generation" + str(self.currentGeneration)
				pickle.dump(ax, open("pickle/"+str(filename), 'wb'))

	# Correction needed before being applied to real world datasets
	def labelAttribution(self):
		# Get the index of missing labels using the bestInd
		bi_raw = self.bestIndividual.trainingValuePredictions
		bi_true = self.bestIndividual.convertLabelsToInt(self.bestIndividual.training_Y)
		bi_pred = self.bestIndividual.getPred(bi_raw, bi_true)
		missing = []
		for i in bi_pred:
			missing.append([0,i[1]])

		# Count the nº of correct predictions on missing labels
		for i in self.population:
			ind_raw = i.trainingValuePredictions
			ind_true = i.convertLabelsToInt(i.training_Y)
			ind_pred = i.getPred(ind_raw, ind_true)
			for index, j in enumerate(ind_pred):
				missing[index][0] += j[0]

		# Split them into class 0 and class 1 (for debugging)
		class0 = []
		class1 = []
		for j in missing:
			# CORRECTION HERE
			if j[1] < 20:
				class0.append(j)
			if 19 < j[1] < 40:
				class1.append(j)
			if 39 < j[1] < 60:
				class0.append(j)
			if j[1] > 59:
				class1.append(j)

		# Attribution Datasets (CORRECTION HERE)
		if self.currentGeneration == self.max_generation-1:

			ds = pd.read_csv(DATASETS_DIR+DATASETS[0])

			# New datasets with the attributed labels
			for i in missing:
				if i[0] > 0.75*self.population_size:
					ds.at[i[1], 'Y'] = 1
				else:
					ds.at[i[1], 'Y'] = -1
			ds.to_csv(DATASETS_DIR+str(len(self.runCounter)-1)+DATASETS[0])

			sort0 = sorted(class0, key=lambda l:l[1])
			sort1 = sorted(class1, key=lambda l:l[1])

			# Print attribution results
			print("\n")
			print(ds.tail(40))
			print("\n")
			print("Class 0: " + str(sort0))
			print("Class 1: " + str(sort1))

def calculateIndividualAccuracy_MultiProcessing(ind, fitArray, indIndex):
	fitArray[indIndex] = ind.getTrainingAccuracy()

def fitIndividuals(a):
	ind,x,y = a
	ind.fit(x,y)

	return ( ind, ind.predict(x) )

def getTrainingPredictions(ind):
	return ind.getTrainingPredictions()
