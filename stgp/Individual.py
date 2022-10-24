from .Node import Node
from .SimpleThresholdClassifier import SimpleThresholdClassifier

import rpy2.robjects.packages as rpackages
from rpy2 import robjects

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019-2021 J. E. Batista
#

class Individual:
	training_X = None
	training_Y = None

	operators = None
	terminals = None
	max_depth = None

	labelToInt = None
	intToLabel = None

	head = None
	size = 0
	depth = 0

	trainingClassPredictions = None
	trainingValuePredictions = None
	testClassPredictions = None
	testValuePredictions = None
	fitness = None

	model_name = ["SimpleThresholdClassifier"][0]
	model = None

	fitnessType = ["Accuracy", "RMSE"][1]

	labelErrors = None
	labeledSamplesX = []
	labeledSamplesY = []

	# VOR Calculation Tools
	dcme = rpackages.importr('dcme')
	rf2 = robjects.r['F2']

	def __init__(self, operators, terminals, max_depth):
		self.operators = operators
		self.terminals = terminals
		self.max_depth = max_depth

	def create(self):
		self.head = Node()
		self.head.create(self.operators, self.terminals, self.max_depth, full=True)
		
	def copy(self, head):
		self.head = head


	def __gt__(self, other):
		sf = self.getFitness()
		ss = self.getSize()

		of = other.getFitness()
		os = other.getSize()

		# Troca o best individual se tiver melhor fitness ou a mesma mas tamanho menor
		return (sf < of) or (sf == of and ss < os)

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return str(self.head)


	def fit(self, Tr_x, Tr_y):
		'''
		Trains the classifier which will be used in the fitness function
		'''
		if self.model is None:
			self.training_X = Tr_x
			self.training_Y = Tr_y

			# Obtain the Tr set elements without missing labels (for accuracy calculations)
			trLen = len(Tr_x)
			self.labeledSamplesX = Tr_x.head(int(trLen/2))
			self.labeledSamplesY = Tr_y.head(int(trLen/2))

			self.labelToInt = {}
			self.intToLabel = {}
			classes = list(set(self.training_Y))
			for i in range(len(classes)):
				self.labelToInt[classes[i]] = i
				self.intToLabel[i] = classes[i]

			if self.model_name == "SimpleThresholdClassifier":
				self.model = SimpleThresholdClassifier()

			hyper_X = self.calculate(Tr_x)

			self.model.fit(hyper_X,Tr_y)


	def getSize(self):
		'''
		Returns the total number of nodes within an individual.
		'''
		return self.head.getSize()


	def getDepth(self):
		'''
		Returns the depth of individual.
		'''
		return self.head.getDepth()


	def clone(self):
		'''
		Returns a deep clone of the individual's list of dimensions.
		'''
		ret = Individual()
		ret.copy(head.clone())
		return ret

	def convertLabelsToInt(self, Y):
		ret = [ self.labelToInt[label] for label in Y ]
		return ret

	def convertIntToLabels(self, Y):
		ret = [ self.intToLabel[value] for value in Y ]
		return ret

	def ssupRMSE(self, y_raw, y_true):
		'''
		Replaces the mean_squared_error fuction from sklearn with a semi-supervised one
		'''
		total = 0
		# y_true = original class label
		# y_raw = raw model prediction value
		for i in range(len(y_true)):
			# After labelToInt: class 1 = 0
			if y_true[i] == 0:
				diff = 1 - y_raw[i]
				sq_diff = diff ** 2
				total = total + sq_diff
			# After labelToInt: class -1 = 2
			elif y_true[i] == 2:
				diff = -1 - y_raw[i]
				sq_diff = diff ** 2
				total = total + sq_diff
			# After labelToInt: class 123 = 1
			elif y_true[i] == 1:
				if y_raw[i] < 0:
					y_pred = -1
				else:
					y_pred = 1
				diff = y_pred - y_raw[i]
				sq_diff = diff ** 2
				total = total + sq_diff
		mse = total/len(y_true)
		rmse = mse ** 0.5
		return rmse

	def getPred(self, y_raw, y_true):
		pred = []
		for i in range(len(y_true)):
			# y_true é a class label
			if y_true[i] == 2 or y_true[i] == 123:
				if y_raw[i] < 0.5:
					y_pred = 0
				else:
					y_pred = 1
				pred.append((y_pred, self.training_Y.index[i]))
		return pred

	def getCertainty(self):
		'''
		Calculates the certainty of the individual for each sample
		'''
		if self.labelErrors is None:

			self.getTrainingValuePredictions()
			y_raw = self.trainingValuePredictions
			y_true = self.convertLabelsToInt(self.training_Y)
			self.labelErrors = [0] * len(y_true)

			for i in range(len(y_raw)):

				# Certeza que a label deve ser -1 e não é
				if y_raw[i] >= -1.1 and y_raw[i] <= -0.9:
					if y_true[i] != -1:
						self.labelErrors[i] += 1
				# Certeza que a label deve ser 1 e não é
				elif y_raw[i] >= 0.9 and y_raw[i] <= 1.1:
					if y_true[i] != 1:
						self.labelErrors[i] += 1
		return self.labelErrors

	def getProximity(self):
		'''
		Calculates the proximity of the individual for each sample
		based on what class label it is closest to
		'''
		if self.labelErrors is None:

			self.getTrainingValuePredictions()
			y_raw = self.trainingValuePredictions
			y_true = self.training_Y.values.tolist()
			self.labelErrors = [0] * len(y_true)

			for i in range(len(y_raw)):

				# Certeza que a label deve ser -1 e não é
				if y_raw[i] < 0:
					if y_true[i] != -1:
						self.labelErrors[i] += 1
				# Certeza que a label deve ser 1 e não é
				elif y_raw[i] >= 0:
					if y_true[i] != 1:
						self.labelErrors[i] += 1
		return self.labelErrors

	def getFitness(self):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness is None:

			if self.fitnessType == "Accuracy":
				self.getTrainingClassPredictions()
				acc = accuracy_score(self.trainingClassPredictions, self.convertLabelsToInt(self.training_Y))
				self.fitness = acc

			if self.fitnessType == "RMSE":
				self.getTrainingValuePredictions()
				waf = mean_squared_error(self.trainingValuePredictions, list(self.training_Y))**0.5
				#waf = self.ssupRMSE(self.trainingValuePredictions, self.convertLabelsToInt(self.training_Y))
				self.fitness = waf

		return self.fitness

	def getTrainingClassPredictions(self):
		if self.trainingClassPredictions is None:
			self.trainingClassPredictions = self.predict(self.training_X, classOutput = True) #labeledSamplesX/training_X

		return self.trainingClassPredictions

	def getTestClassPredictions(self,X):
		if self.testClassPredictions is None:
			self.testClassPredictions = self.predict(X, classOutput = True)

		return self.testClassPredictions

	def getTrainingValuePredictions(self):
		if self.trainingValuePredictions is None:
			self.trainingValuePredictions = self.predict(self.training_X, classOutput = False)

		return self.trainingValuePredictions

	def getTestValuePredictions(self,X):
		if self.testValuePredictions is None:
			self.testValuePredictions = self.predict(X, classOutput = False)

		return self.testValuePredictions



	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingClassPredictions()
		elif pred == "Te":
			pred = self.getTestClassPredictions(X)
		else:
			pred = self.predict(X)

		return accuracy_score(pred, Y)


	def getRMSE(self, X, Y,pred=None):
		'''
		Returns the individual's WAF.
		'''
		if pred == "Tr":
			pred = self.getTrainingValuePredictions()
		elif pred == "Te":
			pred = self.getTestValuePredictions(X)
		else:
			pred = self.predict(X, classOutput = False)

		waf = mean_squared_error(pred, Y) ** 0.5
		#waf = self.ssupRMSE(pred, self.convertLabelsToInt(Y))
		return waf



	def calculate(self, X):
		'''
		Returns the converted input space.
		'''
		calc = self.head.calculate(X)
		return self.head.calculate(X)


	def predict(self, X, classOutput=True):
		'''
		Returns the class prediction of a sample.
		'''
		hyper_X = self.calculate(X)
		if classOutput:
			predictions = self.model.predict(hyper_X)
		else:
			predictions = hyper_X

		return predictions

	def fromString(self,string):
		'''
		Criar modelos a partir de string
		'''
		n = Node()
		n.fromString(string.split(" "), [0], self.operators, self.terminals)
		self.head = n

	def prun(self):
		'''
		Remove the dimensions that degrade the fitness.
		If simp==True, also simplifies each dimension.
		'''
		done = False
		while not done:
			state = str(self.head)
			self.head.prun(self.training_X)
			done = state == str(self.head)



