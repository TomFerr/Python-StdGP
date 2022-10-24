from .Population import Population

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#

class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message


class STGP:
	population = None

	operators = None
	max_depth = None
	population_size = None
	max_generation = None
	tournament_size = None
	elitism_size = None
	limit_depth =None
	threads = None
	verbose = None

	def checkIfTrained(self):
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")


	def __init__(self, operators=["+","-","*","/"], max_depth = 6, population_size = 500, 
		max_generation = 100, tournament_size = 5, elitism_size = 1, limit_depth = 17, 
		threads=1, verbose = True):

		if sum( [0 if op in ["+","-","*","/","max","min"] else 0 for op in operators ] ) > 0:
			print( "[Warning] Some of the following operators may not be supported:", operators)
		self.operators = operators
		self.max_depth = max_depth
		self.population_size = population_size
		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size
		self.limit_depth = limit_depth
		self.threads = max(1, threads)
		self.verbose = verbose
		pass

	def __str__(self):
		self.checkIfTrained()
		
		return str(self.getBestIndividual())
		

	def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
		if self.verbose:
			print("Training a model with the following parameters: ", end="")
			print("{Operators : "+str(self.operators)+"}, ", end="")
			print("{Max Initial Depth : "+str(self.max_depth)+"}, ", end="")
			print("{Population Size : "+str(self.population_size)+"}, ", end="")
			print("{Max Generation : "+str(self.max_generation)+"}, ", end="")
			print("{Tournament Size : "+str(self.tournament_size)+"}, ", end="")
			print("{Elitism Size : "+str(self.elitism_size)+"}, ", end="")
			print("{Depth Limit : "+str(self.limit_depth)+"}, ", end="")
			print("{Threads : "+str(self.threads)+"}, ", end="\n")

		self.population = Population(Tr_X, Tr_Y, Te_X, Te_Y, self.operators, self.max_depth,
			self.population_size, self.max_generation, self.tournament_size, self.elitism_size, 
			self.limit_depth, self.threads, self.verbose)
		self.population.train()

		self.getBestIndividual().prun()


	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.population.getBestIndividual().predict(dataset)

	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		self.checkIfTrained()

		return self.population.getBestIndividual()

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.population.getTrainingAccuracyOverTime(), self.population.getTestAccuracyOverTime()]

	def getRMSEOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.population.getTrainingRMSEOverTime(), self.population.getTestRMSEOverTime()]


	def getSizeOverTime(self):
		'''
		Returns the size and number of dimensions of the best model in each generation.
		'''
		self.checkIfTrained()

		return self.population.getSizeOverTime()

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.population.getGenerationTimes()

	def getNumCorrections(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.population.getNumCorrections()

	def getListCorrections(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.population.getListCorrections()