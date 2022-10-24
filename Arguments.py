from sys import argv

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#


# Operators to be used by the models
# Only these operators are available. To add mode, edit m3gp.Node.calculate(self, sample)
OPERATORS = ["+","-","*","/"]  # "max","min"

# Initial Maximum depth (3/6)
MAX_DEPTH = 3

# Number of models in the population (500)
POPULATION_SIZE = 300

# Maximum number of iterations (100)
MAX_GENERATION = 100

# Fraction of the dataset to be used as training (used by Main_M3GP_standalone.py)
TRAIN_FRACTION = 0.70

# Number of individuals to be used in the tournament (5)
TOURNAMENT_SIZE = 5

# Number of best individuals to be automatically moved to the next generation (1)
ELITISM_SIZE = 1

# Shuffle the dataset (used by Main_M3GP_standalone.py) (True)
SHUFFLE = True

# Dimensions maximum depth (6/17)
LIMIT_DEPTH = 6

# Number of runs (used by Main_M3GP_standalone.py) (30)
RUNS = 1

# Use K-Fold data splitting or not
KFOLD = 0

# Use Semi-supervised classification or not
SSUP = 0

# Number of folds for K-Fold, not used if KFOLD = 0
SPLITS = 10

# Verbose
VERBOSE = True

# Number of CPU Threads to be used
THREADS = 1


DATASETS_DIR = "datasets/"
OUTPUT_DIR = "results/"

DATASETS = ["BrasilScalNeg.csv"]  # BrasilScalNeg / ProteinScalNeg
OUTPUT = "Classification"




if "-dsdir" in argv:
	DATASETS_DIR = argv[argv.index("-dsdir")+1]

if "-odir" in argv:
	OUTPUT_DIR = argv[argv.index("-odir")+1]

if "-d" in argv:
	DATASETS = argv[argv.index("-d")+1].split(";")

if "-runs" in argv:
	RUNS = int(argv[argv.index("-runs")+1])

if "-op" in argv:
	OPERATORS = argv[argv.index("-op")+1].split(";")

if "-md" in argv:
	MAX_DEPTH = int(argv[argv.index("-md")+1])

if "-ps" in argv:
	POPULATION_SIZE = int(argv[argv.index("-ps")+1])

if "-mg" in argv:
	MAX_GENERATION = int(argv[argv.index("-mg")+1])

if "-tf" in argv:
	TRAIN_FRACTION = float(argv[argv.index("-tf")+1])

if "-ts" in argv:
	TOURNAMENT_SIZE = int(argv[argv.index("-ts")+1])

if "-es" in argv:
	ELITISM_SIZE = int(argv[argv.index("-es")+1])

if "-dontshuffle" in argv:
	SHUFFLE = False

if "-s" in argv:
	VERBOSE = False

if "-t" in argv:
	THREADS = int(argv[argv.index("-t")+1])


