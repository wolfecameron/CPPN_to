import numpy as np
from SIMPLE_CPPN_Structure import Genotype
#from old_struct import Genotype, CPPN
from SIMPLE_CPPN_DEAP_alg import var_algo, selectPop3
from CPPNActivationFunctions import simpleAct
import math
import csv
from pic_reader import getPixels, convertBinary, graphImage
import matplotlib.pyplot as plt

#stores the size of the structure/image
numX = 50
numY = 50

clearInput = True if(input("Would you like to clear contents of csvfile? (y/n)") == 'y') else False


inputs = []
#creates input values for CPPN for spring optimization
for x in range(1, numX + 1):
    inputs.append(x)

tmp = np.array(inputs, copy = True)
MEAN = np.mean(tmp)
STD = np.std(tmp)

#list of normalized inputs
normIn = [] 

#creates input list with normalized vectors, values of input are of form (x,y) in a list of tuples
for y in range(0,numY):
    for x in range(0,numX):
        tup = (np.fabs(x - MEAN)/STD, np.fabs(y-MEAN)/STD)
        normIn.append(tup)

#holds genome and outputs of final individuals
finalGen = []
finalInds = []
	

#finds the fittest genotype out of all fittest examples of a given structure
def getFittestKey(bestInds):
	keys = list(bestInds.keys())
	fittest = bestInds[keys[0]]
	for i in range(1, len(keys)):
		if(fittest.fitness > bestInds[keys[i]].fitness):
			fittest = bestInds[keys[i]]

	return fittest


#calculates the standard deviation of last 20 fitness values
def getSTDTrailingFitness(trailingFitness):
	np_trail = np.array(trailingFitness, copy = True)
	std = np.std(np_trail)
	return std


#defined constants and hyperparameters
NUM_INPUTS = 2
NUM_OUTPUTS = 1
POP_SIZE = 100

#probability crossover, mutatuion, number of generations
cxpb , mutpb, ngen = .05, .05, 500

#theshold for how little change signals a structural mutation
STD_THRESHOLD = 35.0

#pressure for the population to select, higher pressure limits sample space more
SEL_PRESSURE = .5

#float value refers to how many generations the network can remain stagnant for before needing structural change
STAG_GENS = 25.0

generations = 0 #keeps track of number of generations that have passed 

#true if population requires structural change
structChange = False

#gets pixel values from image for fitness evaluation
pixels = getPixels('./Images/test3.png', numX, numY)
pixels_np = np.array(pixels, copy = True)

#assigns fitness for different CPPN structures 
def evalNetwork(g_param, gen):
	#fitness should be minimized to eliminate difference between CPPN output and existing picture
	fitness = 0

	outputs = []
	for x in range(len(normIn)):

		outputs.append(g_param.evaluate([normIn[x][0],normIn[x][1]])[0])
	
	outputs_np = np.array(outputs, copy = True)

	diff = np.subtract(pixels_np,outputs_np)
	
	diff[diff>=.5] = 4 #creates greater penalty for missing a pixel contained in the spring

	diff = np.absolute(diff)
	fitness = np.sum(diff)

	#appends final generation to a cached list for later observation
	if(gen == ngen -1): 
		finalInds.append(g_param)
		finalGen.append(outputs)

	return fitness,

#dictionary of the best individuals: key is the structure of the network (number of nodes, number of connections)
#with key being fittest Genotype
bestInds = {}


#creates initial population
pop = []


for i in range(POP_SIZE):
	geno = Genotype(NUM_INPUTS, NUM_OUTPUTS)
	pop.append(geno)

#sets initial fitness of the population
for ind in pop:
	fit = evalNetwork(ind,0)
	ind.fitness = fit


if(clearInput):
	with open("or5.csv",'w') as csvfile:
		#this is used to clear contents of file before writing data if needed
		print("CSV File was Cleared!")

#keeps track of average fitness values across generations
AVG_FITNESSES = []

#keeps track of average fitness for last 20 generations
fitnessTrail = []

#runs evolutionary algorithm
for g in range(ngen):
	print('Running Generation: ' + str((g+1)))
	
	if(g > 0):
		with open("or5.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',')
			filewriter.writerow([generations, genFitness/float(POP_SIZE), 'NumNodes: ' + str(len(pop[0].nodeList)), 'NumConnect: ' + str(len(pop[0].connectionList)), 'True' if(structChange) else 'False'])
	genFitness = 0		
	#updates population
	pop = selectPop3(pop)
	pop = var_algo(pop,cxpb, mutpb, structChange) #runs the evolutionary algorithm, returns offspring

	#best individual of current population set to first individual
	bestInd = pop[0]

	for ind in pop:
		ind.fitness = evalNetwork(ind,g)
		genFitness += ind.fitness[0]
		#finds fittest individual
		if(ind.fitness < bestInd.fitness):
			bestInd = ind
	
	#creates a key based on structure of current population, stored as a tuple of values representing number of nodes and number of connections
	key = (pop[0].size, len(pop[0].connectionList))
	#if the key not present in best individuals or the fitness is higher than the other
	#example at the key, store the individual
	if (key not in bestInds or bestInds[key].fitness > bestInd.fitness):
		bestInds[key] =  bestInd
	
	structChange = False

	if(len(fitnessTrail) < STAG_GENS):
		fitnessTrail.append(genFitness/float(POP_SIZE))
	else:
		a = getSTDTrailingFitness(fitnessTrail)
		#if fitness has not imporved over a threshold from the average of last STAG_GENS
		#time for structural mutation
		if(a < STD_THRESHOLD):
				structChange = True
				print("Structural change was added at generation " + str(g))
				fitnessTrail = []
		else:
			del fitnessTrail[0] #deletes first element in list
			fitnessTrail.append(genFitness/float(POP_SIZE))
			


	AVG_FITNESSES.append(genFitness/float(POP_SIZE))
	generations += 1
	#print str(generations) + ": " + str(totalFitness/POP_SIZE)

def printResultsForwardFeed(bestInds):
	genotype = getFittestKey(bestInds)
	outputs = []
	for x in range(len(normIn)):
		outputs.append(genotype.evaluate([normIn[x][0],normIn[x][1]])[0])
	graphImage(outputs, numX, numY)

#print(getFittestKey(bestInds).__str__())
printResultsForwardFeed(bestInds)	

check = 'y'
for x,y in zip(finalGen,finalInds):
	if(check == 'y'):
		graphImage(x, numX, numY)
		#input("Here is the individual.")
		y.graphGenotype()
		input("Here is the network structure of this individual.")
		check = input("would you like to keep viewing (y/n)?")
	else: 
		break

print("Here are the best individuals of each structure.")
check = 'y'
keys = bestInds.keys()
for i in keys:
	if(check == 'y'):
		ind = bestInds[i]
		outputs = []
		for x in range(len(normIn)):
			outputs.append(ind.evaluate([normIn[x][0],normIn[x][1]])[0])
		
		graphImage(outputs,numX,numY)
		#input("Here is the individual.")
		ind.graphGenotype()
		input("Here is the network structure of this individual.")
		check = input("keep viewing? (y/n)")
	else:
		break


#plots all average fitnesses throughout evolution process
plt.plot(AVG_FITNESSES)
plt.ylabel('Average Fitness')
plt.xlabel('Generation')
plt.title('Fitness Versus Generation')
plt.show()
input("Here is your final fitness graph!")