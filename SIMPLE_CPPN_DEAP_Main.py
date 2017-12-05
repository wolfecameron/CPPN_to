
import numpy as np
from SIMPLE_CPPN_Structure import Genotype
#from old_struct import Genotype, CPPN
from SIMPLE_CPPN_DEAP_alg import var_algo, selectPop2
from CPPNActivationFunctions import simpleAct
import math
import csv


#evaluates network based on its XOR performance
def evalNetwork(g_param):
	fitness = 0 #initializes fitness to be empty (0)
	idealResults = [0,1,1,1]
	results = []
	#evaluates first set of inputs
	results.append(float(g_param.evaluate([0,0])))
	results.append(float(g_param.evaluate([0, 1])))
	results.append(float(g_param.evaluate([1, 0])))
	results.append(float(g_param.evaluate([1, 1])))
	for i in range(0,len(idealResults)):
		fitness = fitness + math.fabs(float(idealResults[i]) - results[i]) #keeps values positive
	
	return fitness,
	

#finds the fittest genotype out of all fittest examples of a given structure
def getFittestKey(bestInds):
	keys = bestInds.keys()
	fittest = bestInds[keys[0]]
	for i in range(1, len(keys)):
		#print(bestInds[keys[i]].fitness)
		if(fittest.fitness > bestInds[keys[i]].fitness):
			fittest = bestInds[keys[i]]
		#	print(fittest.fitness)
	return fittest

#calculates the average fitness of the past ten generatopms
def getAverageTrailingFitness(trailingFitness):
	total = 0
	for i in range(len(trailingFitness)):
		total += trailingFitness[i]
	return total/ STAG_GENS

#defined constants and hyperparameters
NUM_INPUTS = 2
POP_SIZE = 50
#probability crossover, mutatuion, number of generations
cxpb , mutpb, ngen = .1, .01, 200
#theshold for how little change signals a structural mutation
STAG_THRESHOLD = 5
#pressure for the population to select, higher pressure limits sample space more
SEL_PRESSURE = .1
#float value refers to how many generations the network can remain stagnant for before needing structural change
STAG_GENS = 10.0

generations = 0 #keeps track of number of generations that have passed 
#
#true if population requires structural change
structChange = False
#dictionary of the best individuals: key is the structure of the network (number of nodes, number of connections)
#with key being fittest Genotype
bestInds = {}
fitnessTrail = []

#creates initial population
pop = []


for i in range(POP_SIZE):
	geno = Genotype(NUM_INPUTS)
	pop.append(geno)

#sets initial fitness of the population
for ind in pop:
	fit = evalNetwork(ind)
	ind.fitness = fit
changeBool = 0
#runs evolutionary algorithm
for g in range(ngen):
	if(g > 0):
		with open("or5.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',')
			if(structChange):
				changeBool = 1
			filewriter.writerow([generations - 1, totalFitness, changeBool])#
			changeBool = 0
	#updates population
	pop = selectPop2(pop, SEL_PRESSURE)
	pop = var_algo(pop,cxpb, mutpb, structChange) #runs the evolutionary algorithm, returns offspring

	totalFitness = 0 #stores total fitness of whole population

	#best individual of current population set to first individual
	bestInd = pop[0]

	for ind in pop:
		ind.fitness = evalNetwork(ind)
		#increments totalFitness of population
		totalFitness = totalFitness + ind.fitness[0]
		#finds fittest individual
		if(ind.fitness < bestInd.fitness):
			bestInd = ind
	#creates a key based on structure of current population
	key = (pop[0].size,len( pop[0].connectionList))
	#if the key not present in best individuals or the fitness is higher than the other
	#example at the key, store the individual
	if (key not in bestInds or bestInds[key].fitness > bestInd.fitness):
		bestInds[key] =  bestInd
	
	structChange = False

	if(len(fitnessTrail) < STAG_GENS):
		fitnessTrail.append(totalFitness)
	else:
		a = getAverageTrailingFitness(fitnessTrail)
		#if fitness has not imporved over a threshold from the average of last STAG_GENS
		#time for structural mutation
		if(a - totalFitness < STAG_THRESHOLD):
				structChange = True
				fitnessTrail = []
		else:
			fitnessTrail.pop(0)
			fitnessTrail.append(totalFitness)
	generations = generations + 1
	print str(generations) + ": " + str(totalFitness)

def printResultsForwardFeed(bestInds):
	genotype = getFittestKey(bestInds)
	print(genotype.evaluate([0,0]))
	print(genotype.evaluate([0,1]))
	print(genotype.evaluate([1, 0]))
	print(genotype.evaluate([1, 1]))

print(getFittestKey(bestInds).__str__())
printResultsForwardFeed(bestInds)
