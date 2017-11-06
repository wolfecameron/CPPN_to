from deap import algorithms, base, creator, tools
import numpy as np
from SIMPLE_CPPN_Structure import Genotype, CPPN
from SIMPLE_CPPN_DEAP_alg import var_algo, selectPop2
from CPPNActivationFunctions import simpleAct
import math




#evaluates network based on its XOR performance
def evalNetwork(g_param):
	fitness = 0 #initializes fitness to be empty (0)

	idealResults = [0,1,1,0]
	results = []

	
	#evaluates first set of inputs
	g_param.inputValues([0,0])
	CPPN = g_param.getCPPNNodes()
	result1 = float(CPPN.evaluateCPPN())
	results.append((result1))
	
	#each result is appended to the result list
	#2nd set
	g_param.inputValues([0,1])
	CPPN2 = g_param.getCPPNNodes()
	result2 = float(CPPN2.evaluateCPPN())
	results.append((result2))
	
	#3rd set
	g_param.inputValues([1,0])
	CPPN3 = g_param.getCPPNNodes()
	result3 = float(CPPN3.evaluateCPPN())
	results.append((result3))
	
	#4th set
	g_param.inputValues([1,1])
	CPPN4 = g_param.getCPPNNodes()
	result4 = float(CPPN4.evaluateCPPN())
	results.append((result4))

	#increments the fitness based on the squared distance between results and optimal results 
	for i in range(0,len(idealResults)):
		fitness = fitness + math.fabs(float(idealResults[i]) - results[i]) #keeps values positive
	
	return fitness,
	


def checkSizes(population):
	origSize = len(population[0].nodeList)
	origConnect = len(population[0].connectionList)
	problem = False
	for i in range( len( population)):
		if(len(population[i].nodeList) != origSize):
			print("NODE NUMBER NOT THE SAME: " + str(i))
			print(" " + str(len(population[i].nodeList)) + " versus " + str(origSize ))
			problem = True
		if(len(population[i].connectionList) != origConnect):
			print("NODE CONNECTIONS NOT THE SAME: " + str(i))
			problem = True
	if(not problem):
		print("No issue")

def checkSamePointer(population):
	problem = False;
	for i in range(len(population)):
		for j in range(len(population)):
			if(i != j and population[j] == population[i]):
				print("same instance" + str(i) + str(j))
				problem = True;
	if(not problem):
		print("NO PROBLEM")
def getFittestKey(bestInds):
	keys = bestInds.keys()
	fittest = bestInds[keys[0]]
	for i in range(1, len(keys)):
		#print(bestInds[keys[i]].fitness)
		if(fittest.fitness > bestInds[keys[i]].fitness):
			fittest = bestInds[keys[i]]
		#	print(fittest.fitness)
	return fittest
def getAverageTrailingFitness(trailingFitness):
	total = 0
	for i in range(len(trailingFitness)):
		total += trailingFitness[i]
	return total/ 10.0
NUM_INPUTS = 2
POP_SIZE = 50
T_SIZE = 2 #size of each tournament

#must define variables before for pointMutate	
cxpb , mutpb, ngen = 0, .01, 200

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("individual", tools.initRepeat, creator.Individual, Genotype, NUM_INPUTS)
#tb.register("individual", tools.initRepeat, creator.Individual, Genotype, NUM_INPUTS, n=1)
tb.register("population", tools.initRepeat, list, tb.individual)
#tb.register("population", tools.initRepeat, list, tb.individual, n = POP_SIZE)
tb.register("evaluate", evalNetwork)

tb.register("select", tools.selTournament, tournsize = T_SIZE)
tb.register("map", map)



generations = 0 #keeps track of number of generations that have passed 
#print(generations)

#pop = tb.population(n = POP_SIZE);
pop = [] #creates original population
for i in range(POP_SIZE):
	geno = Genotype(NUM_INPUTS)
	pop.append( geno)
	#print(i)

#runs evaluation on all members of population, gets fitness values in a list fits[]
#fitnessList = []
for ind in pop: 
	fit = evalNetwork(ind)
	ind.fitness = fit
	#fitnessList.append(fit)
	
#print float(sum(fitnessList)/len(fitnessList))		

#last fitness that increased more than threshold from previous generation
baseFitness = 0
#the minimum amount of growth needed to not require a structural mutation after
#3maxStagnantGens 
stagnantThreshold = 5
#if population can get structural mutation
structChange = False
successFitness = 0
bestInds = {}

count = 0
fitnessTrail = []
for g in range(ngen):
	pop = selectPop2(pop, .1)
	pop = var_algo(pop,cxpb, mutpb, structChange) #runs the evolutionary algorithm, returns offspring
	totalFitness = 0 #stores total fitness of whole population
	max = 0
	loop = 0
	bestInd = pop[0]
	for ind in pop:
		fits = evalNetwork(ind)
		ind.fitness = fits
		totalFitness = totalFitness + fits[0]
#		print(ind.__str__())
		if(fits < bestInd.fitness):
			bestInd = ind
#			print("above was best")
	key = (pop[0].size,len( pop[0].connectionList))
	if (key not in bestInds or bestInds[key].fitness > bestInd.fitness):
		bestInds[key] =  bestInd
	
	structChange = False
	if(len(fitnessTrail) < 10):
		fitnessTrail.append(totalFitness)
	else:
		a = getAverageTrailingFitness(fitnessTrail)
		if(a - totalFitness < stagnantThreshold):
				structChange = True
				fitnessTrail = []
		else:
			fitnessTrail.pop(0)
			fitnessTrail.append(totalFitness)
	#if(totalFitness - baseFitness > stagnantThreshold):
	#	stagnantCount = stagnantCount + 1
	#else:
#		print("update")
	#	baseFitness = totalFitness
	#	stagnantCount = 0
	#if(stagnantCount >= maxStagnantGens):
		#structChange = True
		#stagnantCount = 0
	generations = generations + 1

	
	#avgFit = (fit/float(len(pop))) #computes average fitness for each generation
	print str(generations) + ": " + str(totalFitness)
print(bestInds)
print getFittestKey(bestInds)
#test all possible inputs
result1 = getFittestKey(bestInds)
result1.inputValues([0,0])
CPPN1 = result1.getCPPNNodes()
print simpleAct(CPPN1.evaluateCPPN())


result2 = getFittestKey(bestInds)
result2.inputValues([0,1])
CPPN2 = result2.getCPPNNodes()
print simpleAct(CPPN2.evaluateCPPN())

result3 = getFittestKey(bestInds)
result3.inputValues([1,0])
CPPN3 = result3.getCPPNNodes()
print simpleAct(CPPN3.evaluateCPPN())

result4 = getFittestKey(bestInds)
result4.inputValues([1,1])
CPPN4 = result4.getCPPNNodes()
print simpleAct(CPPN4.evaluateCPPN())

'''
fits = tb.map(tb.evaluate,population)

for ind,fit in zip(range(len(population)),fits):
	ind.fitness.values = fit
	
for g in range(ngen):
	population = tb.select(population, k = len(population))
'''
