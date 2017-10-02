from deap import algorithms, base, creator, tools
import numpy as np
from CPPNStructure import Genotype, CPPN
from DEAP_alg import var_algo, select
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
	

NUM_INPUTS = 2
POP_SIZE = 50
T_SIZE = 3 #size of each tournament

#must define variables before for pointMutate	
cxpb , mutpb, ngen = .2, .2, 200

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("individual", tools.initRepeat, creator.Individual, Genotype, NUM_INPUTS, n=1)
tb.register("population", tools.initRepeat, list, tb.individual, n = POP_SIZE)
tb.register("evaluate", evalNetwork)

tb.register("select", tools.selTournament, tournsize = T_SIZE)
tb.register("map", map)



generations = 0 #keeps track of number of generations that have passed 
#print(generations)

pop = [] #creates original population
for i in range(POP_SIZE):
	pop.append(Genotype(NUM_INPUTS))
	#print(i)

#runs evaluation on all members of population, gets fitness values in a list fits[]
#fitnessList = []
for ind in pop: 
	fit = evalNetwork(ind)
	ind.fitness = fit
	#fitnessList.append(fit)
	
	
#print float(sum(fitnessList)/len(fitnessList))		
finalMax = 0
for g in range(ngen): 
	pop = select(pop,len(pop), T_SIZE)
	pop = var_algo(pop,cxpb, mutpb) #runs the evolutionary algorithm, returns offspring
	
	max = 0
	loop = 0
	for ind in pop:
		fits = evalNetwork(ind)
		ind.fitness = fits
		if(fits > pop[0].fitness):
			max = loop
		loop = loop + 1
	finalMax = max 	
	generations = generations + 1	
	#avgFit = (fit/float(len(pop))) #computes average fitness for each generation
	print str(generations)# + ": " + str(avgFit)

	
	

#test all possible inputs
result1 = pop[finalMax]
result1.inputValues([0,0])
CPPN1 = result1.getCPPNNodes()
print simpleAct(CPPN1.evaluateCPPN())


result2 = pop[finalMax]
result2.inputValues([0,1])
CPPN2 = result2.getCPPNNodes()
print simpleAct(CPPN2.evaluateCPPN())

result3 = pop[finalMax]
result3.inputValues([1,0])
CPPN3 = result3.getCPPNNodes()
print simpleAct(CPPN3.evaluateCPPN())

result4 = pop[finalMax]
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
