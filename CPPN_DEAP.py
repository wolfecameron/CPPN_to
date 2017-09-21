from deap import algorithms, base, creator, tools
import numpy as np
from CPPNStructure import Genotype, CPPNNode
from DEAP_alg import var_algo
import math




#evaluates network based on its XOR performance
def evalNetwork(g_param):
	fitness = 0 #initializes fitness to be empty (0)

	idealResults = [0,1,1,0]
	results = []
	
	inputs = np.matrix([[0,0],[0,1],[1,0],[1,1]])
	
	#evaluates first set of inputs
	g_param.setInput(inputs[0])
	CPPN = g_param.getCPPNNodes()
	results.append(CPPN.evaluateCPPN())
	
	#each result is appended to the result list
	#2nd set
	g_param.setInput(inputs[1])
	CPPN = g_param.getCPPNNodes()
	results.append(CPPN.evaluateCPPN())
	
	#3rd set
	g_param.setInput(inputs[2])
	CPPN = g_param.getCPPNNodes()
	results.append(CPPN.evaluateCPPN())
	
	#4th set
	g_param.setInput(inputs[3])
	CPPN = g_param.getCPPNNodes()
	results.append(CPPN.evaluateCPPN())

	#increments the fitness based on the squared distance between results and optimal results 
	for i in range(0,len(idealResults)):
		fitness = fitness + math.pow((idealResults[i] - results[i]),2) #squared to keep values positive
	
	
	
	return fitness,
	

NUM_INPUTS = 2
POP_SIZE = 15

#must define variables before for pointMutate	
cxpb , mutpb, ngen = .05, .05, 5
	
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("individual", tools.initRepeat, creator.Individual, Genotype, NUM_INPUTS, n=1)
tb.register("population", tools.initRepeat, list, tb.individual, n = POP_SIZE)
tb.register("evaluate", evalNetwork)
'''
tb.register("mate", Genotype.crossover)
tb.register("pointMutate", Genotype.pointMutate,mutpb)
tb.register("nodeMutate", Genotype.nodeMutate)
tb.register("linkMutate", Genotype.linkMutate)
tb.register("disableMutate", Genotype.disableMutate)
'''
tb.register("select", tools.selTournament, tournsize = 5)
tb.register("map", map)

pop = [] #creates original population
for i in range(POP_SIZE):
	pop.append(Genotype(NUM_INPUTS))

fits = tb.map(tb.evaluate, pop) #runs evaluation on all members of population, gets fitness values in a list fits[]

for ind,fit in zip(pop,fits): #assigns fitness value to each ind in pop[]
	ind.fitness = fit 

for g in range(ngen): 
	pop = tb.select(pop, k = len(pop))
	pop = var_algo(pop, tb, cxpb, mutpb) #runs the evolutionary algorithm, returns offspring
	
	fits = tb.map(tb.evaluate, pop) #runs evaluation on all members of population, gets fitness values in a list fits[]

	for ind,fit in zip(pop,fits): #assigns fitness value to each ind in pop[]
		ind.fitness = fit 
	
	

result = pop[0]
result.setInput([0,0])
CPPN = result.getCPPNNodes()
print CPPN.evaluateCPPN()

'''
fits = tb.map(tb.evaluate,population)

for ind,fit in zip(range(len(population)),fits):
	ind.fitness.values = fit
	
for g in range(ngen):
	population = tb.select(population, k = len(population))
'''
