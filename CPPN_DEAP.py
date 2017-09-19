from deap import algorithms, base, creator, tools
import numpy as np
from CPPNStructure import Genotype, CPPNNode




#evaluates network based on its XOR performance
def evalNetwork(g_param):
	fitness = 0 #initializes fitness to be empty (0)

	idealResults = [0,1,1,0]
	
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
		fitness = fitness + np.square((idealResults[i] - results[i]), 2)
	
	
	
	return fitness,
	

#must define variables before for pointMutate	
cxpb , mutpb, ngen = .05, .05, 400
	
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("CPPN", Genotype,2)
tb.register("individual", tools.initRepeat, creator.Individual, tb.CPPN, n=1)
tb.register("population", tools.initRepeat, list, tb.individual, n = 400)
tb.register("evaluate", evalNetwork)
tb.register("mate", Genotype.crossover)
tb.register("pointMutate", Genotype.pointMutate,mutpb)
tb.register("nodeMutate", Genotype.nodeMutate)
tb.register("linkMutate", Genotype.linkMutate)
tb.register("disableMutate", Genotype.disableMutate)
tb.register("select", tools.selTournament, tournsize = 5)
tb.register("map", map)


'''
fits = tb.map(tb.evaluate,population)

for ind,fit in zip(range(len(population)),fits):
	ind.fitness.values = fit
	
for g in range(ngen):
	population = tb.select(population, k = len(population))
'''
