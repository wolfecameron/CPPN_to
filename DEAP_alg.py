import random
from CPPNStructure import Genotype

#this function is the varAnd DEAP function adapted to fit the needs of our structure
def var_algo(population, toolbox, cxpb, mutpb):
	
	#creates copy of population to vary
	offspring = [toolbox.clone(ind) for ind in population]

	for i in range(len(offspring)):
		r = random.random()
		if r < (mutpb/2):
			random.choice([offspring[i].pointMutate(mutpb), offspring[i].nodeMutate()])
			offspring[i].fitness = 0
		
		elif r > (mutpb/2) and r < mutpb:
			random.choice([offspring[i].linkMutate(), offspring[i].disableMutate()])
			offspring[i].fitness = 0
	
	return offspring