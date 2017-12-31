import random
from SIMPLE_CPPN_Structure import Genotype
import copy
import matplotlib.pyplot as plt


def var_algo(population, cxpb, mutpb, structChange):
	# creates copy of population to vary
	#for each offspring
	offspring = [ind for ind in population]
	randStartNode = random.randint(0, offspring[0].size - 1)
	randEndNode = random.randint(offspring[0].numIn, offspring[0].size - 1) #should start of randint be randStartNode
	tries = 0
	if structChange:
		choice = random.choice([1,2])
		if(choice == 1):
			successLink = linkMutatePop(offspring, randStartNode, randEndNode)
			if(not successLink):
				nodeMutatePop(offspring, randStartNode,randEndNode)
				#print("NODE MUTATE")
			#else:
				#print("LINK MUTATE")
		
		if(choice == 2):
			nodeMutatePop(offspring, randStartNode, randEndNode)
			#print("NODE MUTATE")

			#	print (offspring[0].__str__())
			#	if(structChange):
			#		print("Testing after mutate ")
			#		testingSameStruct(offspring)
	'''
	DONT NEED TO ADD EXTRA POINT MUTATIONS IMMEDIATELY AFTER STRUCTURAL MUTATION
	for i in range(len(offspring)):
		if (offspring[i].weightMutate(mutpb)):
			offspring[i].fitness = 0
		if(offspring[i].activationMutate(mutpb)):
			offspring[i].fitness = 0


		# elif r > (mutpb/2) and r < mutpb and structChange:
		# random.choice([offspring[i].linkMutate(), offspring[i].disableMutate()])
		# offspring[i].fitness = 0
		#	print("Testing after point mutate")
		#	testingSameStruct(offspring)

	for i in range(1, len(offspring)):
		x = random.random()
		if (x < cxpb):
			offspring[i] = offspring[i].crossover(offspring[i - 1])
			offspring[i].fitness = 0

		#	print("Testing after crossover")
		#	testingSameStruct(offspring)
	# print ("Var worked")
	'''
	#Returns population with new structural mutation
	return offspring



#tests entire population to make sure the individuals have the same genotype (all same structure different weights)
def testingSameStruct(pop):
	for i in range(len(pop)):
		for j in range(len(pop)):
			if(not(i == j)):
				if(not(len(pop[i].nodeList) == len(pop[j].nodeList)) or not(len(pop[i].connectionList) == len(pop[j].connectionList))):
					print("problem " + str(pop[i]) + " " + str(pop[j])) 


def getSecondStartNode(a, firstNode):
	secondNode = random.randint(0, a)
	tries = 0
	while (secondNode == firstNode and tries < 10):
		secondNode = random.randint(0,a)
		tries = tries + 1
	return secondNode



def selRand(individuals):
	#randomly selects k individuals out of the population
	return individuals[random.randint(0,len(individuals)-1)]
	


def findFittest(tourn):
	fittest = tourn[0] #sets fittest to an initial value
	for i in range(1,len(tourn)):
		if(tourn[i].fitness < fittest.fitness):
			fittest = tourn[i]
	
	return fittest



#creates a new connection in all of individuals in a population (same connection with different weight for every individual)
def linkMutatePop(offspring, randStartNode, randEndNode):
	tries = 0
	valid = offspring[0].linkMutate(randStartNode, randEndNode);
	while (not valid and  tries < 20):
		tries = tries + 1
		randStartNode = random.randint(0, offspring[0].size - 1)
		randEndNode = random.randint(offspring[0].numIn, offspring[0].size - 1)
		valid = offspring[0].linkMutate(randStartNode, randEndNode);
	if valid:
#		print("SUCCESS LINK MUTATE")
#		print(offspring[0].__str__())
		offspring[0].fitness = 0
		for i in range(1, len(offspring)):
#				print("NODE "+ str(i))
#				print ("connections " + str(len(offspring[i].connectionList)))
			offspring[i].linkMutate(randStartNode, randEndNode)
# print(len(offspring[i].connectionList))
			offspring[i].fitness = 0
	else: 
		print("Population Link Mutate Was Unsuccessful After 10 Tries")


	return valid


#inserts new node into entire population (same node into each individual with different weights)
def nodeMutatePop(offspring, randStartNode, randEndNode):
	tries = 0
	layerNum = random.randint(1, offspring[0].highestHidden + 1)
	randStartNode2 = getSecondStartNode(offspring[0].size - 1, randStartNode)
	valid = offspring[0].nodeMutate(randStartNode, randStartNode2, randEndNode, layerNum)
	while (not valid and tries < 100):
		tries = tries + 1
		randStartNode = random.randint(0, offspring[0].size - 1)
		randStartNode2 = getSecondStartNode(offspring[0].size - 1, randStartNode)
		randEndNode = random.randint(offspring[0].numIn, offspring[0].size - 1)
		layerNum = random.randint(1, offspring[0].highestHidden + 1)
		valid = offspring[0].nodeMutate(randStartNode, randStartNode2, randEndNode, layerNum)
	if valid:
		#			print("SUCCESS NODE MUTATE")
		#			print(offspring[0].__str__())
		offspring[0].fitness = 0
		#			print("node mutate")
		for i in range(1, len(offspring)):
			#				print("node " + str(i))
			offspring[i].nodeMutate(randStartNode, randStartNode2, randEndNode, layerNum)
			offspring[i].fitness = 0


	
def selectPop(population, numReturn, tournSize): #inputs: population list, number of individuals to return, number of individuals in each tournament
	newPop = [] #holds list of selected individuals
	for i in range(numReturn):
		competitors = selRand(population, tournSize)
		#for x in competitors:
			#print x.fitness
			
		#print("|")
		
		
		newPop.append(copy.deepcopy(findFittest(competitors)))
		
	return newPop



def selectPop2(population, selectPressure):
	sortedPop = sorted(population, key=lambda ind: ind.fitness, reverse=False)
	newPop = []
	topNum = int((1.0 - selectPressure) * len(population))
	for i in range(topNum):
		newPop.append(sortedPop[i])
	for i in range(len(population) - topNum):
		newPop.append(copy.deepcopy(selRand(newPop))) #why is a deep copy needed?
	
	return newPop


'''
#tests the select function	
population = []	
for i in range(100):
	population.append(Genotype(4))
	population[i].fitness = random.randint(0,100)
	

population = select(population, 100,5) 

for i in population:
	print i.fitness
	
print len(population)

tests the findFittest Function		
x = Genotype(4)
x.fitness = 10
y = Genotype(4)
y.fitness = 9
z = Genotype(4)
z.fitness = -6
a = Genotype(4)
a.fitness = 7
b = Genotype(4)
b.fitness = 6

compList = [x,y,z,a,b]

k = findFittest(compList)
print k.fitness
'''
