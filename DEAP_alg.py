import random
from CPPNStructure import Genotype


#this function is the varAnd DEAP function adapted to fit the needs of our structure
def var_algo(population,cxpb, mutpb):
	
	#creates copy of population to vary
	offspring = [ind for ind in population]

	for i in range(len(offspring)):

		r = random.random()
		if r < (mutpb/2):
			random.choice([offspring[i].pointMutate(mutpb),offspring[i].nodeMutate()])
			offspring[i].fitness = 0
		
		
		elif r > (mutpb/2) and r < mutpb:
			random.choice([offspring[i].linkMutate(), offspring[i].disableMutate()])
			offspring[i].fitness = 0
		
	
	for i in range(1,len(offspring)):
		x = random.random()
		if(x<cxpb):
			offspring[i] = offspring[i].crossover(offspring[i-1])
			offspring[i].fitness = 0
			
	
	#print ("Var worked")
	return offspring

def selRand(individuals, k):
	#randomly selects k individuals out of the population
	size = len(individuals)
	return [individuals[random.randint(0,size-1)] for i in range(k)]
	
def findFittest(tourn):
	fittest = tourn[0] #sets fittest to an initial value
	for i in range(1,len(tourn)):
		if(tourn[i].fitness < fittest.fitness):
			fittest = tourn[i]
	
	return fittest
	
def select(population, numReturn, tournSize): #inputs: population list, number of individuals to return, number of individuals in each tournament
	chosen = [] #holds list of selected individuals
	for i in range(numReturn):
		competitors = selRand(population, tournSize)
		#for x in competitors:
			#print x.fitness
			
		#print("|")
		
		
		chosen.append(findFittest(competitors))
		
	return chosen	
	
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