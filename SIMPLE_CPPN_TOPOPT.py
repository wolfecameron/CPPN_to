from __future__ import print_function
from SIMPLE_CPPN_Structure import Genotype
import numpy as np
import os
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.special import expit
from SIMPLE_CPPN_DEAP_alg import selectPop2, var_algo
# from matplotlib import colors
# import matplotlib.pyplot as plt
import topopt

# NEAT implementation of topological optimization
POPSIZE = 50
NGEN = 150
numX = 30
numY = 10
volfrac = 0.4
inputs = []

# sets input for nn as the (x,y) locations of each node as a tuple


for x in range(1, numX + 1):
    inputs.append(x)

tmp = np.array(inputs, copy=True)
MEAN = np.mean(tmp)
STD = np.std(tmp)

# list of normalized inputs
normIn = []

# creates input list with normalized vectors
for y in range(0, numY):
    for x in range(0, numX):
        tup = (np.fabs(x - MEAN) / STD, np.fabs(y - MEAN) / STD)
        normIn.append(tup)

top_outputs = []

AVG_FITNESSES = []
INDIVIDUALS = []
NETS = []


def sigmoid_act(x):
    return expit(x)


def eval_genomes(genomes):
    avgFitness = 0
    counter = 0

    # LIST OF VARIABLES FOR fitnes
    nelX = numX
    nelY = numY
    volfrac = .5
    rmin = 5.4
    penal = 3.0
    ft = 1

    for genome in genomes:
        genome.fitness = 0
            #net = neat.nn.FeedForwardNetwork.create(genome, config)
        NETS.append(genome)
        outList = []
            # must unzip all the elements and run the network for each
            # input in input list
        for z in range(len(normIn)):
                # always passes output through sigmoid
            x = sigmoid_act(genome.evaluate(normIn[z]))
                # must make x either 1 or 0
                # if it can be .5 the program will yield a completely grey solution
            if (x >= .5):
                x = 1
            else:
                x = 0
            outList.append(x)

        # initialized as a regular list and copied as a numpy array to resize
        x = np.array(outList, copy=True)
        # calculate the amount of material used
        real_volfrac = np.sum(x) / len(x)
        INDIVIDUALS.append(x)
        # x = x.reshape((numX, numY))
        # fitness function imported from topopt.py file
        fit = topopt.main(nelX, nelY, volfrac, rmin, penal, ft,
                          x)

        avgFitness = avgFitness + fit  # keeps track of average fitness for each generation
        counter = counter + 1
        # penalize the genome if it uses too much material or too little material
        # penalty is scaled based on how for away it is from desired volfrac
        if (real_volfrac == 1 or real_volfrac == 0):
            # attempts to eliminate empty solutions
            genome.fitness -= sys.maxint
        elif (real_volfrac > volfrac):
            penalty = real_volfrac - volfrac
            genome.fitness -= (1 + penalty) * fit
        elif (real_volfrac < .2):
            # cannot let this solution have objective of 0 because it will become optimal
            # must add something to fitness because multiplying 0 by penalty still yields 0
            penalty = np.fabs(real_volfrac - volfrac)
            genome.fitness -= (1 + penalty) * fit + 1000 * (1 + penalty)
        else:
            # just use normal fitness if solution is within volfrac
            genome.fitness -= fit

    print("done")

    AVG_FITNESSES.append(float(avgFitness) / counter)




# finds the fittest genotype out of all fittest examples of a given structure
def getFittestKey(bestInds):
    keys = bestInds.keys()
    fittest = bestInds[keys[0]]
    for i in range(1, len(keys)):
        # print(bestInds[keys[i]].fitness)
        if (fittest.fitness > bestInds[keys[i]].fitness):
            fittest = bestInds[keys[i]]
            #	print(fittest.fitness)
    return fittest


# calculates the average fitness of the past ten generatopms
def getAverageTrailingFitness(trailingFitness):
    total = 0
    for i in range(len(trailingFitness)):
        total += trailingFitness[i]
    return total / STAG_GENS


# defined constants and hyperparameters
NUM_INPUTS = 2
POP_SIZE = 50
# probability crossover, mutatuion, number of generations
cxpb, mutpb, NGEN = .1, .01, 150
# theshold for how little change signals a structural mutation
STAG_THRESHOLD = 5
# pressure for the population to select, higher pressure limits sample space more
SEL_PRESSURE = .1
# float value refers to how many generations the network can remain stagnant for before needing structural change
STAG_GENS = 20.0

generations = 0  # keeps track of number of generations that have passed
#
# true if population requires structural change
structChange = False
# dictionary of the best individuals: key is the structure of the network (number of nodes, number of connections)
# with key being fittest Genotype
bestInds = {}
fitnessTrail = []

# creates initial population
pop = []

for i in range(POP_SIZE):
    geno = Genotype(NUM_INPUTS)
    pop.append(geno)

# sets initial fitness of the population
#for ind in pop:
fit = eval_genomes(pop)

changeBool = 0
# runs evolutionary algorithm
for g in range(NGEN):
    print(g)
    pop = selectPop2(pop, SEL_PRESSURE)
    pop = var_algo(pop, cxpb, mutpb, structChange)  # runs the evolutionary algorithm, returns offspring

    totalFitness = 0  # stores total fitness of whole population
    # best individual of current population set to first individual
    bestInd = pop[0]
    eval_genomes(pop)
    for ind in pop:
        # increments totalFitness of population
        totalFitness = totalFitness + ind.fitness
        # finds fittest individual
        if (ind.fitness < bestInd.fitness):
            bestInd = ind
    # creates a key based on structure of current population
    key = (pop[0].size, len(pop[0].connectionList))
    # if the key not present in best individuals or the fitness is higher than the other
    # example at the key, store the individual
    if (key not in bestInds or bestInds[key].fitness > bestInd.fitness):
        bestInds[key] = bestInd

    structChange = False

    if (len(fitnessTrail) < STAG_GENS):
        fitnessTrail.append(totalFitness)
    else:
        a = getAverageTrailingFitness(fitnessTrail)
        # if fitness has not imporved over a threshold from the average of last STAG_GENS
        # time for structural mutation
        if (a - totalFitness < STAG_THRESHOLD):
            structChange = True
            fitnessTrail = []
        else:
            fitnessTrail.pop(0)
            fitnessTrail.append(totalFitness)
    generations = generations + 1


def printResultsForwardFeed(bestInds):
    genotype = getFittestKey(bestInds)
    print(genotype.evaluate([0, 0]))
    print(genotype.evaluate([0, 1]))
    print(genotype.evaluate([1, 0]))
    print(genotype.evaluate([1, 1]))


# Show output of the most fit genome against training data.

fullOutput = []
# this loop doesnt work because every evaluation needs the full
for xi in normIn:
    fullOutput.append(1 if ((sigmoid_act(getFittestKey(bestInds).evaluate(xi))) >= .5) else 0)

print(fullOutput)
raw_input("Enter anything to plot result")
# Initialize plot and plot the initial design
x = np.array(fullOutput, copy=True)
plt.ion()  # Ensure that redrawing is possible
fig, ax = plt.subplots()
im = ax.imshow(-x.reshape((numX, numY)).T, cmap='gray',
               interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
fig.show()

raw_input("Begin viewing final generation")

# check variable used to stop viewing final generation
# often don't want to look at every individual because there are so many
check = 'a'

for i in range(NGEN * POPSIZE - POPSIZE, NGEN * POPSIZE):
    if (not (check == 'z')):
        plt.ion()
        fig, ax = plt.subplots()
        x = INDIVIDUALS[i]
        im = ax.imshow(-x.reshape(numX, numY).T, cmap='gray',
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        fig.show()
        net = NETS[i]
        print('\nNet Info:\n{!s}'.format(net))
        check = raw_input("Hit anything to view next individual. Enter 'z' to stop viewing.")

raw_input("Enter anything to plot.")

generations = []
for i in range(1, NGEN + 1):
    generations.append(i)

plt.scatter(generations, AVG_FITNESSES)
plt.title("Fitness Over Generations")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()

raw_input("Press anything to end program.")