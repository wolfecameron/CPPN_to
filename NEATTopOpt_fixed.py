from __future__ import print_function
import neat
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
# from matplotlib import colors
# import matplotlib.pyplot as plt
import topopt


# NEAT implementation of topological optimization
POPSIZE = 15
NGEN = 100
numX = 60
numY = 20
volfrac = 0.5
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


def eval_genomes(genomes, config):
    avgFitness = 0
    counter = 0

    # LIST OF VARIABLES FOR fitnes
    nelX = numX
    nelY = numY
    volfrac = .5
    rmin = 5.4
    penal = 3.0
    ft = 1

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        NETS.append(genome)
        outList = []
        # must unzip all the elements and run the network for each
        # input in input list
        for z in range(len(normIn)):
            # always passes output through sigmoid
            x = sigmoid_act(net.activate(normIn[z])[0])
            # must make x either 1 or 0
            # if it can be .5 the program will yield a completely grey solution
            if(x >= .5):
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
        if(real_volfrac > volfrac):
            penalty = real_volfrac - volfrac
            genome.fitness += 100 * (1 + penalty) * fit
        elif(real_volfrac < .2):
            # cannot let this solution have objective of 0 because it will become optimal
            # must add something to fitness because multiplying 0 by penalty still yields 0
            penalty = np.fabs(real_volfrac - volfrac)
            genome.fitness += (1 + penalty) * fit + 100000 * (1 + penalty)
        else:
            # just use normal fitness if solution is within volfrac
            genome.fitness += fit

    AVG_FITNESSES.append(float(avgFitness) / counter)



# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-topOpt')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for 100 generations.
winner = p.run(eval_genomes, n=NGEN)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
print('\nNet Info:\n{!s}'.format(winner))
fullOutput = []
# this loop doesnt work because every evaluation needs the full
for xi in normIn:
    fullOutput.append(1 if((sigmoid_act(winner_net.activate(xi)[0])) >= .5) else 0)

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
    if(not(check == 'z')):
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
