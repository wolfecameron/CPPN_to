from __future__ import print_function
import neat
import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from topopt import main as fitness

# NEAT implementation of topological optimization
numX = 20
numY = 10
volfrac = 0.4
top_inputs = []

# sets input for nn as the (x,y) locations of each node as a tuple

for x in range(1, numX + 1):
    for y in range(1, numY + 1):
        top_inputs.append((x - .5, y - .5))

# print(top_inputs)
top_outputs = []


def eval_genomes(genomes, config):
    avgFitness = 0
    counter = 0

    # LIST OF VARIABLES FOR fitnes
    nelX = 20
    nelY = 10
    volfrac = .4
    rmin = 5.4
    penal = 3.0
    ft = 1

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        outList = []
        # must unzip all the elements and run the network for each
        # input in input list
        for xi in top_inputs:
            outList.append(net.activate(xi))

        # initialized as a regular list and copied as a numpy array to resize
        x = np.array(outList, copy=True)
        x = x.reshape((numX, numY))

        # fitness function imported from topopt.py file
        fit = fitness(nelX, nelY, volfrac, rmin, penal, ft, x)  # should plot with every iteration
        print(fit)

        avgFitness = avgFitness + fit
        counter = counter + 1

        genome.fitness += fit

    print("Average Fitness is " + str(float(avgFitness / counter)))


print("Step 1")
# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-topOpt')

print("Step 2")
# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

print("Step 3")
# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

print("Step 4")
# Run for 100 generations.
winner = p.run(eval_genomes, n=50)

print("Step 5")
# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


'''
plt.title("Population's average and best fitness")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()
'''
fullOutput = []
# this loop doesnt work because every evaluation needs the full
for xi in zip(top_inputs):
    fullOutput.append(winner_net.activate(xi))

raw_input("Enter anything to plot result")
# Initialize plot and plot the initial design
x = np.array(fullOutput, copy=True)
plt.ion()  # Ensure that redrawing is possible
fig, ax = plt.subplots()
im = ax.imshow(-x.reshape((numX, numY)).T, cmap='gray',
               interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
fig.show()
