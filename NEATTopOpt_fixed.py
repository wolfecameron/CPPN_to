from __future__ import print_function
import neat
import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors
# from matplotlib import colors
# import matplotlib.pyplot as plt
import topopt


# NEAT implementation of topological optimization
POPSIZE = 50
NGEN = 3
numX = 10
numY = 5
volfrac = 0.4
top_inputs = []

# always used for network output


def sigmoid_act(x):
    return 1 / (1 + np.exp(-x))

# sets input for nn as the (x,y) locations of each node as a tuple


for x in range(1, numX + 1):
    for y in range(1, numY + 1):
        top_inputs.append((x - .5, y - .5))

# print(top_inputs)
top_outputs = []

AVG_FITNESSES = []
INDIVIDUALS = []


def eval_genomes(genomes, config):
    avgFitness = 0
    counter = 0

    # LIST OF VARIABLES FOR fitnes
    nelX = numX
    nelY = numY
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
            x = sigmoid_act(net.activate(xi)[0])
            if(x > .5):
                x = 1
            else:
                x = 0  # always passes output through sigmoid
            outList.append(x)

        # initialized as a regular list and copied as a numpy array to resize
        x = np.array(outList, copy=True)
        INDIVIDUALS.append(x)
        # x = x.reshape((numX, numY))

        # fitness function imported from topopt.py file
        fit = topopt.main(nelX, nelY, volfrac, rmin, penal, ft,
                          x)

        avgFitness = avgFitness + fit  # keeps track of average fitness for each generation
        counter = counter + 1

        genome.fitness -= fit

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


fullOutput = []
# this loop doesnt work because every evaluation needs the full
for xi in top_inputs:
    fullOutput.append(sigmoid_act(winner_net.activate(xi)[0]))

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

for i in range(NGEN * POPSIZE - POPSIZE, NGEN * POPSIZE):
    plt.ion()
    fig, ax = plt.subplots()
    x = INDIVIDUALS[i]
    im = ax.imshow(-x.reshape((numX, numY)).T, cmap='gray',
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig.show()
    raw_input("Hit anything to view next individual")


raw_input("Enter anything to plot.")

generations = []
for i in range(1, NGEN + 1):
    generations.append(i)

plt.scatter(generations, AVG_FITNESSES)
plt.title("Fitness Over Generations")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()

raw_input("Press anything to end")
