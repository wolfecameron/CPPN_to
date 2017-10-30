import numpy as np
import random
from deap import algorithms, base, creator, tools
from topopt import main as fitness
import matplotlib.pyplot as plt
from matplotlib import colors

NUMX = 180
NUMY = 60
VOLFRAC = .4
RMIN = 5.4
PENAL = 3
FT = 1

TOTAL_ELEMENTS = NUMX * NUMY


def evalWeights(individual):

    x = np.array(individual, copy=True)
    return (fitness(NUMX, NUMY, VOLFRAC, RMIN, PENAL, FT, x),)



# configures all settings for DEAP framework
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
tb = base.Toolbox()
tb.register("bit", random.randint, 0, 1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n=TOTAL_ELEMENTS)
tb.register("population", tools.initRepeat, list, tb.individual, n=50)
tb.register("evaluate", evalWeights)
tb.register("mate", tools.cxTwoPoint)
tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
tb.register("select", tools.selTournament, tournsize=5)
tb.register("map", map)


# creates population and evolves it
cxpb, mutpb, ngen = .05, .05, 200
pop = tb.population()
pop = algorithms.eaSimple(pop, tb, cxpb, mutpb, ngen)

winner = pop[0][0]
for x in range(len(winner)):
    if(winner[x] < .4):
        winner[x] = 0
    else:
        winner[x] = 1

print(winner)

raw_input("Enter anything to plot result")
# Initialize plot and plot the initial design
x = np.array(winner, copy=True)
plt.ion()  # Ensure that redrawing is possible
fig, ax = plt.subplots()
im = ax.imshow(-x.reshape((NUMX, NUMY)).T, cmap='gray',
               interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
fig.show()

raw_input("It's over!")
