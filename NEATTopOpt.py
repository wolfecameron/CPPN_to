from __future__ import print_function
import neat
import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

# NEAT implementation of topological optimization
numX = 6
numY = 4
volfrac = 0.4
top_inputs = volfrac * np.ones(numY * numX, dtype=float

# top_inputs for num in range(numX*numY):

print(top_inputs)

top_outputs=[]

# element stiffness matrix
# NOTE: Stiffness matrix is hard coded - used as approximation


def lk():
    E=1
    nu=0.3
    k=np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu /
                  8, -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    KE=E / (1 - nu**2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                     [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                     [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                     [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                     [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                     [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                     [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                     [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


def eval_genomes(genomes, config):
    avgFitness=0
    count=0
    for genome_id, genome in genomes:
		genome.fitness=0
    	net=neat.nn.FeedForwardNetwork.create(genome, config)

		x=np.array(net.activate(top_inputs)).reshape((numX, numY))

		nelx=numX
    nely=numY
    rmin=5.4
    penal=3.0
    ft=1
    # print("Minimum compliance problem with OC")
    # print("ndes: " + str(nelx) + " x " + str(nely))
    # print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    # print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    # Max and min stiffness
    Emin=1e-9
    Emax=1.0

    # dofs: (2 dof for each node, x and y)
    ndof=2 * (nelx + 1) * (nely + 1)

    # Allocate design variables (as array), initialize and allocate sens.
    # THIS IS WHERE X FROM CPPN WILL BE PASSED IN!!
    # This script initializes x's to all ones, ours will initialize with CPPN output
    # ''' X WILL BE INPUT INSTEAD OF BEING INITIALIZED WITH ALL 1S
    # x=volfrac * np.ones(nely*nelx,dtype=float) '''

    xold=x.copy()
    xPhys=x.copy()
    g=0  # must be initialized to use the NGuyen/Paulino OC approach
    dc=np.zeros((nely, nelx), dtype=float)
    # FE: Build the index vectors for the for coo matrix format.
    KE=lk()
    edofMat=np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el=ely + elx * nely
            n1=(nely + 1) * elx + ely
            n2=(nely + 1) * (elx + 1) + ely
            edofMat[el, :]=np.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * \
                                    n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
    # Construct the index pointers for the coo format
    iK=np.kron(edofMat, np.ones((8, 1))).flatten()
    jK=np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    # had to cast as int type because was crashing
    nfilter=int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH=np.zeros(nfilter)
    jH=np.zeros(nfilter)
    sH=np.zeros(nfilter)
    cc=0
    for i in range(nelx):
        for j in range(nely):
            row=i * nely + j
            kk1=int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2=int(np.minimum(i + np.ceil(rmin), nelx))
            ll1=int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2=int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col=k * nely + l
                    fac=rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0, fac)
                    cc=cc + 1

    # Finalize assembly and convert to csc format
    H=coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs=H.sum(1)

    # BC's and support
    dofs=np.arange(2 * (nelx + 1) * (nely + 1))
    fixed=np.union1d(dofs[0:2 * (nely + 1):2],
                       np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free=np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f=np.zeros((ndof, 1))
    u=np.zeros((ndof, 1))

    # Set load
    f[1, 0]=-1

    '''
    Do not need to plot for now

    # Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
    interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.show()

    '''

    # Keep compliance calculation with no loop - ONE CALCULATION FOR EACH FUNCTION CALL
    # Set loop counter and gradient vectors
    # loop=0

    change=1
    dv=np.ones(nely * nelx)  # lists of length nelx*nely
    dc=np.ones(nely * nelx)
    ce=np.ones(nely * nelx)
    # while change>0.01 and loop<2000:
    # loop=loop+1
    # Setup and solve FE problem
    sK=((KE.flatten()[np.newaxis]).T * (Emin + (xPhys)
                                          ** penal * (Emax - Emin))).flatten(order='F')
    K=coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    # Remove constrained dofs from matrix
    K=K[free, :][:, free]
    # Solve system
    u[free, 0]=spsolve(K, f[free, 0])
    # Objective and sensitivity
    ce[:]=(np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
             * u[edofMat].reshape(nelx * nely, 8)).sum(1)
    # this is the compliance?
    obj=((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
    dc[:]=(-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
    dv[:]=np.ones(nely * nelx)

    # Sensitivity filtering:
    if ft == 0:
        dc[:]=np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
    elif ft == 1:
        dc[:]=np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
        dv[:]=np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

    genome.fitness += obj
    avgFitness += genome.fitness

plt.plot(counter, avgFitness / len(genomes))
counter += 1


# Load configuration.
config=neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-topOpt')

# Create the population, which is the top-level object for a NEAT run.
p=neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run for 100 generations.
winner=p.run(eval_genomes, n=50)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net=neat.nn.FeedForwardNetwork.create(winner, config)

plt.title("Population's average and best fitness")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()

for xi in zip(top_inputs):
    output=winner_net.activate(xi)
    print("  input {!r}, got {!r}".format(xi, output))
