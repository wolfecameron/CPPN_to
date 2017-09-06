import math
import numpy as np
#This file will contain an extensive list of activation functions, which will then be kept as a property of each node. 

#sigmoid activation function
def sig(x):
    return (1/(1+np.exp(-x)))