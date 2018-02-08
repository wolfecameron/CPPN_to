import numpy as np
from scipy.special import expit
from scipy.signal import gaussian
#Tkinter
#This file will contain an extensive list of activation functions, which will then be kept as a property of each node. 


#no activation function (all nodes initialized with this)
def noAct(x):
	return x

#simple threshold activation
def simpleAct(x):
	if(x>.5):
		return 1;
	else:
		return 0;

#sigmoid activation function
def sig(x):
    return expit(x)
		
#ReLU activation function (linear if greater than 0, else 0)
def relu(x):
	return 0 if x<0 else x
	
#sin function (using numpy)	
def sinAct(x):
	return np.sin(x)
	
#tanh function (using numpy)
def tanhAct(x):
	return np.tanh(x)

#tangent activation function
def tanAct(x):
	return np.tan(x)

#gaussian activation function
def gauss(x):
	return x

#logistic activation function
#pre: x > 0
def log(x):
	if(not (x > 0)):
		return 0
	return np.log(x)

#exponential activation function
def exp(x):
	return np.exp(x)

#square activation function
def square(x):
	return x**2



	
	
