import random
import CPPNActivationFunctions as Funcs
import numpy as np
import sys
from CPPNActivationFunctions import simpleAct
import math


class Node:  # stores the number of total nodes and the type (input,hidden,output) of each node
	def __init__(self, nodeNumber, nodeValue, layerNum):
		
		self.nodeNumber = nodeNumber
		self.nodeValue = nodeValue  # all node values are 0 until CPPN is evaluated besides the inputs, these values are
		self.layerNum = layerNum  # input, output, or hidden
		self.connectingNodes = []  # stores a list of numbers for the nodes it connects to, used to avoid recurring connections
		self.visited = False #scratch variable for connection checking
		self.activationFunction = random.choice( [1, 2,3, 4,5])

class Connection:  # stores node connections and corresponding weights, may be activated or not, innovation number stores gene history
	def __init__(self, nodeIn, nodeOut, weight, activationStatus, innovNum):
		self.nodeIn = nodeIn  # data type of nodeIn and nodeOut is Integer - correlates to the nodes position in the nodeList
		self.nodeOut = nodeOut
		self.weight = weight
		self.activationStatus = activationStatus  # boolean value - connection will not be evaluated if false
		self.innovNum = innovNum  # tracks gene history of the Genotype

	def equalConnection(self, inNode, outNode):
		if (self.nodeIn == inNode and self.nodeOut == outNode):
			return True
		return False

class CPPN:  # this node structure is actually used to mathmatically evalutate the network with inputs
	def __init__(self):
		self.arraysCreated = False  # checks if numpy arrays have been created yet to evaluated CPPN
		self.CPPNConnections = []  # stores list of CPPNs that connect to the self node
		self.connections = []  # connections and weights list used to store values from Genotype until numpy arrays are created
		self.weights = []
		self.value = 0  # only used for input nodes to return value of input to following nodes

	def createArrays(self):  # creates numpy arrays in place of the weights and connections list so one can take dot products with numpy
		self.arraysCreated = True
		if (len(self.connections) > 0):
			self.connectArr = np.empty(
				[1, len(self.connections)])  # defines the size of the arrays with place holder values
			self.weightArr = np.empty([1, len(self.weights)])
			for i in range(0, len(self.connections)):
				self.connectArr[0, i] = 0  # adds actual values to the arrays - all connections are 0 until evaluated
				self.weightArr[0, i] = self.weights[i]

	def evaluateCPPN(self):
		if (self.arraysCreated == False):  # checks to see if arrays have been created yet for evaluation
			self.createArrays()
		if (len(
				self.connections) == 0):  # will return when it reaches an input node, which has 0 nodes connected to it
			return self.value
		for i in range(0, self.connectArr.size):
			self.connectArr[0, i] = self.CPPNConnections[
				i].evaluateCPPN()  # recurses through all the array connections
		# print (np.dot(self.connectArr,self.weightArr.T)[0,0])
		x = np.dot(self.connectArr, self.weightArr.T)
		return simpleAct(x)  # dots connection values with weights to yield node value

class Genotype:  # Genotype class contains all mutation/evolutionary method/all info about network structure

	def __init__(self, numIn):  # takes argument for number of input nodes in CPPN

		# creates properties to contain all nodes/connections/information for each genotype
		self.numIn = numIn  # track number of inputs into network
		self.outputIndex = numIn;  # stores index of output node
		self.globalInnovation = 0
		self.nodeList = []  # stores all nodes in a single list
		self.connectionList = []  # stores information for all conections within the neural network
		self.size = numIn + 1  # stores number of total nodes in the network
		self.fitness = 0
		self.highestHidden = 0

		# node creation Loop
		for i in range(0, numIn):
			# nodes initialized with value of zero, values created later
			self.nodeList.append(Node(i, 0, 0))  # creates input nodes (number of inputs is parameterized)

		self.nodeList.append(Node(numIn, 0, sys.maxint))  # networks start with a single hidden node and output node to maximize simplicity
		#self.nodeList.append(Node(numIn + 1, 0, "output"))  # creates output node

		# connectionList creation loop
		for i in range(0, numIn):
			# creates connections between the inputs and output (start w/0 hidden nodes)
			self.connectionList.append(Connection(i, self.numIn, random.uniform(-1, 1), True, self.globalInnovation))
			#self.connectionList.append(Connection(i, self.numIn, 1, True, self.globalInnovation))
			self.nodeList[self.numIn].connectingNodes.append(i)
			
		'''
		# creates final connection to output
		self.connectionList.append(Connection(self.numIn, self.numIn+1, random.randint(-2,2), True, self.globalInnovation))
		self.nodeList[self.numIn+1].connectingNodes.append(self.numIn)
		#self.makeConnection(numIn, numIn + 1, random.randint(-2, 2))
		'''
		
		
	#toString method for genotype: prints all information 
	def __str__(self):
		print "NETWORK SIZE: " + str(self.size)
		print " " 
		
		print("NODE LIST:")
		for i in self.nodeList:
			print "Layer " + str( i.layerNum) + " --> " + str(i.nodeNumber)
		
		print " " 
		
		
		print "CONNECTION LIST: "
		counter = 1
		for x in self.connectionList:
			print "Connection #" + str(counter) + ":" + " [" + str(x.nodeIn) + "]---(" + str(x.weight) + ")-->[" + str(x.nodeOut) + "]" 
			print "Status: " + str(x.activationStatus)
			counter = counter + 1
			print " " 
		
		
		print "Number of Connections: " + str(len(self.connectionList))
		return " " 
	
	
	# MUST USE THIS FUNCTION BEFORE EVALUATING CPPN IN ANY WAY
	def inputValues(self, values):  # sets the values of input nodes equal to the input list 'values'
		if (self.numIn != len(values)):
			return "Number of inputs must match the length of provided input values"

		for i in range(0, self.numIn):
			self.nodeList[i].nodeValue = values[i]

			
			
    # adds a new node at the end of the nodeList and increments global size of the network
	def makeNode(self, hidden):
		self.nodeList.append(Node(self.size, 0, hidden))
		if(self.highestHidden < hidden):
			self.highestHidden = hidden
		self.size = self.size + 1

		
	
	#checks if any path exists between nodes, BFS algorithm	
	def validConnection(self, indIn, indOut):
		#checks obvious bad connections first
		connectionExists = False
		for connection in self.connectionList:
			if(connection.equalConnection(indIn, indOut)):
				connectionExists = True
		if(self.nodeList[indIn].layerNum < self.nodeList[indOut].layerNum):
			return True
		return False
		
		
	#always use to append the connectionList
	def makeConnection(self, indIn, indOut, weight):  #checks connection before creating 
		
		if(self.validConnection(indIn, indOut)):
			self.connectionList.append(Connection(indIn, indOut, weight, True, self.globalInnovation))
			self.nodeList[indOut].connectingNodes.append(indIn)
			self.globalInnovation = self.globalInnovation + 1
			
				
			
	# crosses 2 different genotypes, keeps all connections unless two connections are same
	def crossover(self, parent2):
#		print("Doing crossover");
		#child = Genotype(self.numIn)
		#childSize = self.size if (self.size >= parent2.size) else parent2.size

		#for i in range(0, childSize - self.numIn + 2):  # makes child the maximum size of both parents (so all connections work properly)
		#	child.makeNode()

		for i  in range(len( self.connectionList)):
			point = random.random()
			newWeight = self.connectionList[i].weight
			if (point >= 0.5):
				newWeight = parent2.connectionList[i].weight
			#child.makeConnection(i.nodeIn, i.nodeOut, float(i.weight))
			self.connectionList[i].weight = newWeight
		return self

		
		
		
	# randomly updates the weight of a connection gene
	def pointMutate(self):
		# the upper limit of this mutation value should be the mutation probability of the evolutationary algorithm
		#print(len(self.connectionList))
		mutIndex = random.randint(0, len(self.connectionList)-1)
		self.connectionList[mutIndex].weight = self.connectionList[mutIndex].weight + random.uniform(-1, 1)

		# randomly updates the weight of a connection gene
	def weightMutate(self, muxpb):
		# the upper limit of this mutation value should be the mutation probability of the evolutationary algorithm
		# print(len(self.connectionList))
		mutate = False
		for i in self.connectionList:
			rand = random.random()
			if(rand < muxpb):
				mutate = True
				i.weight = i.weight + random.uniform(-.25, .25)
		return mutate
	
	
	# creates a new randomly connected node in the CPPN structure
	def nodeMutate(self, a, b,  d, layerNum):
	#	a = random.randint(0, self.size - 1)
	#	d = random.randint(self.numIn, self.size - 1)
		
		# adds a new empty node onto the end of the nodeList
#		print("length node list")
#		print(len(self.nodeList))
		self.nodeList.append(Node(self.size, 0, layerNum))
#		print(len(self.nodeList))
		#print(len(self.nodeList))
		validChange = self.validConnection(a, self.size) and self.validConnection(self.size,d) and self.validConnection(b, self.size)
		if(validChange):
			self.makeConnection(a, self.size, random.uniform(-2, 2))
			self.makeConnection(b, self.size, random.uniform(-2, 2))
			self.makeConnection(self.size, d, random.uniform(-2, 2))
			self.size = self.size + 1  # increments size of structurei
		else:
			self.nodeList.pop()
		return validChange

		
	# randomly adds a connection into the network
	def linkMutate(self, a, b):  # mutIn and mutOut should be randomly selected integers within range of nodeList
		changeValid = self.validConnection(a, b) 
		if changeValid:
			self.makeConnection(a,b,random.uniform(-2, 2))
		
			
		return changeValid
		

	#e toggles the activation status of a randomly selected connection
	def disableMutate(self):
		eprint(len(self.connectionList))
		mutIndee = random.randint(0, len(self.connectionList)-1)
		sele.connectionList[mutIndex].activationStatus = (not (self.connectionList[mutIndex].activationStatus))

		
	#e creates a list of CPPN nodes to perform activation upon, must call this function on a genotype to evaluate the CPPN
	def getCPPNNodes(self):
		CPPNList = []
		for i in range(self.size):
			CPPNList.append(CPPN())
			CPPNList[i].value = self.nodeList[i].nodeValue

		# starts all connection and weight lists as regular python lists
		for x in self.connectionList:
			indIn = x.nodeIn
			indOut = x.nodeOut
			CPPNList[indOut].connections.append(self.nodeList[indIn].nodeValue)
			CPPNList[indOut].CPPNConnections.append(CPPNList[indIn])
			CPPNList[indOut].weights.append(x.weight)

		return CPPNList[self.outputIndex]  # returns the output node of CPPN


# all code below this is just used for testing
'''
x = Genotype(4)
for i in range(50):
	x.nodeMutate()
x.makeConnection(0,4, 10)
x.inputValues([0,1,2,3])
CPPN = x.getCPPNNodes()


print(x)

print(CPPN.evaluateCPPN())
'''
