import random
import CPPNActivationFunctions as Funcs
import numpy as np
import sys
from CPPNActivationFunctions import noAct, simpleAct, sig, relu, sinAct, tanhAct, tanAct, gauss, log, exp, square
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from tkinter import *
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from tk_cppn import main
import os


# global variable for genome construction
# decides whether to create genome from scratch or pull data from file
# second global variable holds filepath to genome info
PULL_DATA = False
GENOME_FILE = "/home/wolfecameron/Desktop/fullGenome3"

class Node:  # stores the number of total nodes and the type (input,hidden,output) of each node
	def __init__(self, nodeNumber, nodeValue, layerNum):
		
		self.nodeNumber = nodeNumber
		self.nodeValue = nodeValue  # all node values are 0 until CPPN is evaluated besides the inputs, these values are
		self.layerNum = layerNum  # input, output, or hidden
		self.connectingNodes = []  # stores a list of numbers for the nodes it connects to, used to avoid recurring connections
		self.activationKey = random.choice([3]) #random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) #denotes which activation function is used by the node

	def updateConnectingNodeWeights(self, node, weight):
		for nodeData in self.connectingNodes:
			if(nodeData[0] == node):
				nodeData[1] = weight
				#returns immediately upon finding node to update
				return

class Connection:  # stores node connections and corresponding weights, may be activated or not, innovation number stores gene history
	def __init__(self, nodeIn, nodeOut, weight, activationStatus):
		self.nodeIn = nodeIn  # data type of nodeIn and nodeOut is Integer - correlates to the nodes position in the nodeList
		self.nodeOut = nodeOut
		self.weight = weight
		self.activationStatus = activationStatus  # boolean value - connection will not be evaluated if false
		#self.innovNum = innovNum  # tracks gene history of the Genotype

	def equalConnection(self, inNode, outNode):
		if (self.nodeIn == inNode and self.nodeOut == outNode):
			return True
		
		return False


class Genotype:  # Genotype class contains all mutation/evolutionary method/all info about network structure
	
	# pull data is the boolean argument that decides whether to pull data from file or create from scratch
	def __init__(self, numIn, numOut):  # takes argument for number of input nodes and output nodes in CPPN

		# creates properties to contain all nodes/connections/information for each genotype
		self.numIn = numIn  # track number of inputs into network
		self.numOut = numOut
		self.outputIndex = numIn;  # stores index of output node
		self.nodeList = []  # stores all nodes in a single list
		self.connectionList = []  # stores information for all conections within the neural network
		self.size = numIn + numOut  # stores number of total nodes in the network
		self.fitness = 0
		self.highestHidden = 0 #highest hidden layer currently in network

		if(PULL_DATA):
			self.size = 0
			pullData = open(GENOME_FILE, "r").read()
			dataList = pullData.split("\n")
			pastNodes = False
			# find all node data and append to node list
			# then find all connection data and 
			for d in dataList:
				if("CONNECTIONS" in d):
					pastNodes = True
				elif(not pastNodes):
					self.makeNode(int(d))
					if(int(d) < sys.maxsize):
						self.nodeList[self.size - 1].activationKey = 3
					else:
						self.nodeList[self.size - 1].activationKey = 2

				elif(pastNodes and len(d) > 0):
					splitData = d.split(',')
					self.makeConnection(int(splitData[0]), int(splitData[1]), float(splitData[2]))

		#regular constructor
		else:
			# node creation Loop
			for i in range(0, numIn):
				# nodes initialized with value of zero, values created later
				self.nodeList.append(Node(i, 0, 0))  # creates input nodes (number of inputs is parameterized)
			
			#layer number of outputs is sys.maxsize to always keep it at the end of the list
			for x in range(numOut):
				self.nodeList.append(Node(numIn + x, 0, sys.maxsize))  # networks start with input nodes and output nodes (no hidden) to maximize simplicity
				self.nodeList[-1].activationKey = 2 #have to make sure this activation key never changes, must always be sigmoid
			#self.nodeList.append(Node(numIn + 1, 0, "output"))  # creates output node

			# connectionList creation loop
			#connects all inputs fully to outputs
			for i in range(0, numIn):
				for x in range(numOut):
					weight = random.uniform(-1, 1)
					# creates connections between the inputs and output (start w/ hidden nodes)
					self.connectionList.append(Connection(i, self.numIn + x, weight, True))
			
					self.nodeList[self.numIn + x].connectingNodes.append([self.nodeList[i], weight])
		
		
	#toString method for genotype: prints all information 
	def __str__(self):
		print("NETWORK SIZE: " + str(self.size))
		print(" ") 
		
		print("NODE LIST:")
		for i in self.nodeList:
			if(i.layerNum == 0):
				layer = "input"
			if(i.layerNum == sys.maxsize):
				layer = "output"
			else:
				layer = str(i.layerNum)
			activation = ["no activation", "simple activation", "sigmoid", "relu", "sine", "hyperbolic tangent" , "tangent", "gaussian", "log", "exponential", "square"]

			print("Node Number: " + str(i.nodeNumber) + " Layer: " + layer +  " Activation: " + activation[i.activationKey - 1])
		
		print(" ") 
		
		
		print("CONNECTION LIST: ")
		counter = 1
		for x in self.connectionList:
			print("Connection #" + str(counter) + ":" + " [" + str(x.nodeIn) + "]---(" + str(x.weight) + ")-->[" + str(x.nodeOut) + "]")
			print("Status: " + str(x.activationStatus))
			counter += 1
			print(" ") 
		
		
		print("Number of Connections: " + str(len(self.connectionList)))
		return " " 
	

	#activate a given node with function given by the node's activation key
	def activate(self, node):
		key = node.activationKey
		if(key == 1):
			node.value = noAct(node.value)
		elif(key == 2):
			node.value = simpleAct(node.value)
		elif(key == 3):
			node.value = sig(node.value)
		elif(key == 4):
			node.value = relu(node.value)
		elif(key == 5):
			node.value = sinAct(node.value)
		elif(key == 6):
			node.value = tanhAct(node.value)
		elif(key == 7):
			node.value = tanAct(node.value)
		elif(key == 8):
			node.value = gauss(node.value)
		elif(key == 9):
			node.value = log(node.value)
		elif(key == 10):
			node.value = exp(node.value)
		elif(key == 11):
			node.value = square(node.value)
		else:
			print("ERROR: ActivationKey exceeded current number of activation functions.")

	#evaluates the network based on given list of inputs
	def evaluate(self, inputs):
		sortedNodes = sorted(self.nodeList,key = lambda x: x.layerNum) #sorts node list by layer number

		outputs = []

		for i in range(len(sortedNodes)):
			nodeVal = 0 #used to get sum of connecting nodes values
			node = sortedNodes[i]
			if(node.layerNum == 0):
				nodeVal = inputs[i]
			for other in node.connectingNodes: #inputs will not have any connecting nodes listed
				
				nodeVal += (other[0].value * float(other[1])); #multiplies node value by its weight
				
			node.value = nodeVal
			self.activate(node)
		
		#adds all output values to the list of outputs
		for x in range(len(sortedNodes) - self.numOut, len(sortedNodes)):
			outputs.append(sortedNodes[x].value)


		for node in sortedNodes: #sets node values back to 0 for next evaluation
			node.value = 0
		
		return outputs

	#creates node given hidden layer number and updates state of the network
	def makeNode(self, hidden):
		self.nodeList.append(Node(self.size, 0, hidden)) #nodeValue of 0 and layer given by "hidden" parameter
		if(self.highestHidden < hidden):
			self.highestHidden = hidden
		#print(hidden)
		#print(self.highestHidden)
		self.size = self.size + 1


	#node can only connect to a node with a higher layer number than its own
	def validConnection(self, indIn, indOut):
		#checks obvious bad connections first
		goodConnection = True
		for connection in self.connectionList:
			if(connection.equalConnection(indIn, indOut)):
				goodConnection = False
				return goodConnection
		#ensures all connections go upward in term of layer number
		if(self.nodeList[indIn].layerNum < self.nodeList[indOut].layerNum):
			return goodConnection
		else:
			goodConnection = False
			return goodConnection
		
		
	#always use to append the connectionList
	def makeConnection(self, indIn, indOut, weight):  #checks connection before creating 
		if(self.validConnection(indIn, indOut)):
			self.connectionList.append(Connection(indIn, indOut, weight, True))
			self.nodeList[indOut].connectingNodes.append([self.nodeList[indIn], weight])
			
			
				
			
	#uniform crossover for two genotypes, switches weight/activation functions of different connections (xpb is swap probability)
	#should both parents be changed in this situation?
	def crossover(self, parent2, xpb):
		crossover = False
		if(not(len(self.connectionList) == len(parent2.connectionList))):
			print("Connection lists are not same length - something is wrong!")
		
		else:
			for i  in range(len(self.connectionList)):
				if (random.random() <= xpb):
					#swaps weights of parents
					w1 = parent2.connectionList[i].weight
					self.connectionList[i].weight = w1
					self.nodeList[self.connectionList[i].nodeOut].updateConnectingNodeWeights(self.nodeList[self.connectionList[i].nodeIn],w1) #updates connection data for evaluation
					crossover = True
					#parent2.connectionList[i].weight = w1

				if(random.random() <= xpb):
					#swaps activation functions of parents for both nodeIn and nodeOut of connection
					#key1 = self.nodeList[self.connectionList[i].nodeOut].activationKey
					self.nodeList[self.connectionList[i].nodeOut].activationKey = parent2.nodeList[parent2.connectionList[i].nodeOut].activationKey
					self.nodeList[self.connectionList[i].nodeIn].activationKey = parent2.nodeList[parent2.connectionList[i].nodeIn].activationKey
					crossover = True

			
			return self

	#mutates an individual weights in a genotype based on mutpb, returns true if mutation occurs
	def weightMutate(self, mutpb):
		# the upper limit of this mutation value should be the mutation probability of the evolutationary algorithm
		mutate = False
		for i in self.connectionList:
			if(random.random() <= mutpb):
				i.weight += random.uniform(-mutpb, mutpb)
				self.nodeList[i.nodeOut].updateConnectingNodeWeights(self.nodeList[i.nodeIn], i.weight)
				mutate = True
		return mutate

	#mutates the activation function used in an individual node based on mutpb returns true if mutation occurs
	def activationMutate(self, mutpb):
		mutate = False
		for i in self.nodeList:
			#activation of outputs should not be changed from sigmoid
			if (random.random() <= mutpb and not(i.layerNum == sys.maxsize)):
				mutate = True
				i.activationKey = random.choice([3])#random.choice([2,3,4,5,6,7,8,9,10,11])
		return mutate


	# creates a new randomly connected node in the CPPN structure
	# connects two lower layer nodes to given node and node to one higher level node
	# a, b lower nodes
	#d higher node
	# layerNum is the layer the node being added will have
	#returns true if node mutate is successful
	#pre a.layerNum,b.layerNum < layerNum < d.layerNum
	def nodeMutate(self, a, b,  d, layerNum):
		self.nodeList.append(Node(self.size, 0, layerNum))

		validChange = self.validConnection(a, self.size) and self.validConnection(self.size,d) and self.validConnection(b, self.size)
		if(validChange):
			self.makeConnection(a, self.size, random.uniform(-2, 2)) #creates random weights so they are different for all individuals
			self.makeConnection(b, self.size, random.uniform(-2, 2))
			self.makeConnection(self.size, d, random.uniform(-2, 2))
			self.size +=1  # increments size of structure
			if (layerNum > self.highestHidden):
				self.highestHidden = layerNum
		else:
			#deletes node if the connections are bad - can try again without adding unconnected nodes
			self.nodeList.pop()
		return validChange

	#adds a new connection to the network, returns true if connection from a to b is valid
	def linkMutate(self, a, b):
		validChange = self.validConnection(a, b)
		if(validChange):
			self.makeConnection(a, b, random.uniform(-2,2))
		return validChange


	#helper function to graph output of the CPPN
	def graphOutput(self,outList, numX, numY):
			if(not(len(outList) == numX * numY)):
				print("Error: Length of Output does not match x and y dimensions.")
			else: 
				plt.ion()
				x = np.array(outList, copy = True)
				fig,ax = plt.subplots()
				im = ax.imshow(-x.reshape(numX, numY), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
				fig.show()

	def graphGenotype(self):
		possibleColors = ['#C0C0C0', '#000000', '#FF0000', '#FFFF00', '#808000', '#00FF00', '#00FFFF', '#0000FF','#FF00FF', '#008080', '#800080']
		node_dict = {}
		sort_nodes = sorted(self.nodeList, key = lambda x: x.layerNum)
		layerCount = 0
		nodeCount = 0
		foundOutput = False
		for x in sort_nodes:
			
			if(x.layerNum > layerCount and not foundOutput):
				layerCount += 1
				if(x.layerNum == sys.maxsize):
					foundOutput = True
			
			if(layerCount not in list(node_dict.keys())):
				node_dict[layerCount] = []
			
			#add current node to list of nodes at that layer
			node_dict[layerCount].append(x)
		
		pointDict = {}

		pointsX = []
		pointsY = []
		graphColors = []
		keys = list(node_dict.keys())
		#creates positions for all nodes on the graph
		for x in range(len(keys)):
			nodeCounter = 0
			for i in node_dict[keys[x]]:
				#hashes all new points to existing node numbers
				pointDict[i.nodeNumber] = (x,nodeCounter)
				act = i.activationKey
				graphColors.append(possibleColors[act-1])
				pointsX.append(x)
				pointsY.append(nodeCounter)
				nodeCounter += 1
		
		plt.scatter(pointsX,pointsY, color = graphColors, s = 400)
		#illusrates all connections
		for x in self.connectionList:
			currLine = []
			inNode = x.nodeIn
			outNode = x.nodeOut
			point1 = pointDict[inNode]
			point2 = pointDict[outNode]
			xList = [point1[0], point2[0]]
			yList = [point1[1], point2[1]]
			plt.plot(xList,yList)

		#creates legend for activation colors
		patch1 = mpatches.Patch(color='#C0C0C0', label='No Act')
		patch2 = mpatches.Patch(color='#000000', label='Step')
		patch3 = mpatches.Patch(color='#FF0000', label='Sigmoid')
		patch4 = mpatches.Patch(color='#FFFF00', label='Relu')
		patch5 = mpatches.Patch(color='#808000', label='Sine')
		patch6 = mpatches.Patch(color='#00FF00', label='Tanh')
		patch7 = mpatches.Patch(color='#00FFFF', label='Tan')
		patch8 = mpatches.Patch(color='#0000FF', label='Gaussian')
		patch9 = mpatches.Patch(color='#FF00FF', label='Logarithmic')
		patch10 = mpatches.Patch(color='#008080', label='Exponential')
		patch11 = mpatches.Patch(color='#800080', label='Sqaure')
		plt.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8, patch9, patch10, patch11], loc='upper right')
		plt.show()


	#writes all info from a genome into a text file so that it can be used later
	def writeGenomeInfo(self, filenumber):
		if(filenumber > 0):
			filename = "genome_info" + str(filenumber) + ".txt"
		else:
			filename = "genome_info.txt"

		with open(filename, "w") as genome_file:
			genome_file.write("NODES\n")
			for x in self.nodeList:
				genome_file.write("Node #" + str(x.nodeNumber) + ", Node Layer: " + str(x.layerNum) + "\n")
				genome_file.write("Activation: " + str(x.activationKey) + "\n")

			genome_file.write("CONNECTIONS\n")
			num_connection = 0
			for x in self.connectionList:
				genome_file.write("Connection #: " + str(num_connection) + ", NodeIn: " + str(x.nodeIn) + ", NodeOut: " + str(x.nodeOut) + "\n")
				genome_file.write("Weight: " + str(x.weight) + "\n")
				num_connection += 1

	# function that writes all info needed to reconstruct CPPN to a file
	# finds free filename and creates/writes to this file
	# writes all info from genome, allows genome to be later reconstructed and examined
	def writeFullGenomeInfo(self, filepath):
		baseFileName = "fullGenome"
		fileCounter = 0
		# finds file name that is not already used to created file
		while(os.path.isfile(filepath + "/" + baseFileName + str(fileCounter))):
			fileCounter += 1
		# holds string value for full filepath that is not being used
		fullPath = filepath + "/" + baseFileName + str(fileCounter)
		with open(fullPath, "w") as file:
			# only needed parameter for adding a node to list is the layer number
			for n in self.nodeList:
				file.write(str(n.layerNum) + "\n")

			file.write("CONNECTIONS\n")
			# adding a connection required nodeIn, nodeOut, and weight
			for c in self.connectionList:
				file.write(str(c.nodeIn) + "," + str(c.nodeOut) + "," + str(c.weight) + "\n")

	# main function that runs the playground for a instance of the genome
	def playground(self, numX, numY):
		self.graphGenotype()

		f = Figure(figsize=(5,5),dpi=100)
		a = f.add_subplot(111) #one-by-one and it is chart number one

		#writes all info from current file into a genome so it can be used for animation and real time updating of the CPPN
		self.writeGenomeInfo(0)

		
		#initiates GUI
		main(self,numX,numY, f , a)
		

if __name__ == '__main__':
	# generate inputs
	inputs = []
	#creates input values for CPPN for spring optimization
	for x in range(1, 50 + 1):
		inputs.append(x)

	tmp = np.array(inputs, copy = True)
	MEAN = np.mean(tmp)
	STD = np.std(tmp)

	#list of normalized inputs
	normIn = [] 

	#creates input list with normalized vectors, values of input are of form (x,y) in a list of tuples
	for y in range(0,50):
		for x in range(0,50):
			tup = ((x - MEAN)/STD, (y-MEAN)/STD, 1) #include the 1 for bias
			normIn.append(tup)

	curr = Genotype(2,1)

	outputs = []
	for x in range(len(normIn)):
		outputs.append(curr.evaluate([normIn[x][0],normIn[x][1],normIn[x][2]])[0])


	
	curr.graphOutput(outputs,50,50)
	input("Here it is")