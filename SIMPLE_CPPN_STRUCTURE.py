import random
import CPPNActivationFunctions as Funcs
import numpy as np
import sys
from CPPNActivationFunctions import simpleAct, sig, noAct, relu, sinAct, tanhAct
import math


class Node:  # stores the number of total nodes and the type (input,hidden,output) of each node
    def __init__(self, nodeNumber, nodeValue, layerNum):

        self.nodeNumber = nodeNumber
        self.nodeValue = nodeValue  # all node values are 0 until CPPN is evaluated besides the inputs, these values are
        self.layerNum = layerNum  # input, output, or hidden
        self.connectingNodes = []  # stores a list of numbers for the nodes it connects to, used to avoid recurring connections
        self.activationKey = random.choice([2, 3, 4, 5, 6])  # denotes which activation function is used by the node

    def updateConnectingNodeWeights(self, node, weight):
        for nodeData in self.connectingNodes:
            if (nodeData[0] == node):
                nodeData[1] = weight


class Connection:  # stores node connections and corresponding weights, may be activated or not, innovation number stores gene history
    def __init__(self, nodeIn, nodeOut, weight, activationStatus, innovNum):
        self.nodeIn = nodeIn  # data type of nodeIn and nodeOut is Integer - correlates to the nodes position in the nodeList
        self.nodeOut = nodeOut
        self.weight = weight
        self.activationStatus = activationStatus  # boolean value - connection will not be evaluated if false

    # self.innovNum = innovNum  # tracks gene history of the Genotype

    def equalConnection(self, inNode, outNode):
        if (self.nodeIn == inNode and self.nodeOut == outNode):
            return True
        return False


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
        self.highestHidden = 0  # highest hidden layer currently in netwoek

        # node creation Loop
        for i in range(0, numIn):
            # nodes initialized with value of zero, values created later
            self.nodeList.append(Node(i, 0, 0))  # creates input nodes (number of inputs is parameterized)
        # layer number of output is sys.maxint to always keep it at the end of the list
        self.nodeList.append(Node(numIn, 0,
                                  sys.maxint))  # networks start with a single hidden node and output node to maximize simplicity
        self.nodeList[
            -1].activationKey = 2  # have to make sure this activation key never changes, must always be sigmoid
        # self.nodeList.append(Node(numIn + 1, 0, "output"))  # creates output node

        # connectionList creation loop
        for i in range(0, numIn):
            weight = random.uniform(-1, 1)
            # creates connections between the inputs and output (start w/0 hidden nodes)
            self.connectionList.append(Connection(i, self.numIn, weight, True, self.globalInnovation))
            # self.connectionList.append(Connection(i, self.numIn, 1, True, self.globalInnovation))
            self.nodeList[self.numIn].connectingNodes.append([self.nodeList[i], weight])

        '''
        # creates final connection to output
        self.connectionList.append(Connection(self.numIn, self.numIn+1, random.randint(-2,2), True, self.globalInnovation))
        self.nodeList[self.numIn+1].connectingNodes.append(self.numIn)
        #self.makeConnection(numIn, numIn + 1, random.randint(-2, 2))
        '''

    # toString method for genotype: prints all information
    def __str__(self):
        print "NETWORK SIZE: " + str(self.size)
        print " "

        print("NODE LIST:")
        for i in self.nodeList:
            if (i.layerNum == 0):
                layer = "input"
            if (i.layerNum == sys.maxint):
                layer = "output"
            else:
                layer = str(i.layerNum)
            activation = ["no activation", "step function", "sigmoid", "rectifer", "sine", "hyperbolic tangent"]

            print "Node Number: " + str(i.nodeNumber) + " Layer: " + layer + " Activation: " + activation[
                i.activationKey - 1]

        print " "

        print "CONNECTION LIST: "
        counter = 1
        for x in self.connectionList:
            print "Connection #" + str(counter) + ":" + " [" + str(x.nodeIn) + "]---(" + str(x.weight) + ")-->[" + str(
                x.nodeOut) + "]"
            print "Status: " + str(x.activationStatus)
            counter += 1
            print " "

        print "Number of Connections: " + str(len(self.connectionList))
        return " "

    # activate a given node with function given by the node's activation key
    def activate(self, node):
        key = node.activationKey
        if (key == 1):
            node.value = noAct(node.value)
        elif (key == 2):
            node.value = simpleAct(node.value)
        elif (key == 3):
            node.value = sig(node.value)
        elif (key == 4):
            node.value = relu(node.value)
        elif (key == 5):
            node.value = sinAct(node.value)
        elif (key == 6):
            node.value = tanhAct(node.value)
        else:
            print("ERROR: ActivationKey exceeded current number of activations.")

    # evaluates the network based on given list of inputs
    def evaluate(self, inputs):
        sortedNodes = sorted(self.nodeList, key=lambda x: x.layerNum)  # sorts node list by layer number

        for i in range(len(sortedNodes)):
            nodeVal = 0  # used to get sum of connecting nodes value
            node = sortedNodes[i]
            if (node.layerNum == 0):
                nodeVal = inputs[i]
            for other in node.connectingNodes:  # inputs will not have any connecting nodes listed
                # self.inputNotCalcError(other[0])
                nodeVal = nodeVal + (other[0].value * float(other[1]));
            # nodeVal =  nodeVal + node.connectingNodes[i][1]
            # node.value = activate(node.activation, nodeVal)
            node.value = nodeVal
            self.activate(node)
        # node.visited = True
        answer = sortedNodes[-1].value
        # sortedNodes v self.nodeList???
        for node in sortedNodes:  # sets node values back to 0 for next evaluation
            #	if(node.visited == False):
            #		print("ERROR - ALL NODES NOT EVALUATED")
            node.value = 0
        # node.visited = False
        return answer

    # creates node given hidden layer number and updates state of the network
    def makeNode(self, hidden):
        self.nodeList.append(Node(self.size, 0, hidden))
        if (self.highestHidden < hidden):
            self.highestHidden = hidden
        print(hidden)
        print(self.highestHidden)
        self.size = self.size + 1

    # node can only connect to a node with a higher layer number than its own
    def validConnection(self, indIn, indOut):
        # checks obvious bad connections first
        connectionExists = False
        for connection in self.connectionList:
            if (connection.equalConnection(indIn, indOut)):
                return False
        if (self.nodeList[indIn].layerNum < self.nodeList[indOut].layerNum):
            return True
        return False

    # always use to append the connectionList
    def makeConnection(self, indIn, indOut, weight):  # checks connection before creating

        if (self.validConnection(indIn, indOut)):
            self.connectionList.append(Connection(indIn, indOut, weight, True, self.globalInnovation))
            self.nodeList[indOut].connectingNodes.append([self.nodeList[indIn], weight])
            self.globalInnovation = self.globalInnovation + 1

    # uniform crossover for two genotypes, switches weight/activation functions of different connections (xpb is swap probability)
    # should both parents be changed in this situation?
    def crossover(self, parent2, xpb):
        # only included for debugging purposes
        if (not (len(self.connectionList) == len(parent2.connectionList))):
            print("Connection lists are not same length - something is wrong!")

        else:
            for i in range(len(self.connectionList)):
                if (random.random() >= xpb):
                    # swaps weights of parents
                    w1 = parent2.connectionList[i].weight
                    self.connectionList[i].weight = w1
                    self.nodeList[self.connectionList[i].nodeOut].updateConnectingNodeWeights(
                        self.nodeList[self.connectionList[i].nodeIn], w1)  # updates connection data for evaluation
                # parent2.connectionList[i].weight = w1

                if (random.random() >= xpb):
                    # swaps activation functions of parents
                    key1 = self.nodeList[self.connectionList[i].nodeOut].activationKey
                    self.nodeList[self.connectionList[i].nodeOut].activationKey = parent2.nodeList[
                        parent2.connectionList[i].nodeOut].activationKey
                    # parent2.nodeList[parent2.connectionList[i].nodeOut].activationKey = key1

            return self

    # mutates an individual weights in a genotype based on mutpb, returns true if mutation occurs
    def weightMutate(self, mutpb):
        # the upper limit of this mutation value should be the mutation probability of the evolutationary algorithm
        # print(len(self.connectionList))
        mutate = False
        for i in self.connectionList:
            if (random.random() < mutpb):
                mutate = True
                i.weight = i.weight + random.uniform(-mutpb, mutpb)
                self.nodeList[i.nodeOut].updateConnectingNodeWeights(self.nodeList[i.nodeIn], i.weight)
                # print(self.connectionListi.nodeOut)
        return mutate

    # mutates the activation function used in an individual node based on mutpb returns true if mutation occurs
    def activationMutate(self, mutpb):
        mutate = False
        for i in self.nodeList:
            if (random.random() < mutpb):
                mutate = True
                i.activationKey = random.choice([2, 3, 4, 5, 6])
                # print(self.connectionListi.nodeOut)
        return mutate

    # creates a new randomly connected node in the CPPN structure
    # connects two lower layer nodes to given node and node to one higher level node
    # a, b lower nodes
    # d higher node
    # layerNum is the layer the node being added will have
    # returns true if node mutate is successful
    # pre a,b < layerNum < d
    def nodeMutate(self, a, b, d, layerNum):
        self.nodeList.append(Node(self.size, 0, layerNum))

        # should implement something to have this repeat if it breaks, adding complexity must occur when needed!!!!!
        # this could be easily implemented with return values in the evolutionary code
        validChange = self.validConnection(a, self.size) and self.validConnection(self.size,
                                                                                  d) and self.validConnection(b,
                                                                                                              self.size)
        if (validChange):
            self.makeConnection(a, self.size, random.uniform(-2,
                                                             2))  # creates random weights so they are different for all individuals
            self.makeConnection(b, self.size, random.uniform(-2, 2))
            self.makeConnection(self.size, d, random.uniform(-2, 2))
            self.size += 1  # increments size of structure
            if (layerNum > self.highestHidden):
                self.highestHidden = layerNum
        else:
            self.nodeList.pop()
        return validChange

    # adds a new connection to the network, returns true if connection from a to b is valid
    def linkMutate(self, a, b):
        validChange = self.validConnection(a, b)
        if (validChange):
            self.makeConnection(a, b, random.uniform(-2, 2))
        return validChange