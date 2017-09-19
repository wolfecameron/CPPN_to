import random
import CPPNActivationFunctions as Funcs
import numpy as np

class Node: #stores the number of total nodes and the type (input,hidden,output) of each node
	def __init__(self,nodeNumber,nodeValue,nodeLayer):
		self.nodeNumber = nodeNumber
		self.nodeValue = nodeValue #all node values are 0 until CPPN is evaluated besides the inputs, these values are 
		self.nodeLayer = nodeLayer #input, output, or hidden
		self.connectingNodes = [] #stores a list of numbers for the nodes it connects to, used to avoid recurring connections 
        
class Connection: #stores node connections and corresponding weights, may be activated or not, innovation number stores gene history
	def __init__(self, nodeIn, nodeOut, weight, activationStatus, innovNum):
		self.nodeIn = nodeIn #data type of nodeIn and nodeOut is Integer - correlates to the nodes position in the nodeList
		self.nodeOut = nodeOut
		self.weight = weight 
		self.activationStatus = activationStatus #boolean value - connection will not be evaluated if false
		self.innovNum = innovNum #tracks gene history of the Genotype

        

class CPPNNode: #this node structure is actually used to mathmatically evalutate the network with inputs
	def __init__(self):
		self.arraysCreated = False #checks if numpy arrays have been created yet to evaluated CPPN
		self.CPPNConnections = [] #stores list of CPPNNodes that connect to the self node
		self.connections = [] #connections and weights list used to store values from Genotype until numpy arrays are created
		self.weights = []
		self.value = 0 #only used for input nodes to return value of input to following nodes
    
    
	def createArrays(self): #creates numpy arrays in place of the weights and connections list so one can take dot products with numpy
		self.arraysCreated = True
		if(len(self.connections) > 0):
			self.connectArr = np.empty([1,len(self.connections)]) #defines the size of the arrays with place holder values
			self.weightArr = np.empty([1,len(self.weights)])
			for i in range(0,len(self.connections)):
				self.connectArr[0,i] = 0 #adds actual values to the arrays - all connections are 0 until evaluated
				self.weightArr[0,i] = self.weights[i]
    
	def evaluateCPPN(self):
		if(self.arraysCreated == False): #checks to see if arrays have been created yet for evaluation
			self.createArrays()
		if(len(self.connections) == 0): #will return when it reaches an input node, which has 0 nodes connected to it
			return self.value
		for i in range(0,self.connectArr.size): 
			self.connectArr[0,i] = self.CPPNConnections[i].evaluateCPPN() #recurses through all the array connections
        
		#print (np.dot(self.connectArr,self.weightArr.T)[0,0])
		return Funcs.sig(np.dot(self.connectArr,self.weightArr.T)[0,0]) #dots connection values with weights to yield node value
        
    
    
class Genotype: #Genotype class contains all mutation/evolutionary method/all info about network structure
       
	def __init__(self,numIn): #takes argument for number of input nodes in CPPN
	
			#keeps track of fitness for evaluations in GA
			self.fitness = 0 
           
			#creates properties to contain all nodes/connections/information for each genotype
			self.numIn = numIn #track number of inputs into network
			self.outputIndex = numIn+1; #stores index of output node
			self.globalInnovation = 0;
			self.nodeList = [] #stores all nodes in a single list
			self.connectionList = [] #stores information for all conections within the neural network
			self.size = numIn + 2 #stores number of total nodes in the network
           
            
			#node creation Loop
			for i in range(0,numIn):
				#nodes initialized with value of zero, values created later
				self.nodeList.append(Node(i,0,"input")) #creates input nodes (number of inputs is parameterized)
            
			self.nodeList.append(Node(numIn,0,"hidden")) #networks start with a single hidden node and output node to maximize simplicity
			self.nodeList.append(Node(numIn + 1, 0, "output")) #creates output node
            
			#connectionList creation loop
			for i in range(0,numIn):
				#creates connections between the inputs and hidden node
				self.makeConnection(i,numIn,random.randint(-2,2))
              
			#creates final connection to output                         
			self.makeConnection(numIn,numIn + 1, random.randint(-2,2))
        
    
	#MUST USE THIS FUNCTION BEFORE EVALUATING CPPN IN ANY WAY
	def setInput(self, values):#sets the values of input nodes equal to the input list 'values'
		if(self.numIn!=len(values)):
			return "Number of inputs must match the length of provided input values"
            
		for i in range(0,self.numIn):
			self.nodeList[i].nodeValue = values[i]
    

	#adds a new node at the end of the nodeList and increments global size of the network
	def makeNode(self):
		self.nodeList.append(Node(self.size,0,"hidden"))
		self.size = self.size + 1
    
	'''
	#checks if connection creates recursion issue
	def checkConnection(self,indIn,indOut):
		for i in self.nodeList[indOut].connectingNodes:
			if(i == indIn):
				return False
		
		for i in self.nodeList[indIn].connectingNodes:
			if(i == indOut):
				return False
    
		return True
	'''
    
	#this function should be always used to append the connectionList
	def makeConnection(self,indIn, indOut, weight): #creates a connection and checks that it does not already exist
		check1,check2 = (indIn in self.nodeList[indOut].connectingNodes),(indOut in self.nodeList[indIn].connectingNodes)
		if((not check1) and (not check2)):
			self.connectionList.append(Connection(indIn,indOut, weight,True,self.globalInnovation))
			self.nodeList[indOut].connectingNodes.append(indIn)
			self.globalInnovation = self.globalInnovation + 1
		else:
			print check1
			print check2
			
    
	#crosses 2 different genotypes, keeps all connections unless two connections are same
	def crossover(self, parent2):
		child = Genotype(self.numIn)
		childSize = self.size if (self.size>=parent2.size) else parent2.size
		
		for i in range(0,childSize - (self.numIn + 2)): #makes child the maximum size of both parents (so all connections work properly)
			child.makeNode()
        
		for i,z in zip(self.connectionList,parent2.connectionList):
			child.makeConnection(i.nodeIn,i.nodeOut,i.weight)
			child.makeConnection(z.nodeIn,z.nodeOut,z.weight)
    
		return child
    
    
	#randomly updates the weight of a connection gene
	def pointMutate(self, mutpb):
		#the upper limit of this mutation value should be the mutation probability of the evolutationary algorithm
		mutIndex = random.randint(0,self.size-1)
		self.connectionList[mutIndex].weight = self.connectionList[mutIndex].weight + random.uniform(0,mutpb) 
    
    
	#creates a new randomly connected node in the CPPN structure
	def nodeMutate(self):
		a = random.randint(0,self.size - 1) 
		while(a==self.outputIndex):
			a = random.randint(0,self.size -1)
		b = self.size
		c = self.size
		d = random.randint(self.numIn,self.size-1)
        
		#adds a new empty node onto the end of the nodeList
		self.nodeList.append(Node(self.size,0,"hidden"))
		self.makeConnection(a, self.size,random.randint(-2,2))
		self.makeConnection(self.size,d,random.randint(-2,2))
		self.size = self.size + 1 #increments size of structure
        
          
    
    
	#randomly adds a connection into the network
	def linkMutate(self):#mutIn and mutOut should be randomly selected integers within range of nodeList
		self.makeConnection(random.randint(0,self.size-1),random.randint(self.numIn,self.size-1),random.randint(-2,2))

            
    
	#toggles the activation status of a randomly selected connection
	def disableMutate(self):
		mutIndex = random.randint(0,self.size-1)
		self.connectionList[mutIndex].activationStatus = not(self.connectionList[mutIndex].activationStatus)
    
    
    #creates a list of CPPN nodes to perform activation upon, must call this function on a genotype to evaluate the CPPN
	def getCPPNNodes(self):
		CPPNNodeList = []
		for i in range(0,self.size):
			CPPNNodeList.append(CPPNNode())
			CPPNNodeList[i].value =  0
            
		#starts all connection and weight lists as regular python lists
		for x in self.connectionList:
			indIn = x.nodeIn
			indOut = x.nodeOut
			CPPNNodeList[indOut].connections.append(self.nodeList[indIn].nodeValue)
			CPPNNodeList[indOut].CPPNConnections.append(CPPNNodeList[indIn])
			CPPNNodeList[indOut].weights.append(x.weight)
		
		return CPPNNodeList[self.outputIndex] #returns the output node of CPPN
        
        
                
   
#all code below this is just used for testing 

x = Genotype(4)
x.setInput([0,1,2,3])
x.nodeMutate()
x.nodeMutate()
k = x.getCPPNNodes()
print k.evaluateCPPN()