import tkinter as tk
import numpy as np
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.animation as animation

LARGE_FONT = ("Verdana", 12)



#f = Figure(figsize=(5,5),dpi=100)
#a = f.add_subplot(111) #one-by-one and it is chart number one

def animate(i, genome, numX, numY, f, a):
    pullData = open("genome_info.txt", "r").read()
    dataList = pullData.split("\n")
    foundConnections = False
    nodeCounter = 0
    connectionCounter = 0

    for line in dataList:
        if(not foundConnections and "Activation" in line):
            activationValue = int(line[12:])
            #print(activationValue)
            genome.nodeList[nodeCounter].activationKey = activationValue
            nodeCounter += 1
        if(foundConnections and "Weight" in line):
            weightValue = float(line[8:])
            #print(weightValue)
            genome.connectionList[connectionCounter].weight  = weightValue
            #now must update weight in nodeList
            n_out = genome.nodeList[genome.connectionList[connectionCounter].nodeOut]
            n_in = genome.nodeList[genome.connectionList[connectionCounter].nodeIn]
            #searches connecting nodes and updates weight for correct input node
            for node_data in n_out.connectingNodes:
                if(node_data[0] == n_in):
                    node_data[1] = weightValue
            connectionCounter += 1

        elif("CONNECTIONS" in line):
            foundConnections = True
    
    """CREATES LIST OF INPUTS TO RUN NETWORK"""
    inputs = []
    #creates input values for CPPN for spring optimization
    for x in range(1, numX + 1):
        inputs.append(x)

    tmp = np.array(inputs, copy = True)
    MEAN = np.mean(tmp)
    STD = np.std(tmp)

    #list of normalized inputs
    normIn = [] 

    #creates input list with normalized vectors, values of input are of form (x,y) in a list of tuples
    for y in range(0,numY):
        for x in range(0,numX):
            tup = (np.fabs(x - MEAN)/STD, np.fabs(y-MEAN)/STD)
            normIn.append(tup)
    """*******************"""

    output_list = []
    for x in range(len(normIn)):
        output_list.append(genome.evaluate([normIn[x][0],normIn[x][1]])[0])

    #creates numpy array and resizes it for graphing
    g = np.array(output_list, copy=True)

    #want to clear data each time so that the points do not add up every time they are changed
    a.clear()

    g = np.reshape(g, (numX,numY))
    a.imshow(g, cmap='Greys')
    
        
#creates class for starting page
#inherits from Frame class
class GUIFrame(tk.Frame):
    def __init__(self, f):
        super().__init__()
        self.initUI(f)

    def initUI(self, f):
        label = Label(text = "Graph Page", font=LARGE_FONT)
        #easiest to use pack when you're only including a few things in the window
        label.pack(pady=10,padx=10)
        self.master.title("CPPN Playground")

        quit_button = Button(self.master, text ="Quit", command = self._quit)
        quit_button.pack(side = BOTTOM)

        canvas = FigureCanvasTkAgg(f,self.master)
        canvas.show()
        canvas.get_tk_widget().pack()

    #root can be accessed with master, this is used to exit out of page
    def _quit(self):
        self.master.quit()
        self.master.destroy()


def main(g_param, numX, numY, f, a):
    #initialize main window
    root = Tk()
    root.geometry("500x300+200+100")
    #initialize frame
    app = GUIFrame(f)
    #link animation to figure, pass in the animation function, and specify an update interval in milliseconds
    ani = animation.FuncAnimation(f,lambda x: animate(1000, g_param, numX, numY, f, a))
    root.mainloop()



if __name__ == '__main__':
    main()