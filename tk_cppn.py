import tkinter as tk
import numpy as np
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.animation as animation

LARGE_FONT = ("Verdana", 12)



f = Figure(figsize=(5,5),dpi=100)
a = f.add_subplot(111) #one-by-one and it is chart number one

def animate(i):
    pullData = open("/Users/cameronwolfe/Desktop/py_notebooks/sample.txt","r").read()
    dataList = pullData.split("\n")
    xList = []
    yList = []
    for each in dataList:
        x,y = each.split(',')
        xList.append(int(x))
        yList.append(int(y))
    #want to clear data each time so that the points do not add up every time they are changed
    a.clear()
    g = np.floor(np.random.random((100, 100)) + .5)
    a.imshow(g, cmap='Greys')
    
        
#creates class for starting page
#inherits from Frame class
class GUIFrame(tk.Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label = Label(text = "Graph Page", font=LARGE_FONT)
        #easiest to use pack when you're only including a few things in the window
        label.pack(pady=10,padx=10)
        self.master.title("CPPN Playground")

        quit_button = Button(self.master, text ="Quit", command = self._quit)
        quit_button.pack(side = BOTTOM)
        
        #f = Figure(figsize=(5,5),dpi=100)
        #a = f.add_subplot(111)
        #a.plot([1,2,3],[1,2,3])

        canvas = FigureCanvasTkAgg(f,self.master)
        canvas.show()
        canvas.get_tk_widget().pack()

    #root can be accessed with master, this is used to exit out of page
    def _quit(self):
        self.master.quit()
        self.master.destroy()


def main():
    #initialize main window
    root = Tk()
    root.geometry("500x300+200+100")
    #initialize frame
    app = GUIFrame()
    #link animation to figure, pass in the animation function, and specify an update interval in milliseconds
    ani = animation.FuncAnimation(f,animate, interval = 1000)
    root.mainloop()



if __name__ == '__main__':
    main()