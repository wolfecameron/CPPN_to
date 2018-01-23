from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

#converts rgb pixel values to binary (1 == black, 0 == white)
def convertBinary(tupList):
	binList = []
	for x in tupList:
		if(x[0] > 155):
			binList.append(0)
		else:
			binList.append(1)
	return binList

#filepath should be a string containing relative filepath to desired image
def getPixels(filepath, numX,numY):
	#declares prefered size of image
	SIZE = (numX,numY)

	#opens image from its source file (original size)
	try:
		im_tmp = Image.open(filepath)
	except:
		#if image cannot open, function exits
		print("ERROR: File Path was not valid.")
		return

	#resizes image to preferred size
	im = im_tmp.resize(SIZE)

	#im.show()
	#raw_input("check image")

	#im = im.rotate(90)

	#list containing rgb tuples with original pixel values from image
	og_pixels = list(im.getdata())

	#contains binary pixel values - created from original pixel list
	binary_pixels = convertBinary(og_pixels)

	return binary_pixels

#pre: inputted list is list of binary pixels, numX*numY must match length of binPixels
#graphs input pixels using matplotlib
def graphImage(binPixels, numX, numY):
	#creates a numpy array from inputted binary list for graphing purposes
	x = np.array(binPixels, copy = True)
	plt.ion()
	fig,ax = plt.subplots()
	im = ax.imshow(-x.reshape(numX, numY), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
	fig.show()
	input("Here is your plot!")

	
