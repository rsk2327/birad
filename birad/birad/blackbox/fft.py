
import pandas as pd 
import numpy as np
from scipy import fftpack
from birad import *

def plot_spectrum(im_fft, figsize = (10,10)):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.figure(figsize = figsize)
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()



def getImageFFT(filename):

	try:
		img = getImageData('75425032_L_MLO_1')
	except:
		raise ValueError('Inccorrect filename')

	im_fft = fftpack.fft2(img)

	return im_fft

def plotImageFFT(filename, figsize):

	im_fft = getImageFFT(filename)

	plot_spectrum(im_fft, figsize)




