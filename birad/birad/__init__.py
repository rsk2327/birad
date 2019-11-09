import pandas as pd 
import numpy
from imageio import imread
import matplotlib.pyplot as plt


global truth


def setTruthFile(df):
	global truth
	truth = df.copy()


def getRaceLabel(x,binary=False):
    """
    Given a filename, returns the race of the patient
    African American : 0
    White : 1
    Rest : 2

    For the function to work, the truth file needs to be set 
    using the setTruthFile method
    """
    ID = x.split("/")[-1].split("_")[0]
    label = truth[truth.DummyID == int(ID)]['Medview_Race'].values[0]

    if label == 'African American':
        return 0
    elif label == "White":
        return 1
    else:
        return 2


def getFileLocation(filename):

    #Removing file extensions
    filename = filename.split(".")[0]

    fileLocation = pd.read_csv('/home/santhosr/Documents/Birad/fileLocation.csv')
    
    try:
        location = fileLocation.loc[fileLocation.filename==filename].values[0][0]
        
        return location
    except:

        raise ValueError('Filename doesnt exist in fileLocation csv')


def plotImageData(filename, figsize = (10,10)):
    
    img = getImageData(filename)
    
    plt.figure(figsize = figsize)
    plt.imshow(img,'gray')
    
    
def getImageData(filename):
    
    #Removing file extensions
    filename = filename.split(".")[0]
        
    fileLocation = pd.read_csv('/home/santhosr/Documents/Birad/fileLocation.csv')
    
    try:
        img = imread(fileLocation.loc[fileLocation.filename==filename].values[0][0])
        
        return img
    except:
        raise ValueError('Filename doesnt exist in fileLocation csv')
    
    
    
    
    