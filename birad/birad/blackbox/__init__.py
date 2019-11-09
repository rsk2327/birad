import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from birad import *
from .fft import *

global my_embedding

def getRatio(df):
    
    totalCount = len(df)
    
    try:
        rcount = df.Side.value_counts()['R']
        Rratio= np.round(rcount/totalCount,2)
    except:
        Rratio = 0
        
    try:
        cccount = df.View.value_counts()['CC']
        CCratio= np.round(cccount/totalCount,2)
    except:
        CCratio = 0
        
    try:
        racecount = df.Medview_Race.value_counts()['White']
        Raceratio= np.round(racecount/totalCount,2)
    except:
        Raceratio = 0
        
    return Raceratio,Rratio,CCratio



def analyzeFeature(df, truth, featureIndex, n, figsize = (10,10), createPlots = True,numCols=5, return_df = False):

    df = df.copy()
    
    df = df.sort_values(featureIndex, ascending=False)

    df = df[['filename',featureIndex]][:n]

    df['DummyID'] = df.filename.apply(lambda x : int(x.split("_")[0]))
    df['Side'] = df.filename.apply(lambda x : x.split("_")[-3])
    df['View'] = df.filename.apply(lambda x : x.split("_")[-2])
    
    df = pd.merge(df, truth[['DummyID','Medview_Race','Density_Overall']], on='DummyID', how = 'left' )
    
    print("Race - White {} | Side - R {}  | View - CC {} ".format(*getRatio(df) ))
    
    if createPlots == True:
        ##Collecting all the image file paths
        imageFileList = []

        for i in range(len(df)):

            filename = df.loc[i]['filename'] + '.jpg'

            
            imageFileList.append(getFileLocation(filename))

            

        
        numRows = n//numCols + 1
        stopIter = False
        
        plt.figure(figsize=figsize)
        
        for i in range(numRows):

            if stopIter == True:
                break

            for j in range(numCols):

                index = i*numCols + j

                if index == len(df):
                    stopIter= True
                    break


                plt.subplot(numRows, numCols, index+1)
                img = imread(imageFileList[index])
                plt.title(imageFileList[index].split("/")[-1].split(".")[0])
                plt.imshow(img, 'gray')

            
    if return_df ==True:
        return df
    else:
        return getRatio(df)
    
            
    

def generateFeatures(learn, dataset, trainFolder, validFolder):


    learn.model.eval()

    #Setting the layer from which we are extracting features
    layer = list(learn.model.children())[1][4]

    global my_embedding
    my_embedding = 0

    def copyData(m, inp, out):
        global my_embedding
        out1 = out.detach().cpu().numpy()
        my_embedding = out1

    #Registering a forward hook
    feat = layer.register_forward_hook(copyData)

    ## Train dataset
    for i in tqdm(range(len(dataset.train_ds.items))):
    
        e=dataset.one_item(dataset.train_ds.x[i])
        pred = learn.model(e[0])
        
        filename = dataset.train_ds.items[i].split("/")[-1].split(".")[0]
        
        np.save( os.path.join(trainFolder, filename+'.npy'), my_embedding )

    ## Valid dataset
    for i in tqdm(range(len(dataset.valid_ds.items))):
    
        e=dataset.one_item(dataset.valid_ds.x[i])
        pred = learn.model(e[0])
        
        filename = dataset.valid_ds.items[i].split("/")[-1].split(".")[0]
        
        np.save( os.path.join(validFolder, filename+'.npy'), my_embedding )