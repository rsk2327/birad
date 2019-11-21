import pandas as pd 
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import os

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




#### DATASET CREATION FUNCTIONS ####

def getBMIData(aa, white, bmi_min, bmi_max):
    
    aaSub = aa[(aa.BMI>=bmi_min) & (aa.BMI<bmi_max)]
    whiteSub = white[(white.BMI>=bmi_min) & (white.BMI<bmi_max)]

    maxSamples = min(len(whiteSub), len(aaSub))
    
    if maxSamples==0:
        return pd.DataFrame()
    
    
    #Subsetting to keep only maxSamples number of samples in each subset
    aaSub = aaSub.sample(n=maxSamples, replace = False)
    whiteSub = whiteSub.sample(n=maxSamples, replace = False)
    
    #Figuring out the number of samples in train/valid
    numTrain = int(0.8*maxSamples)
    numValid = maxSamples - numTrain
    
    
    
    aaTrain = np.random.choice(aaSub.DummyID, numTrain,replace = False)
    whiteTrain = np.random.choice(whiteSub.DummyID, numTrain,replace = False)
    
    aaValid = np.array(list(set(aaSub.DummyID.values).difference(set(aaTrain))))
    whiteValid = np.array(list(set(whiteSub.DummyID.values).difference(set(whiteTrain))))
    
    d =  pd.concat([
        pd.DataFrame({'DummyID':aaTrain,'train':False}),
        pd.DataFrame({'DummyID':whiteTrain,'train':False}),
        pd.DataFrame({'DummyID':aaValid,'train':True}),
        pd.DataFrame({'DummyID':whiteValid,'train':True})
    ])
    
    print("Max Samples : {} numTrain  : {} df len : {}".format(maxSamples, numTrain, len(d)))
    
    return d
    
    

def createBMIDataset(bmi_buckets = [0,20,30,40,50,55,60,100]):
    
    patientList = []
    fullFileList = []

    inputFolder1 = '/home/santhosr/Documents/Birad/ProcessedData/FullRes'
    truthFile1 = '/home/santhosr/Documents/Birad/birad_targetFile.csv'

    inputFolder2 = '/home/santhosr/Documents/Birad/ProcessedData/PennExtra_3500/'
    truthFile2 = '/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv'

    df1 = pd.read_csv('/home/santhosr/Documents/Birad/birad_targetFile.csv')
    df1.drop(['PresIntentType','DBT'],inplace = True,axis=1)


    df2 = pd.read_csv('/home/santhosr/Documents/Birad/RaceDL_ExtraCaucasian.csv')
    df2.Medview_Race = 'White'

    ## Removing IDs from df2 which are already present in df1
    idList = list(df1.DummyID.values)
    df2 = df2[~df2.DummyID.isin(idList)]

    truth = pd.concat([df1,df2],sort=True)

    ## Reading from set 1
    for i in range(1,5):

        folder = os.path.join(inputFolder1,str(i))
        fileList = os.listdir(folder)
        fileList = [os.path.join('FullRes',str(i),x) for x in fileList]
        fullFileList = fullFileList + fileList
#         print(len(fileList))

        patientList = patientList + [int(x.split("/")[-1].split("_")[0]) for x in fileList]

    patientList1 = patientList.copy()
    ## Reading from set 2
    print(len(patientList))

    fileList= os.listdir(inputFolder2)
    fileList = [os.path.join('PennExtra_3500',x) for x in fileList]
    d = pd.DataFrame(fileList)
    d[1] = d[0].apply(lambda x : int(x.split("/")[1].split("_")[0]))
    d = d[d[1].isin(df2.DummyID.values)]
    fileList = list(d[0].values)
    fullFileList += list(d[0].values)

    patientList += [int(x.split("/")[-1].split("_")[0]) for x in fileList]
    print(len(patientList))

    patientList2 = patientList.copy()

    #Retaining only the patients with 4 views
    k=pd.Series(patientList).value_counts().reset_index()
    patientList = k[k[0]==4]['index'].values
    print("total number of patients",len(patientList))

    patientList = np.array(list(set(patientList)))
    df = pd.DataFrame({'DummyID':patientList})
    df = pd.merge(df,truth, how='left')
    df1 = df1.copy()
    df = df.drop_duplicates(subset=['DummyID'])

    #Creates equal number of patients from White and AA groups
    white = df[df.Medview_Race=='White']
    AA = df[df.Medview_Race=='African American']

    
    outputDf = pd.DataFrame()

    for i in range(len(bmi_buckets)-1):
        out = getBMIData(AA,white, bmi_buckets[i], bmi_buckets[i+1])

        outputDf = pd.concat([outputDf, out])
        
    temp = pd.DataFrame(fullFileList)
    temp.columns = ['filename']

    temp['DummyID'] = temp.filename.apply(lambda x : int(x.split("/")[-1].split("_")[0]))

    trainTemp = temp[temp.DummyID.isin(outputDf[outputDf.train==False].DummyID.values)]
    validTemp = temp[temp.DummyID.isin(outputDf[outputDf.train==True].DummyID.values)]
    
    trainTemp['train'] = False
    validTemp['train'] = True

    df= pd.concat([trainTemp, validTemp], sort = True)

    #Shuffling data
    index = list(range(len(df)))
    np.random.shuffle(index)
    df = df.iloc[index]
    
    return df
    
    

    
    
    
    
    