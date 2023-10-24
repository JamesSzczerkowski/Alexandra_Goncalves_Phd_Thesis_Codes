# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:34:31 2023

@author: jays_
"""

#%% close all and clear all 
     # close all dones work with spyder
#      get_ipython().magic('reset -sf')
#%%
''' # Instructions of modules to install ("pip install")
# numpy
        pip install numpy
# pandas  
        pip install pandas      
# czifile
        pip install czifile
# scipy  
        pip install scipy
# skimage
        pip install scikit-image
# shapely
        pip install scikit-image
# cv2    
        pip install opencv-python
        
        
Note:    
    These installs will be added to a certain place, which is not on PATH
    eg:
        WARNING: The scripts czi2tif.exe and czifile.exe are installed in 'C:\\Users\jszczerkowski\AppData\Roaming\Python\Python310\Scripts' which is not on PATH.
        Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
    
    to add path - 
    copy    "

                if 'C:\\Users\\jszczerkowski\AppData\Roaming\Python\Python310\Scripts' not in sys.path:
                    sys.path.append("C:\\Users\\jszczerkowski\AppData\Roaming\Python\Python310\Scripts")
                    paths = sys.path   
                    # print(paths)
 
            "
    into the code       
    
After installed please restart Kernel (or the program), to activate updates
    
'''

# at work
#AddPath = 'C:\\Users\jszczerkowski\OneDrive - Loyola University Chicago\Programming\Python\Python_Functions'

# at home PC
AddPath = 'D:\\OneDrives\OneDrive\Programming\Python_Functions'

# at home
#AddPath = 'D:\\OneDrives\OneDrive - Loyola University Chicago\Programming\Python\Python_Functions'
# personal 
#AddPath = 'C:\\Users\jszczerkowski\OneDrive\Programming\Python_Functions'


import sys
if AddPath not in sys.path:
    sys.path.append(AddPath)
    paths = sys.path
if 'C:\\Users\\jszczerkowski\AppData\Roaming\Python\Python310\Scripts' not in sys.path:
    sys.path.append("C:\\Users\\jszczerkowski\AppData\Roaming\Python\Python310\Scripts")
    paths = sys.path   
import selectFilesOI 
import tkinter
from tkinter import filedialog
import numpy as np  # Matrix arrays
import pandas as pd # Data storeage
from IPython import get_ipython
from matplotlib import pyplot as plt

# Pillow Libary great from basic image opening and operations   -  Not so use full with [:,:,C,Z,T], Images with cahnnels, Z-stacks and Time-frames
#from PIL import Image as tifopen  #  if error when installing use Pillow instead of PIL 'pip install Pillow'

from IPython import get_ipython # this allow for images to be plotted in new window - this can make them iteractive 

from numpy import random

# skimage move advanced 
from skimage import io as skio
from skimage import img_as_float, img_as_ubyte
from skimage.filters import roberts, sobel, scharr, prewitt, try_all_threshold, threshold_otsu
from skimage.filters.rank import entropy
from skimage.feature import canny
#import skimage.viewer as skview


### Image Filters
from scipy import ndimage as nd 
from scipy.spatial.distance import cdist 


from scipy.optimize import curve_fit
from scipy.signal import convolve2d as conv2

from skimage.restoration import denoise_nl_means, estimate_sigma 




#%% --- Start of custom Def / functions

from Segment import *
from selectFilesOI import *
from customFunctionsCheck import *

    
#--- End of custom Def / functions

#%% --- Python settings
get_ipython().run_line_magic('matplotlib', 'qt5')




#%% --- Input images
                 
   # 
#root = tkinter.Tk()
#root.withdraw() #use to hide tkinter window
#file_path_variable = search_for_file_path()  



# personal 
#file_path_variable = "D:\OneDrives\OneDrive\Programming\XanaImages"
# Personal laptop 
#file_path_variable = "C:\\Users\jszczerkowski\OneDrive - Loyola University Chicago\Jamie\Xana\Xana_Images\ReconCells"
#file_path_variable = "C:\\Users\jszczerkowski\OneDrive\Programming\TestImages"
# at home PC
file_path_variable = "C:\\Users\jays_\OneDrive - Loyola University Chicago\Jamie\Xana\Xana_Images\ReconCells"
# at work 
#file_path_variable = "C:\\Users\jszczerkowski\OneDrive - Loyola University Chicago\Jamie\Xana\Xana_Images"


SubFolder = selectFilesOI.listSubfolders(file_path_variable, ["\\Images\\", "\\Images"])

# finds and selects relevent files from parent path (input 1) that contain images with suffix {'.tif', '.tiff', '.czi', '.nd2'} (CZI and ND2 functionality need to be added)
# Input 2 is possible folder names to be excluded (this includes the folder and all subsequent folders within) 
#print(SubFolder[len(SubFolder)-1].relaventFiles[0])

#%% Find any mismatched images in folder
from selectFilesOI import Organisor
desiredOrder = ['X', 'Y', 'Z', 'C', 'T', 'B', 'V', 'H']
Image_dimentions = []
ixx=0
for FolderID in range(len(SubFolder)):
    ix=0
    Image_dimentions.append(Organisor(Folder=SubFolder[FolderID].Name))
    SetsFound = 0
    Image_dimentions[FolderID].__addField__(SetsFound = dict({'ID':0, 'Name' : 'No files found'}))
    for files in SubFolder[FolderID].relaventFiles:
        NewSetFound = 0 
        Arrange = selectFilesOI.getRearrangementfrom_CZI_MetaData(SubFolder[FolderID].Dir + "\\" + files)
        SetString = ''
        keysList = list(Arrange[1].keys())
        for b in desiredOrder:
            for a in keysList:
                if a == b:
                    if Arrange[1][a] > 0:
                       SetString = SetString+a+"-"+str(Arrange[1][a])+" "
        SetString = SetString[0:len(SetString)-1]  
        if ixx==0:
            Image_dimentions[FolderID].SetsFound.update({'Starting_Set' : SetString})
        if ix==0:
            Image_dimentions[FolderID].SetsFound.update({'ID' : [ix+1]})
            Image_dimentions[FolderID].SetsFound.update({'Name' : [files]})
            Image_dimentions[FolderID].SetsFound.update({'Set' : [SetString]})                        
        else:
            if SetString not in Image_dimentions[FolderID].SetsFound['Set']:
                Image_dimentions[FolderID].SetsFound['ID'].append(ix+1) 
                Image_dimentions[FolderID].SetsFound['Name'].append(files)
                Image_dimentions[FolderID].SetsFound['Set'].append(SetString)
                #eval("Image_dimentions[FolderID].__addField__(Options_for_"+a+"="+str(Arrange[1][a])+")")  
                # how do add field to Organiser _ with eval (adaptibale String Input)
        ix=ix+1
        
# From relevant files 
#%%
#%%
#from descartes import PolygonPatch
#import shapely.geometry as geometry
#from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
from skimage import data, filters, measure, morphology 

#%%
from scipy import ndimage as nd 
from scipy.signal import medfilt
from scipy import ndimage as nd 
from Segment import blurFun
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize, remove_small_objects
import importlib


MaxProject = 1 
Ch_Protusions = [0,1,1,0] 
ChFunction = ['Nuc','Cyto','Cyto','Dapi']
ChName = ['Ki67' , 'Tomato', 'SOI', 'NucMask']
Order = ['Dapi', 'Nuc', 'Cyto']
ChOrder = list()
for Or in Order:
   ChOrder = ChOrder+[a for a, x in enumerate(ChFunction) if x == Or]
   
   
Cn = ChName
ValsOI = ['IntSum','IntMean']  
ValOverlaps = [[1,3,5,4,6,8,9],
                [Cn[0] , Cn[1], Cn[2], Cn[0]+' '+Cn[1] , Cn[0]+' '+Cn[2] , Cn[1]+' '+Cn[2] , Cn[0]+' '+Cn[1]+' '+Cn[2]] ,
                [Cn[0] , Cn[1], Cn[2], [Cn[0], Cn[1]] , [Cn[0],  Cn[2]] , [Cn[1], Cn[2]] , [Cn[0], Cn[1], Cn[2]] ]      ]
#%%   
ChFunction = ['Nuc','Cyto','Nuc','Dapi']
MasterResults = dict()
ParentFolders=list()
for i in range(len(SubFolder)): 
    if len(SubFolder[i].relaventFiles)>0:
        
        ChName[2] = SubFolder[i].Parent[0:4]
        
        
        
        if SubFolder[i].Parent not in ParentFolders:
            ParentFolders.append(SubFolder[i].Parent)
            MasterResults.update({SubFolder[i].Parent:dict()})  
            MasterResults[SubFolder[i].Parent].update({'Children': list()})       
        MasterResults[SubFolder[i].Parent]['Children'].append(SubFolder[i].Name) 
        MasterResults[SubFolder[i].Parent].update({SubFolder[i].Name: dict()}) 
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Average Per Scene Summary' : dict()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Average Concat Summary' : dict()})
        for i3 in ValOverlaps[1]:
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({i3:dict()})
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Total' : list()}) 
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Percentage' : list()}) 
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Files' : list()})
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'].update({i3 : dict()})
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({i3 : dict()})
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'].update({i3 : dict()})
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({'Count' : list()})            
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({'Percentage' : list()})             
            
            for i4 in ValsOI:
                for Ch in enumerate(Cn):
                    if ChFunction[Ch[0]] != 'Dapi':
                        MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({i4+' '+Cn[Ch[0]] : list()})
                        MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Concat '+i4+' '+Cn[Ch[0]] : list()})
                        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'][i3].update({i4+' '+Cn[Ch[0]] : list()})
                        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'][i3].update({i4+' '+Cn[Ch[0]] : list()})
                        
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Count' : list()}) 
            MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Percentage' : list()}) 
            
#%%
'''
# Deal with error in file (eg mismatch cahnnels missed)

## This removes Current Folder to be started again 

i = Folder
if len(SubFolder[i].relaventFiles)>0:
    if SubFolder[i].Parent not in ParentFolders:
        ParentFolders.append(SubFolder[i].Parent)
        MasterResults.update({SubFolder[i].Parent:dict()})  
        MasterResults[SubFolder[i].Parent].update({'Children': list()})       
    MasterResults[SubFolder[i].Parent]['Children'].append(SubFolder[i].Name) 
    MasterResults[SubFolder[i].Parent].update({SubFolder[i].Name: dict()}) 
    MasterResults[SubFolder[i].Parent][SubFolder[i].Name]
    MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Average Per Scene Summary' : dict()})
    MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Average Concat Summary' : dict()})
    for i3 in ValOverlaps[1]:
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({i3:dict()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Total' : list()}) 
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Percentage' : list()}) 
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name].update({'Files' : list()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'].update({i3 : dict()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({i3 : dict()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'].update({i3 : dict()})
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({'Count' : list()})            
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'].update({'Percentage' : list()})             
        
        for i4 in ValsOI:
            for Ch in enumerate(Cn):
                if ChFunction[Ch[0]] != 'Dapi':
                    MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({i4+' '+Cn[Ch[0]] : list()})
                    MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Concat '+i4+' '+Cn[Ch[0]] : list()})
                    MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Per Scene Summary'][i3].update({i4+' '+Cn[Ch[0]] : list()})
                    MasterResults[SubFolder[i].Parent][SubFolder[i].Name]['Average Concat Summary'][i3].update({i4+' '+Cn[Ch[0]] : list()})
                    
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Count' : list()}) 
        MasterResults[SubFolder[i].Parent][SubFolder[i].Name][i3].update({'Percentage' : list()}) 



# remove error file, make sure to save with pickel 
import pickle
os.chdir(file_path_variable)
file = open('MasterResults_SavePython_Error', 'wb')
pickle.dump(MasterResults, file)
file.close()
# Load
file = open('MasterResults_SavePython_Error', 'rb')
ResultsLoaded = pickle.load(file)

# Look at this file 
print(SubFolder[Folder].relaventFiles[Scene])
# File removed from current list 
SubFolder[Folder].relaventFiles.remove(SubFolder[Folder].relaventFiles[Scene])
#%%    
#'''
'''




#%% Check you are have the Right Start and End File Numbers 
ParentFolderNames = list()
StartFile =  0 # 18
EndFile =    len(SubFolder) # 24
for Folder in range(StartFile,EndFile):
   if len(SubFolder[Folder].relaventFiles)>0:
        if  SubFolder[Folder].Parent not in ParentFolderNames:
            ParentFolderNames.append(SubFolder[Folder].Parent)
            print('          1st instance of ' , SubFolder[Folder].Parent)
        print(Folder , SubFolder[Folder].Parent, SubFolder[Folder].Name)    
        if Folder+1 < EndFile:
            if  SubFolder[Folder+1].Parent not in ParentFolderNames:
               print('Run Save Results') 
        else:
            print('Last scene Run Save Results')
            
print('\n',ParentFolderNames,'\n')    

print('\n\n')    
for Folder in range(0,len(SubFolder)):
    print(Folder , SubFolder[Folder].Parent, SubFolder[Folder].Name)   
''' 

AllParentFolder = list();
for Folder in range(0,len(SubFolder)):
   if len(SubFolder[Folder].relaventFiles)>0:
        if  SubFolder[Folder].Parent not in AllParentFolder:
            AllParentFolder.append(SubFolder[Folder].Parent)
           # print('          1st instance of ' , SubFolder[Folder].Parent)   
           
FolderNames = ['ARX', 'Chromagrainin', 'Gcg', 'Ins', 'Ngn3_N1', 'Ngn3_N2', 'Ngn3_N3', 'Nkx22', 'Pax4', 'Sox4']
if len(FolderNames) < len(AllParentFolder):
    print(len(FolderNames) , len(AllParentFolder))
    MisMatchinname
    
RunAnalysisOrResultSorting = 2
import datetime
StartFile = 18 # 0
EndFile =   len(SubFolder)  #  len(SubFolder)
ixx=1; CheckStatusAt=EndFile-StartFile; ParentFolderNames = list();
    
if RunAnalysisOrResultSorting == 1:
    
    for Folder in range(StartFile,EndFile):
      if len(SubFolder[Folder].relaventFiles)>0:
        ChName[2] = SubFolder[Folder].Parent[0:4]
        if ChName[2] in ['Ins4' , 'Gcg4']:
            ChFunction = ['Nuc','Cyto','Cyto','Dapi'] 
        else:
            ChFunction = ['Nuc','Cyto','Nuc','Dapi']
        if  SubFolder[Folder].Parent not in ParentFolderNames:
            ParentFolderNames.append(SubFolder[Folder].Parent)
            print('          1st instance of ' , SubFolder[Folder].Parent)    
        if Folder >= np.round((len(SubFolder)/CheckStatusAt)*ixx) or Folder==EndFile-1:
            print('\n --- Folder Completion', round((100/CheckStatusAt)*ixx),'% ',(Folder+1),'/',EndFile,' ---   ',  datetime.datetime.now())
            print('       Folder: ', Folder+1 , SubFolder[Folder].Parent,'\n SubFolder: ', SubFolder[Folder].Name)
            ixx=ixx+1        
    
        Scenes =  len(SubFolder[Folder].relaventFiles);
        ixxx=0; CheckStatusAt2=Scenes;
        for Scene in range(Scenes):
            if Scene >= np.round((Scenes/CheckStatusAt2)*ixxx) or Scene==Scenes-1:
                print('     --- Scenes Completed', round((100/CheckStatusAt2)*ixxx),'% ',(Scene+1),'/',Scenes,' ---     ',datetime.datetime.now())
                print('     -- Scene -- ', SubFolder[Folder].relaventFiles[Scene]   )
                ixxx=ixxx+1
                
            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Files'].append(SubFolder[Folder].relaventFiles[Scene])
            plt.close("all")
            Image = selectFilesOI.OpenImage_suffixDependant_asNumPy(SubFolder[Folder], Scene)
            Images = dict()
            
            Mask = np.zeros(Image.shape[0:2])
            for Ch in range(Image.shape[3]):
                if MaxProject == 1 and Image.shape[2]>1: 
                    Im = Image[:,:,0:Image.shape[2],0,0]
                if MaxProject == 2 and Image.shape[2]>1:   
                    for Z in range(Image.shape[2]):
                        np.std(Image[:,:,Z,0,0])
                        Imshow(Image = Im, Title = "Im", InteractiveWindow = True)
                Images.update({ChName[Ch] : {'Original' : Image[:,:,0,Ch,0]}})
            
            for Ch in  enumerate(ChOrder):
                print('Making mask of', ChName[Ch[1]], 'with function ',ChFunction[Ch[1]])
                Im = Images[ChName[Ch[1]]]['Original'].astype('int32')    
                Mask, BKG = removelocalBackgorund(Im)    
                #Imshow(Image = Mask, Title = "Mask - "+'Chan'+str(Ch[1]+1)+"  Org", InteractiveWindow = True)  
                #Imshow(Image = BKG, Title = "BKG - "+'Chan'+str(Ch[1]+1)+"  Org", InteractiveWindow = True)              
                Mask =  Mask.astype('bool')
                Mask = remove_small_objects(Mask, min_size = 100) 
                #Imshow(Image = Mask, Title = "Mask - "+'Chan'+str(Ch[1]+1)+"  Org", InteractiveWindow = True)  
                if ChFunction[Ch[1]] != 'Dapi':
                    Mask2 = erosion(Mask,disk(2))
                    Mask2 = remove_small_objects(Mask2, min_size = 100) 
                    Mask2 = dilation(Mask2,disk(3))
                    Mask2[Mask==0]=0
                    Mask = Mask2.copy()
                    #Imshow(Image = Mask, Title = "Mask - "+'Chan'+str(Ch[1]+1)+"  Org", InteractiveWindow = True)      
                Im = BKG  
                ImXX , Mask  = sizeIntensity_Check(Im, Mask, 50, 0.5) 
                Im = Images[ChName[Ch[1]]]['Original'].astype('int32') 
                Im[Mask>0]=0
                Im = ReduceIntensityVariability(Im,Mask)
                Images[ChName[Ch[1]]].update({'Mask' : Mask})
                Images[ChName[Ch[1]]].update({'MaskedInt' : Im})
                Images[ChName[Ch[1]]].update({'BKG' : BKG})
                
                  
            #%
            CellMask = Images[ChName[Ch[1]]]['MaskedInt'].astype('int32') ;  CellMask[:]=0
            for Ch in enumerate(ChOrder):  
              #print('\n\n   --- Label of', ChName[Ch[1]], 'with function ',ChFunction[Ch[1]],' ---\n')
              if ChFunction[Ch[1]] == 'Dapikjhkhjk':
                    Im =   Images[ChName[Ch[1]]]['Original'].astype('int32')
                    Imx =  Images[ChName[Ch[1]]]['MaskedInt'].astype('int32')
                    Mask = Images[ChName[Ch[1]]]['Mask'].copy() 
                    Im2 = convFun2serperateTouchingObjects(Mask,Im,Imx)  
                    Seperated = expand2Boundary(label(Im2.astype('bool')), Mask, disk(1), 50, 'inside')
                    Seperated2 = dilation(Seperated, disk(1))
                    Seperated3 = abs( dilation(abs(Seperated - (np.max(Seperated)+1))) - (np.max(Seperated)+1))
                    Seperated2[Seperated==0] = 0  
                    Seperated3[Seperated==0] = 0
                    Seperated4 = Seperated.copy()
                    Seperated4[:]=0
                    Seperated4[Seperated2 != Seperated]=1
                    Seperated4[Seperated3 != Seperated]=1
                    Seperated4[Seperated==0]=0
                    Seperated[Seperated4>0]=0
                    Im[Seperated>0]=0

                    Images[ChName[Ch[1]]].update({'Label' : Seperated})    
                    #Seperated = randomiseLabledImage(Seperated)
                    #Imshow(Image = randomiseLabledImage(Seperated), Title = "Seperated - "+'Chan'+str(Ch[1]+1)+"  Masked Region", InteractiveWindow = True)    

              if ChFunction[Ch[1]] == 'Nuc':
                  Mask = splitMasks(Images[ChName[Ch[1]]]['Mask'])           
                  Mask = checkIsNuclear(Mask.copy(),Images[ChName[Ch[1]]]['BKG'].copy(), Images['NucMask']['BKG'].copy())
                  Images[ChName[Ch[1]]].update({'Label' : label(Mask)}) 
                  CellMask[Mask>0]=Mask[Mask>0] 
              if ChFunction[Ch[1]] == 'Cyto':
                  Mask = splitMasks(Images[ChName[Ch[1]]]['Mask'])
                  Images[ChName[Ch[1]]].update({'Label' : Mask})  
                  CellMask[Mask>0]=Mask[Mask>0] 
                  Im = Images[ChName[Ch[1]]]['Original'].astype('int32')
                  Im[Mask>0]=0 
                  Prop = regionprops_table (Images[ChName[Ch[1]]]['Mask'], properties=('area', 'MinorAxisLength' , 'BoundingBox' )) 
   
            CellMask = np.zeros(Im.shape)
            ChanAdd = 1; Chans = 0;
            for Ch in enumerate(ChOrder):  
              #%
              if ChFunction[Ch[1]] != 'Dapi':
               Chans=Chans+1 
               Label = Images[ChName[Ch[1]]]['Label'].astype('int32')
               Label[Label>0]=ChanAdd;
               CellMask = CellMask + Label
               ChanAdd= ChanAdd+2
 
            TotalMask =  label(CellMask.astype('bool'))
            t = erosion(TotalMask,disk(2))
            
            Unique = np.unique(np.ravel(CellMask[CellMask>0]))
            CellMask[t==0]=0
            for i in Unique:
                Temp = CellMask.copy()
                Temp[Temp!=i] = 0 
                Temp2 = remove_small_objects(Temp.astype('bool'), min_size = 100)
                CellMask[np.logical_and(Temp2 == 0,  CellMask==i)] = 0

            t = opening(CellMask,disk(1)) 
            for i in Unique:
                Temp = t.copy()
                Temp[Temp!=i] = 0 
                Temp2 = remove_small_objects(Temp.astype('bool'), min_size = 50)
                t[np.logical_and(Temp2 == 0,  CellMask==i)] = 0
            t2 = label(t.astype('bool')) 
            t2 = expand2Boundary(t2, TotalMask.astype('bool'), disk(1), 10, 'inside') 
            t2 = seperateTouchingObjects(t2)
            t = expand2Boundary(t, t2.astype('bool'), disk(1), 10, 'inside') 
            MaxV = np.max(t2); additionalFound = 0;
            Var = np.zeros([MaxV,6])
            CellProps = regionprops_table (t2, properties=('area','label', 'BoundingBox' ))
            MeanArea = np.mean(CellProps['area'])+np.std(CellProps['area'])           
            Results = dict()
            Results.update({'RegionData' :  dict()})
            Results['RegionData'].update({'Area' : list()})
            Results['RegionData'].update({'Label' : list()})
            Results['RegionData'].update({'Overlap' : list()})
            Results['RegionData'].update({'Overlap_ID' : list()})
            for Ch in enumerate(ChOrder):  
              if ChFunction[Ch[1]] != 'Dapi':
                  Results.update({ChName[Ch[1]] : dict()})
                  Results[ChName[Ch[1]]].update({'IntSum' : list()})
                  Results[ChName[Ch[1]]].update({'IntMean' : list()})

            AddCells = 0;     
            for Cell in range(1, MaxV):
                Px = t2==Cell+1
                Uniq = np.unique(t[Px])
                Uniq = np.delete(Uniq,[x2 for x2,x in enumerate(Uniq) if x==0])
                if len(Uniq)==1:
                    Results['RegionData']['Area'].append(CellProps['area'][Cell])
                    Results['RegionData']['Label'].append(Cell+1)
                    Results['RegionData']['Overlap'].append(ValOverlaps[1][np.where(ValOverlaps[0]==Uniq)[0][0]])
                    Results['RegionData']['Overlap_ID'].append(np.where(ValOverlaps[0]==Uniq)[0][0])
                    for Ch in enumerate(ChOrder):  
                      if ChFunction[Ch[1]] != 'Dapi':
                          Results[ChName[Ch[1]]]['IntSum'].append(np.sum(Images[ChName[Ch[1]]]['BKG'][t2 == Cell+1]))
                          Results[ChName[Ch[1]]]['IntMean'].append(np.mean(Images[ChName[Ch[1]]]['BKG'][t2 == Cell+1]))
                    #sdfsdfsdf
                elif len(Uniq)==0:  
                    Error_has_occured
                else: ## uniqe contain more than 1 
                    XY = [CellProps['BoundingBox-0'][Cell], CellProps['BoundingBox-2'][Cell], CellProps['BoundingBox-1'][Cell], CellProps['BoundingBox-3'][Cell]]    
                    ta = t[XY[0]:XY[1], XY[2]:XY[3]].copy();
                    tb = t2[XY[0]:XY[1], XY[2]:XY[3]].copy();
                    ta[tb!=Cell+1]=0
                    tb[tb!=Cell+1]=0  
                    Px2 = tb>0
                    WholeSize = np.sum(Px2)
                    Percentage = np.zeros(len(Uniq)); Area = np.zeros(len(Uniq))
                    for i2 in enumerate(Uniq):
                        Area[i2[0]] = np.sum(ta==i2[1])
                        Percentage[i2[0]] = Area[i2[0]] / WholeSize 
                    LargeCells =  np.where(Area > MeanArea)[0]
                    if len(LargeCells)>0 or WholeSize >  MeanArea*2:
                        
                        for i3 in Uniq:
                            tc = ta.copy();
                            tc[tc!=i3]=0
                            tc = erosion(tc, disk(1))
                            tc = remove_small_objects(tc.astype('bool'), min_size = 100)
                            tc = dilation(tc, disk(1))
                            ta[ta==i3]=0
                            if np.max(tc)>0:
                                tc = RunOpenIm2DetectCells(tc.copy(),tc.copy(),5,3,100,round(MeanArea/4))  
                                tc[tc>0]=i3
                                ta = ta + tc 
                        ta = seperateTouchingObjects(ta)   
                        tc = label(erosion(ta,disk(1)).astype('bool'))
                        val  = np.zeros(np.max(tc))
                        for i3 in range(np.max(tc)):
                            val[i3] = np.unique(ta[tc==i3+1])
                        tc = expand2Boundary(tc, ta.astype('bool'), disk(1), 20, 'inside') 
                        tc = seperateTouchingObjects(tc)   
                        ta = tc.copy()
                        for i3 in range(np.max(tc)):
                            ta[tc==i3+1] = val[i3]
                        add2 = np.max(tc)   
                        tc[tc>0] = tc[tc>0] + MaxV + AddCells
                        Uniq = np.unique(tc[tc>0])
                        for i4 in Uniq:
                            Results['RegionData']['Area'].append(np.sum(tc==i4))
                            Results['RegionData']['Label'].append(i4)
                            Results['RegionData']['Overlap'].append(ValOverlaps[1][np.where(ValOverlaps[0]==np.unique(ta[tc==i4])[0])[0][0]])
                            Results['RegionData']['Overlap_ID'].append(np.where(ValOverlaps[0]==np.unique(ta[tc==i4])[0])[0][0])
                            for Ch in enumerate(ChOrder):  
                              if ChFunction[Ch[1]] != 'Dapi':
                                  Intt = Images[ChName[Ch[1]]]['Original'][XY[0]:XY[1], XY[2]:XY[3]].copy();
                                  Results[ChName[Ch[1]]]['IntSum'].append(np.sum(Intt[tc==i4]))
                                  Results[ChName[Ch[1]]]['IntMean'].append(np.mean(Intt[tc==i4]))
                                  
                                  if np.mean(Intt[tc==i4]) =='nan':
                                      sfasfadf
                        
                        t[Px]= 0 
                        t[XY[0]:XY[1], XY[2]:XY[3]]=t[XY[0]:XY[1], XY[2]:XY[3]]+ta
                            
                        t2[Px]= 0    
                        t2[XY[0]:XY[1], XY[2]:XY[3]]=t2[XY[0]:XY[1], XY[2]:XY[3]]+tc
                        
                        #Imshow(Image =   ta, Title = "ta", InteractiveWindow = True)  
                        #Imshow(Image =   tc, Title = "tc", InteractiveWindow = True)   
                        #Imshow(Image =   t, Title = "t", InteractiveWindow = True)  
                        #Imshow(Image =   t2, Title = "t2", InteractiveWindow = True)   
                        continue
                    #print('\n',Cell,':  ', WholeSize , ' / ', MeanArea, '     ', Percentage, '\n')
                    Changed = 0
                    if np.max(Percentage)<0.5:
                        if len(Uniq) == 3:
                            sections = regionprops_table (ta.astype('int8'), properties=('area','label', 'centroid' ))
                            Whole = regionprops_table (label(ta.astype('bool')), properties=('area','label', 'centroid' ))
                            dist = np.sqrt( np.square(sections['centroid-0']-Whole['centroid-0'])  + np.square(sections['centroid-1']-Whole['centroid-1']) )
                            ta[ta==Uniq[np.where(dist == np.min(dist))[0][0]]]=0   
                            #ta = expand2Boundary(ta, tb.astype('bool'), disk(1), 10, 'inside')
                           
                            Changed = 1
                        else:
                            #Imshow(Image =   ta, Title = "ta max %<0.5  Cell: " +str(Cell+1), InteractiveWindow = True) 
                            edge =  tb-erosion(tb,disk(1))
                            #Imshow(Image =   edge, Title = "ta max %<0.5  Cell: " +str(Cell+1), InteractiveWindow = True) 
                            EdgeNumbers = ta[edge>0]
                            EdgeCount = np.zeros(len(Uniq))
                            for i2 in enumerate(Uniq):      
                                 EdgeCount[i2[0]] = np.sum(EdgeNumbers == i2[1])
                            #Mins= np.zeros(2)   
                            #sdfsdfsdf
                            for i3 in range(len(Uniq)-2):
                                 findMin =  np.where(EdgeCount  == np.min(EdgeCount))[0][0]
                                 ta[ta == Uniq[findMin]] = 0 
                                 EdgeCount[findMin] = 10000
                            #Imshow(Image =   ta, Title = "ta max %<0.5  Cell: " +str(Cell+1), InteractiveWindow = True)   
                            
                            
                            Changed=1     
                    else:       
                        for i2 in enumerate(Uniq):
                            if Percentage[i2[0]]  < 0.5:
                                Changed=1
                                ta[ta==i2[1]]=0   
                    if Changed:
                        ## Imshow(Image =   ta, Title = "ta", InteractiveWindow = True) 
                        #Imshow(Image =   tb, Title = "tb", InteractiveWindow = True)  
                        #Imshow(Image =   ta, Title = "ta max %<0.5  Cell: " +str(Cell+1), InteractiveWindow = True) 
                        ta = expand2Boundary(ta, tb.astype('bool'), disk(1), 10, 'inside') 
                        ta = seperateTouchingObjects(ta)  
                        Uniq = np.unique(ta[ta>0])
                        if len(Uniq)==1:
                            Results['RegionData']['Area'].append(CellProps['area'][Cell])
                            Results['RegionData']['Label'].append(Cell+1)
                            Results['RegionData']['Overlap'].append(ValOverlaps[1][np.where(ValOverlaps[0]==Uniq)[0][0]])
                            Results['RegionData']['Overlap_ID'].append(np.where(ValOverlaps[0]==Uniq)[0][0])
                            for Ch in enumerate(ChOrder):  
                              if ChFunction[Ch[1]] != 'Dapi':
                                    Intt = Images[ChName[Ch[1]]]['BKG'][XY[0]:XY[1], XY[2]:XY[3]].copy();
                                    Results[ChName[Ch[1]]]['IntSum'].append(np.sum(Intt[ta>0]))
                                    Results[ChName[Ch[1]]]['IntMean'].append(np.mean(Intt[ta>0]))
                        else:
                            tc = label(erosion(ta,disk(1)).astype('bool'))
                            val  = np.zeros(np.max(tc))
                            for i3 in range(np.max(tc)):
                                val[i3] = np.unique(ta[tc==i3+1])
            
                            tc = expand2Boundary(tc, ta.astype('bool'), disk(1), 20, 'inside') 
                            tc = seperateTouchingObjects(tc)   
                            ta = tc.copy()
                            for i3 in range(np.max(tc)):
                                ta[tc==i3+1] = val[i3]
                            add2 = np.max(tc)   
                            tc[tc>0] = tc[tc>0] + MaxV + AddCells
                            Uniq = np.unique(tc[tc>0])
                            for i4 in Uniq:
                                Results['RegionData']['Area'].append(np.sum(tc==i4))
                                Results['RegionData']['Label'].append(i4)
                                Results['RegionData']['Overlap'].append(ValOverlaps[1][np.where(ValOverlaps[0]==np.unique(ta[tc==i4])[0])[0][0]])
                                Results['RegionData']['Overlap_ID'].append(np.where(ValOverlaps[0]==np.unique(ta[tc==i4])[0])[0][0])
                                for Ch in enumerate(ChOrder):  
                                  if ChFunction[Ch[1]] != 'Dapi':
                                      Intt = Images[ChName[Ch[1]]]['BKG'][XY[0]:XY[1], XY[2]:XY[3]].copy();
                                      Results[ChName[Ch[1]]]['IntSum'].append(np.sum(Intt[tc==i4]))
                                      Results[ChName[Ch[1]]]['IntMean'].append(np.mean(Intt[tc==i4]))
                                      
                                      if np.mean(Intt[tc==i4]) =='nan':
                                          sfasfadf
                            
                            t[Px]= 0 
                            t[XY[0]:XY[1], XY[2]:XY[3]]=t[XY[0]:XY[1], XY[2]:XY[3]]+ta
                                
                            t2[Px]= 0    
                            t2[XY[0]:XY[1], XY[2]:XY[3]]=t2[XY[0]:XY[1], XY[2]:XY[3]]+tc
            
               # elif len(Uniq)==0:
             
                    #%  check this cell is split correctly   
                   
             #%
            ValsOI = ['IntSum','IntMean']    
            Results.update({'Catagory' :  dict()})
            Results['Catagory'].update({'Count': list()})
            Results['Catagory'].update({'Percentages': list()})
            
            #Ch1=list(); Ch2=list(); Ch3=list(); Ch12=list(); Ch13=list(); Ch23=list() 
            for i in range(len(ValOverlaps[1])): 
                Results['Catagory'].update({ValOverlaps[1][i] : dict()}) 
                for i2 in ValsOI:
                    for Ch in enumerate(ChOrder):  
                        if ChFunction[Ch[1]] != 'Dapi':
                            Results['Catagory'][ValOverlaps[1][i]].update({i2+' '+ChName[Ch[1]]: list()})    
            totalCells = len(Results['RegionData']['Overlap'])
            for i in range(totalCells):   
                #print(Results['RegionData']['Overlap'][i])
                Cat = Results['RegionData']['Overlap'][i]
                Results['Catagory'][Cat]
                for i2 in ValsOI:
                    for Ch in enumerate(ChOrder): 
                        if ChFunction[Ch[1]] != 'Dapi':
                            Results['Catagory'][Cat][i2+' '+ChName[Ch[1]]].append(  Results[ChName[Ch[1]]][i2][i] )
                            
            for i in ValOverlaps[1]:            
                Results['Catagory']['Count'].append(   len(Results['Catagory'][i][i2+' '+ChName[Ch[1]]])     )           
                Results['Catagory']['Percentages'].append(   (len(Results['Catagory'][i][i2+' '+ChName[Ch[1]]]) / totalCells)*100)     
            #%   
                
            def plotSorted(X,Title,x,y):
                  X = sorted(X)
                  plt.figure()
                  plt.bar([*range(1,len(X)+1)],X)
                  plt.plot(X)
                  plt.title(Title)
                  plt.xlabel(x)
                  plt.ylabel(y)
                  plt.show()
              
            ix=0;   
            for i in ValOverlaps[1]:  
                for i2 in ValsOI:
                    for Ch in enumerate(ChOrder): 
                        if ChFunction[Ch[1]] != 'Dapi':
                            #print(ChName[Ch[1]]+' in '+i2+' Cat')
                            #plotSorted(Results['Catagory'][i][i2+' '+ChName[Ch[1]]], ChName[Ch[1]]+' in '+i2+' ('+i+') positive Cells','Cell ID', i2+' (RFU)' )
                            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i][i2+' '+ChName[Ch[1]] ].append(Results['Catagory'][i][i2+' '+ChName[Ch[1]]])
                            Temp = MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Concat '+i2+' '+ChName[Ch[1]]]
                            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Concat '+i2+' '+ChName[Ch[1]]] = Temp + Results['Catagory'][i][i2+' '+ChName[Ch[1]]]
                            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Per Scene Summary'][i][i2+' '+ChName[Ch[1]]].append(np.mean(Results['Catagory'][i][i2+' '+ChName[Ch[1]]]))  
                            #if np.isnan(np.mean(Results['Catagory'][i][i2+' '+ChName[Ch[1]]])):
                                #fsdfsdfsdfsdfsdf
                MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Count'].append(Results['Catagory']['Count'][ix]); 
                MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Percentage'].append(Results['Catagory']['Percentages'][ix]);
    
                ix=ix+1;
            
     #%
     
     
        for i in ValOverlaps[1]:  
            for i2 in ValsOI:
                for Ch in enumerate(ChOrder): 
                    if ChFunction[Ch[1]] != 'Dapi':
                            Temp = MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Concat '+i2+' '+ChName[Ch[1]]]  
                            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Concat Summary'][i][i2+' '+ChName[Ch[1]]] = [np.nanmean(Temp) , np.nanstd(Temp)]
            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Concat Summary']['Count'].append( np.sum( MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Count']))    
            MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Concat Summary']['Percentage'].append(np.mean( MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name][i]['Percentage']))
        MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Total'] = np.sum(MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Concat Summary']['Count'])
        MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Count'] = MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Average Concat Summary']['Count'] 
        MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Percentage']  = np.round((np.array(MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Count'])/MasterResults[SubFolder[Folder].Parent][SubFolder[Folder].Name]['Total'])*100 , 2)    
        if Folder+1 < len(SubFolder):
            if  SubFolder[Folder+1].Parent not in ParentFolderNames:
               print('Run Save Results') 
               selectFilesOI.pickelSave(file_path_variable,MasterResults,[ParentFolderNames[len(ParentFolderNames)-1]])
               selectFilesOI.SortforExcelExport(file_path_variable  ,MasterResults.copy(), [ParentFolderNames[len(ParentFolderNames)-1]], AllParentFolder, FolderNames)         
    
        else:
               print('Last Folder Run Save Results')
               selectFilesOI.pickelSave(file_path_variable,MasterResults,[ParentFolderNames[len(ParentFolderNames)-1]])
               selectFilesOI.SortforExcelExport(file_path_variable  ,MasterResults.copy(), [ParentFolderNames[len(ParentFolderNames)-1]], AllParentFolder, FolderNames)         
    
    
elif RunAnalysisOrResultSorting == 2:
    
    for Folder in range(StartFile,EndFile):
      if len(SubFolder[Folder].relaventFiles)>0:
        ChName[2] = SubFolder[Folder].Parent[0:4]
        if  SubFolder[Folder].Parent not in ParentFolderNames:
            ParentFolderNames.append(SubFolder[Folder].Parent)
            print('          1st instance of ' , SubFolder[Folder].Parent)    
        if  SubFolder[Folder+1].Parent not in ParentFolderNames or Folder==EndFile-1:
            print('\n --- Folder Completion', round((100/CheckStatusAt)*ixx),'% ',(Folder+1),'/',EndFile,' ---   ',  datetime.datetime.now())
            print('       Folder: ', Folder+1 , SubFolder[Folder].Parent,'\n SubFolder: ', SubFolder[Folder].Name)
            ixx=ixx+1  
            print('SaveFile', file_path_variable)
            MasterResults = selectFilesOI.pickelLoad(file_path_variable,MasterResults,[ParentFolderNames[len(ParentFolderNames)-1]])
            selectFilesOI.SortforExcelExport(file_path_variable  ,MasterResults.copy(), [ParentFolderNames[len(ParentFolderNames)-1]], AllParentFolder, FolderNames)         

        #file = open('MasterResults_'+i, 'rb')
        #MasterResults[i] = pickle.load(file)
        #file.close()    

    
    #%
    #%
    # Save in python Format
    # Save
    #%%
    
    
    #%% Load py File for export to Excel
    '''
    os.chdir(file_path_variable)
    file = open('MasterResults_'+i, 'rb')
    MasterResults[i] = pickle.load(file)
    file.close()  
    #'''
    #Results = MasterResults.copy()
    #import selectFilesOI        
        #%
    #%%        
                
def pickelSave(SavePath,MasterResults,ParentFolderNames):
    import pickle
    os.chdir(SavePath)
    for i in ParentFolderNames:
        print(i)
        
        dsfsd
        file = open('MasterResults_'+i, 'wb')
        pickle.dump(MasterResults[i], file)
        file.close()
        # Load
        #file = open('MasterResults_'+i, 'rb')
        #MasterResults[i] = pickle.load(file)
        #file.close()                
                
                
            










#%%
fdfsfsdfsdfsf
for Ch in enumerate(ChOrder):  
  #%
  if ChFunction[Ch[1]] != 'Dapi':
      Results[ChName[Ch[1]]] = dict()
      print('\n\n   --- ', ChName[Ch[1]], ' - ',ChFunction[Ch[1]],'Exstract data ---\n')
      Label = Images[ChName[Ch[1]]]['Label'].astype('int32')
      Im =   Images[ChName[Ch[1]]]['Original'].astype('int32')
      RelIm =   Images[ChName[Ch[1]]]['MaskedInt'].astype('int32')
      
      Imshow(Image = Label   , Title = "Im - "+ChName[Ch[1]]+"  labels removed", InteractiveWindow = True)
      
      CellProps = regionprops_table (Label, properties=('area','label'))
      Area = list();  IntSum = list();  IntMean = list(); RelIntSum=list(); RelIntMean=list();
      for Cell in enumerate(CellProps['label']): 
          Area.append(CellProps['area'][Cell[0]])
          IntSum.append(np.sum(Im[Label == Cell[1]]))
          IntMean.append(np.mean(Im[Label == Cell[1]]))
          RelIntSum.append(np.sum(RelIm[Label == Cell[1]]))
          RelIntMean.append(np.mean(RelIm[Label == Cell[1]]))          
          
          
      #Imshow(Image = Label   , Title = "Label - "+ChName[Ch[1]]+"  Mask", InteractiveWindow = True)
      Results[ChName[Ch[1]]].update( {'Area' : Area})
      Results[ChName[Ch[1]]].update( {'IntSum' : IntSum})
      Results[ChName[Ch[1]]].update( {'IntMean' : IntMean})
      Results[ChName[Ch[1]]].update( {'RelIntSum' : RelIntSum})
      Results[ChName[Ch[1]]].update( {'RelIntMean' : RelIntMean})
      


      plotSorted(IntSum)
      plotSorted(IntMean)
      
      plotSorted(RelIntSum)
      plotSorted(RelIntMean)

  
      Im =   Images[ChName[Ch[1]]]['Original'].astype('int32')
      Imx =  Images[ChName[Ch[1]]]['MaskedInt'].astype('int32')
      Imshow(Image = Label   , Title = "Label - "+ChName[Ch[1]], InteractiveWindow = True)
      Imshow(Image = Im   , Title = "Org - "+ChName[Ch[1]], InteractiveWindow = True)
      Im[Label>0]=0
      Imshow(Image = Im   , Title = "MaskedInt - "+ChName[Ch[1]], InteractiveWindow = True)
      
#%%  
CellMask = np.zeros(Im.shape)
ChanAdd = 1
for Ch in enumerate(ChOrder):  
  #%
  if ChFunction[Ch[1]] != 'Dapi':  
   Label = Images[ChName[Ch[1]]]['Label'].astype('int32')
   Label[Label>0]=ChanAdd;
   CellMask = CellMask + Label
   ChanAdd= ChanAdd +2
Imshow(Image = CellMask   , Title = "CellMask - ", InteractiveWindow = True)
   
   
#%%
pausing 
# pip install shapely
# pip install fiona
from skimage.morphology import erosion, dilation, closing, opening
Image = dict()

Mask = np.zeros(Im.shape[0:2])
for Ch in range(Im.shape[3]):
    Image.update({'Chan'+str(Ch+1) : {'Original' : Im[:,:,0,Ch,0]}})
    Imx  = Image['Chan'+str(Ch+1)]['Original'].copy()  #
    minthreshold = np.mean(Imx) + np.std(Imx)*4
    Imx = Imx - minthreshold
    Imshow(Image = Imx, Title = 'Imx')
    
    Imx[Imx<0]=0
    Imx[Imx>0]=1
    Mask[Imx>0] = Mask[Imx>0]+1
Mask[Mask>0]=1
Mask2 = Mask.copy()
'''
for i in range(5):
    Mask = closing(Mask, disk(10*(i+1)))
    Mask2 = Mask2 + Mask 
    print('Disk size: ', 10*(i+1))
'''
Imshow(Image = Mask2, Title = 'Mask2')
#%%

Mask3 = Mask2.copy()  
Mask[:]=0;
for i in range(1,6):
   Mask3 = Mask2.copy() 
   Mask3[Mask2!=(i+1)]=1
   #Imshow(Image = Mask3, Title = 'Mask'+str(i+1))
   Mask3 = label(Mask3)-1
   
   HigherArea = Mask3.copy() 
   LowerArea = Mask3.copy() 
   
   #Mask4 = dilation(Mask3, disk(1))
   #Mask4 = Mask4 - Mask3
   props = regionprops_table (Mask3, properties=('centroid','area'))
   
   #print(np.sum(Mask2i=i+1))
   AreaThreshold = 500
   AreaROIs = {'Area >' : props['area'][props['area']>AreaThreshold],
               'Ind >' : [AreaROIs2 for AreaROIs2, x in enumerate(props['area']) if x > AreaThreshold],
               'Area <=' : props['area'][props['area']<=AreaThreshold],
               'Ind <=' : [AreaROIs2 for AreaROIs2, x in enumerate(props['area']) if x <= AreaThreshold]
               }
   for i2 in range(0,len(props['area'])+1):
       if i2 in AreaROIs['Ind >']:
           #print('Larger on threshold')
           LowerArea[LowerArea==i2+1]=0
       else:
           #print('Smaller on threshold')
           HigherArea[HigherArea==i2+1]=0

   HigherArea2 =  dilation(HigherArea, disk(1))
   LowerArea2 =  dilation(LowerArea, disk(1))
   HigherArea2 = HigherArea2 - HigherArea
   LowerArea2 = LowerArea2 -LowerArea
   
   #Imshow(Image = HigherArea2, Title = 'HigherArea2'+str(i2+1))          
   #Imshow(Image = LowerArea2, Title = 'LowerArea2'+str(i2+1)) 
   
   for i2 in range(0,len(props['area'])+1):
       Temp = Mask2[LowerArea2==i2]!=0
       if len(Temp)>0:
           sum(Temp)/len(Temp)
           #print(i2, 'Low',sum(Temp)/len(Temp))
           if sum(Temp)/len(Temp) < 0.5 :
               LowerArea[LowerArea==i2] = 0
       Temp = Mask2[HigherArea2==i2]!=0
       if len(Temp)>0:       
           sum(Temp)/len(Temp)
           #print(i2, 'High',sum(Temp)/len(Temp))
           if sum(Temp)/len(Temp) < 0.75 :
               HigherArea[HigherArea==i2] = 0      
   Imshow(Image = HigherArea, Title = 'HigherArea'+str(i+1))          
   Imshow(Image = LowerArea, Title = 'LowerArea'+str(i+1))  
   
   LA = LowerArea.copy()
   LA = LA + HigherArea
   LA[LA>0]=1
   Imshow(Image = LA, Title = 'new LA')  
   
   
   Mask[HigherArea>0]= 1#(i+1)
   Mask[LowerArea>0]= 1#(i+1)
   Imshow(Image = Mask, Title = 'new Mask')    

   Image.update({'Merge' : {'Mask' : Mask}}) 
  
#%%

Image_ID_OverTime = list()
Props_ID_OverTime = list()

for T in range(0,3):
    C=0; Z=0;
    print("T: ",T)
    Im2  = Im[:,:,Z,C,T]  # look at specific Z-stack slice , channel and TimePoint
    plt.show(); plt.title("Org, Timepoint: "+str(T+1)); skio.imshow(Im2, cmap='hot');
    Image_ID_OverTime.append(np.zeros((0,0)))
    Props_ID_OverTime.append(np.zeros((0,0)))
    #% Get mask and skelatonised region
    
    test = remove_patterning(Image = Im2, Blur = 30, ConvF = [20, 'same'])
    
    Imshow(Image = test, Title = 'testing')
    
    
    
    pasuing 
    
    
    Mask_displayFigure = {'Skeleton_BW' : {'Title': 'Skeleton'       , 'Min':0 , 'Max':1 }, 
                          'Expanded_BW' : {'Title': 'Expanded'       , 'Min':0 , 'Max':1 },
                          'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',100]}     }
    #Mask_displayFigure = {'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',150]}     }
    #Mask_displayFigure ={'all' : {'Contrast': {'Min':0,'Max':['np.amax(Im)/',100]}, 'BW': True }}
    
    Im2
    
    
    
    Skeleton_BW, Expanded_BW, Mask  = getMask(Image = Im2, Blur = 30, ConvF = [20, 'same'], ExpandtoEdge = True, getSkeleton = True, 
                                        returnData = ['Skeleton_BW','Expanded_BW', 'Upper'],
                                        displayFigure = Mask_displayFigure ) 
    
    pasuing 
      
    #%   Branch ID detection 
    ##crop =  Im2[1:200,1:200]
    crop1 = Expanded_BW[1:200,1:200]
    crop2 = Skeleton_BW[1:200,1:200]
    
    #Branch_displayFigure ={'all' : {'Contrast': {'Min':0,'Max':['np.amax(Im)/',100]}, 'BW': True, 'ID': '50%' }}
    Branch_displayFigure = {'branch_ID' : {'Title': 'branch_ID', 'ID': '50%' }, 
                            'branchPoint_ID' : {'Title': 'branchPoint_ID', 'ID': '50%'  },
                            'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',100]}     }
    props, branch_ID, branchPoint_ID = network_branches(Image = crop2, Range = [2,5],
                                                              returnData = ['branchProps','branch_ID','branchPoint_ID'],
                                                              displayFigure = None )
    
    #Imshow(Image = branch_ID, Title = 'branch_ID');
    #Imshow(Image = branchPoint_ID, Title = 'branchPoint_ID');
    #% 
    
    props = regionprops_table (branch_ID, properties=('area','centroid',
                                                           'orientation',
                                                           'axis_major_length',
                                                           'axis_minor_length'))
    #%%
    Props_ID_OverTime[T] = props
    Image_ID_OverTime[T] = branch_ID.copy()
    Imshow(Image = Image_ID_OverTime[T] , Title = 'branch_ID-'+str(T+1));
    Imshow(Image = crop1 , Title = 'branch_ID_expaned-'+str(T+1));
    Expanded_ID = expand2Boundary(branch_ID, crop1, disk(1), 20,'inside'); 
    Imshow(Image = Expanded_ID , Title = 'branch_ID_expaned-'+str(T+1));
        
#%%

    
#%%
    
paussing 

import pylab as pl
from matplotlib.collections import LineCollection
from matplotlib.pyplot import gcf

#%% getconvHull


Temp  = np.where(Mask4>0) 
XY =  np.arange(len(Temp[0])*2).reshape(len(Temp[0]),2)
XY[:,0] =  Temp[0];    XY[:,1] =  Temp[1]

XY2 = list(XY)
Imx[Imx>0]=0

ImSz = Mask.shape
 
    
def plot_polygon(**varargin):
    input_name = list(varargin.keys())
    required_inputs   = ['Polygon']
    additional_inputs = ['axSize', 'includeLines', 'includePoints']
    all_varibles = required_inputs + additional_inputs
    contained = [x for x in input_name if x in all_varibles]
    
    polygon = varargin['Polygon']
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    if 'axSize' in contained:  
        axSize = varargin['axSize']
        ax.set_xlim([0, axSize[0]])
        ax.set_ylim([0, axSize[1]])
    else:
        margin = .3
        x_min, y_min, x_max, y_max = polygon.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin]) 
    patch = PolygonPatch(polygon, fc='#999999',  ec='#000000', fill=True,  zorder=-1)
    ax.add_patch(patch)
    if 'includeLines' in contained and varargin['includeLines']:
        lines = LineCollection(edge_points)
        pl.gca().add_collection(lines)
    if 'includePoints' in contained and varargin['includePoints']:
        pl.plot(XY[:,0],XY[:,1],'o', color='#f16824') 
    return fig    
    

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
 
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
 
    coords = points # coordinate already part of input (XY)
    #coords = np.array([point.coords[0]
    #                   for point in points])
 
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
 
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
 
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
 
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
 
        # Here's the radius filter.
        print(circum_r, alpha)
        #if circum_r < 1.0/alpha:
        if circum_r < alpha:   
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
 
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
     
    
alpha = 200
concave_hull, edge_points = alpha_shape(XY, alpha=alpha)
_ = plot_polygon(Polygon = concave_hull, axSize = ImSz, includeLines = True, includePoints = True)



#%%




"""   ###  Edge detection and intro to filters

Edge = edgedetection_Skimage(Im2, "sobel", "yes")
#  runs edge detection from function 

Entropy = entropy(Im2, disk(5))
plt.show(); plt.title("Entropy"); skio.imshow(Entropy);
# plots Entropy (texture based image) can help distinguish between features and background

fig, ax = try_all_threshold(Entropy, figsize=(10,8), verbose=False)
# test a number of different thresholds on image to see whichon works best (chosen Otsu - make sure to import)

threshold_Value = threshold_otsu(Entropy)   # Create threshold value
BW = Entropy >= threshold_Value             # Place value on to image via logical text > / < to make binary image
plt.show(); plt.title("Binary - Otsu thrsholding"); skio.imshow(BW); # plot binary image 

print("Percent of area of interest in whole image" , ((np.sum(BW==1)/np.sum(BW>=0))*100) , "%")
# there is a different in printing - , and +. , throws everthing together + concatenates the sections
#   Not sure what the difference is practically

"""

"""   # get stats from plotable data, (liner regratin equation of trend line, slop, R value ect.... )
from scipy.stats import linregress  # check if scipy is installed

print(linregress(X,Y)) # prints all varible results 
slope, intercept, r_value, p_value, std_err = linregress((X,Y))
print("y = ", slope, "X", " + ", intercept)
print("R squared")
print("R\N{SUPERSCRIPT TWO} = ", r_value**2)

"""






## Gaussian and median filters 
#Blur    = nd.gaussian_filter(Im2, sigma=3)
#Median  = nd.median_filter(Im2, size=3)
#plt.show(); plt.title("Blur"); skio.imshow(Blur); # plot binary image
#plt.show(); plt.title("Median"); skio.imshow(Median); # plot binary image 

#Im = img_as_float(Im[:,:,:,0])

Image_ID_OverTime = list()
Props_ID_OverTime = list()

for T in range(0,3):
    C=0; Z=0;
    print("T: ",T)
    Im2  = Im[:,:,Z,C,T]  # look at specific Z-stack slice , channel and TimePoint
    plt.show(); plt.title("Org, Timepoint: "+str(T+1)); skio.imshow(Im2, cmap='hot');
    Image_ID_OverTime.append(np.zeros((0,0)))
    Props_ID_OverTime.append(np.zeros((0,0)))
    #% Get mask and skelatonised region
    
    
    test = remove_patterning(Image = Im2, Blur = 30, ConvF = [20, 'same'])
    
    
    Imshow(Image = test, Title = 'testing')
    
    
    
    pasuing 
    
    
    Mask_displayFigure = {'Skeleton_BW' : {'Title': 'Skeleton'       , 'Min':0 , 'Max':1 }, 
                          'Expanded_BW' : {'Title': 'Expanded'       , 'Min':0 , 'Max':1 },
                          'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',100]}     }
    #Mask_displayFigure = {'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',150]}     }
    #Mask_displayFigure ={'all' : {'Contrast': {'Min':0,'Max':['np.amax(Im)/',100]}, 'BW': True }}
    
    Im2
    
    
    
    Skeleton_BW, Expanded_BW, Mask  = getMask(Image = Im2, Blur = 30, ConvF = [20, 'same'], ExpandtoEdge = True, getSkeleton = True, 
                                        returnData = ['Skeleton_BW','Expanded_BW', 'Upper'],
                                        displayFigure = Mask_displayFigure ) 
    
    pasuing 
      
    #%   Branch ID detection 
    ##crop =  Im2[1:200,1:200]
    crop1 = Expanded_BW[1:200,1:200]
    crop2 = Skeleton_BW[1:200,1:200]
    
    #Branch_displayFigure ={'all' : {'Contrast': {'Min':0,'Max':['np.amax(Im)/',100]}, 'BW': True, 'ID': '50%' }}
    Branch_displayFigure = {'branch_ID' : {'Title': 'branch_ID', 'ID': '50%' }, 
                            'branchPoint_ID' : {'Title': 'branchPoint_ID', 'ID': '50%'  },
                            'Org'         : {'Title': 'Original Image' , 'Min':0 , 'Max': ['np.amax(Im)/',100]}     }
    props, branch_ID, branchPoint_ID = network_branches(Image = crop2, Range = [2,5],
                                                              returnData = ['branchProps','branch_ID','branchPoint_ID'],
                                                              displayFigure = None )
    
    #Imshow(Image = branch_ID, Title = 'branch_ID');
    #Imshow(Image = branchPoint_ID, Title = 'branchPoint_ID');
    #% 
    
    props = regionprops_table (branch_ID, properties=('area','centroid',
                                                           'orientation',
                                                           'axis_major_length',
                                                           'axis_minor_length'))
    #%%
    Props_ID_OverTime[T] = props
    Image_ID_OverTime[T] = branch_ID.copy()
    Imshow(Image = Image_ID_OverTime[T] , Title = 'branch_ID-'+str(T+1));
    Imshow(Image = crop1 , Title = 'branch_ID_expaned-'+str(T+1));
    Expanded_ID = expand2Boundary(branch_ID, crop1, disk(1), 20,'inside'); 
    Imshow(Image = Expanded_ID , Title = 'branch_ID_expaned-'+str(T+1));
    
    #%%
    
    #string = ''
    
    #PossTracks = pd.Series(range(0,  len(Props_ID_OverTime[T]['area'])))
    #PossTracks = set(range(1,len(Props_ID_OverTime[T]['area'])+1)) #np.zeros(len(Props_ID_OverTime[T]['area']))
    if T == 0:
        ID_len = len(Props_ID_OverTime[T]['area'])
        Track_ID = pd.DataFrame( {'T'+str(T+1) :  range(1,len(Props_ID_OverTime[T]['area'])+1)}, index = range(1,len(Props_ID_OverTime[T]['area'])+1) )
    else:

       # Track_ID['T'+str(T+1)] = range(1,len(Track_ID['MI'])+1)
        Track_ID['T'+str(T+1)] = 0 #range(1,len(Track_ID['T'+str(T)])+1)
        
        
        
        

        def Assign_filabment_2previous(**varargin):
            '''
             from a fragmented skeleton mask this determines the  \n
             example:  Assign_filabment_2previous(PreImage = Image_ID_OverTime[T-1], Image = Image_ID_OverTime[T]
                                displayFigure = {'PreImage' : {'Title': 'Previous Image', },                              \n                 
                                                 'NewAssigment' : {'Title': 'Current Image assigned unique pixels',       \n
                                                 'NewAssigmenMI' : {'Title': 'Current Image assigned any pixels',       \n\n
            accepted_input_types = {
                                    'PreImage':    'ndarray', \n
                                    'PreProp':    'ndarray', \n
                                    'Image':    'ndarray', \n
                                    'CurrentProp':    'ndarray', \n
                                    'returnData' : 'list', \n
                                    'displayFigure': {'set' , 'dict'} \n
                                    }   
                                                        
            returnData: retunrs data this can be anything from values, images or regiion properties
                                    
            displayFigure:  displys figures that are made in function, with range (if specified) uses custom "Imshow" function                                                   
                             image avlaible:  'branchID', 'branchPointID'
                                 \n
                                 with base  - 'PreImage' (Previous image), 'NewAssigment' (newly assigned image), 'NewAssigmenMI' (newly assigned image)   \n
                                \n
            '''
            
            accepted_input_types = {'PreImage':  'ndarray',  'PreProp':     'dict',
                                    'Image':     'ndarray',  'CurrentProp': 'dict', 
                                    'returnData' : 'list',
                                    'displayFigure': {'set' , 'dict'}  # might only be compatible with "dict"
                                    }
            
            # check that inputs are correct
            input_name = list(varargin.keys())
            required_inputs   = ['PreImage', 'PreProp', 'Image', 'CurrentProp']
            additional_inputs = ['returnData', 'displayFigure']
            all_varibles = required_inputs + additional_inputs    
                    
            
            #import random
           
            PreImage =   varargin['PreImage'].copy() 
           # PreProp  =   varargin['PreProp']
           # PreImage_ID = list([*range(1,len(PreProp['area'])+1)])
            ##random.shuffle(PreImage_ID)
           # for ip in range(1 , len(PreImage_ID)):
           #     PreImage[PreImage_ID == ip] = PreImage_ID[ip-1]
                
            #jumbeled[jumbeled>0]=jumbeled[jumbeled>0]+20
            Imshow(Image = PreImage , Title = 'PreImage ID' ); 
            

            
            def DisMID(PreImage, CurrentImage, minDistanceRequired):
                Temp  = np.where(PreImage>0) 
                Ind1 = np.arange(len(Temp[0]))
                Pre_Px =  np.arange(len(Temp[0])*2).reshape(len(Temp[0]),2)
                Pre_Px[:,0] =  Temp[0];    Pre_Px[:,1] =  Temp[1]
                Temp  = np.where(CurrentImage>0) 
                Ind2 = np.arange(len(Temp[0]))
                Cur_Px =  np.arange(len(Temp[0])*2).reshape(len(Temp[0]),2)
                Cur_Px[:,0] =  Temp[0];    Cur_Px[:,1] =  Temp[1]                        
                Dist = cdist(Pre_Px, Cur_Px)
                minDistanceRequired = 10
                Dist[Dist>minDistanceRequired] = np.inf
                return Dist, Pre_Px, Cur_Px


            def distanceAsignment(PreImage, CurrentImage, Distance, UniquePx, Pre_Px, Cur_Px ): 
                Dist = Distance.copy()
                NewAssigment = PreImage.copy()
                NewAssigment[NewAssigment>0] = 0
                
                Dist.shape[0]

                ix = 0
                while ix < min(Dist.shape)  and Dist.min() != np.inf:
                    MinDist1 = Dist.min(axis=0)
                    #MinDisMI = Dist.min(axis=1)    
                    minPxfound = False
                    Ind = [i for i, e in enumerate(MinDist1) if e == MinDist1.min()]
                    for i in Ind:
                        IndAx2 =  [i2 for i2, e in enumerate(Dist[:,i]) if e == MinDist1.min()]
                        for i2 in IndAx2:
                            Dist[i2,:].min()
                            if  Dist[i2,i] == MinDist1.min(): # check that this the min dist for T1 is the min for MI
                                ## so far this will just take the first min encounter 
                                minPxfound = True
                                break
                        #if minPxfound:   
                        #    break
                        if minPxfound:
                            if UniquePx:
                                Dist[i2,:] = np.inf
                                Dist[:,i]  = np.inf
                            else:
                                Dist[i2,i]  = np.inf
                            Pre_ID = PreImage[Pre_Px[i2,0], Pre_Px[i2,1]]
                            Cur_ID = CurrentImage[Cur_Px[i,0], Cur_Px[i,1]]
                            NewAssigment[Cur_Px[i,0], Cur_Px[i,1]] = Pre_ID
                    ix=ix+1 
                Imshow(Image = NewAssigment , Title = 'branch ID Assigment, testing     Dist Iterations  '+str(ix)); 
                return NewAssigment

            def filament_fissioinFusionAssigment(NewAssigment, NewAssigmenMI, PreProp, CurrentProp):
                
                
                CurrentImage_ID = list([*range(1,len(CurrentProp['area'])+1)])
                for ip in range(1 , len(CurrentImage_ID)):
                    fragment = NewAssigment[CurrentImage_ID == ip]
                    
                    print(ip+1, fragment, '/n')
                    print(ip+1, unique(fragment), '/n')
                
                # PreProp  =   varargin['PreProp']
                # PreImage_ID = list([*range(1,len(PreProp['area'])+1)])
                 ##random.shuffle(PreImage_ID)
                # for ip in range(1 , len(PreImage_ID)):
                #     PreImage[PreImage_ID == ip] = PreImage_ID[ip-1] 



            Distance_betweenPre_n_current, Pre_Px, Cur_Px  = DisMID(varargin['PreImage'], varargin['Image'], 10)
            NewAssigment  = distanceAsignment(varargin['PreImage'], varargin['Image'], Distance_betweenPre_n_current, True,  Pre_Px, Cur_Px )
            NewAssigmenMI = distanceAsignment(varargin['PreImage'], varargin['Image'], Distance_betweenPre_n_current, False, Pre_Px, Cur_Px)
        
        
            
        
        
        Assign_filabment_2previous(PreImage = Image_ID_OverTime[T-1], PreProp = Props_ID_OverTime[T-1], Image = Image_ID_OverTime[T], CurrentProp = Props_ID_OverTime[T])
        
#%%
                
        ix = 0
        while ix < min(len(Ind1),len(Ind2))  and DisMI.min() != np.inf:
            MinDist1 = DisMI.min(axis=0)
            #MinDisMI = Dist.min(axis=1)    
            minPxfound = False
            Ind = [i for i, e in enumerate(MinDist1) if e == MinDist1.min()]
            print('Min - ',MinDist1.min())
            for i in Ind:
                IndAx2 =  [i2 for i2, e in enumerate(DisMI[:,i]) if e == MinDist1.min()]
                for i2 in IndAx2:
                    DisMI[i2,:].min()
                    if  DisMI[i2,i] == MinDist1.min(): # check that this the min dist for T1 is the min for MI
                        ## so far this will just take the first min encounter 
                        DisMI[i2,i]  = np.inf
                        Pre_ID = jumbeled[Pre_Px[i2,0], Pre_Px[i2,1]]
                        Cur_ID = Image_ID_OverTime[T][Cur_Px[i,0], Cur_Px[i,1]]
                        NewAssigmenMI[Cur_Px[i,0], Cur_Px[i,1]] = Pre_ID
            
            ix=ix+1         
                
                
    #        Imshow(Image = Image_ID_OverTime[T-1] , Title = 'branch Previous Timepoint  '+str(ix)); 
    #        Imshow(Image = Image_ID_OverTime[T] , Title = 'branch current Timepoint  '+str(ix)); 
    #        Imshow(Image = jumbeled , Title = 'branch ID jumbeled, Dist Iterations  '+str(ix)); 
            Imshow(Image = NewAssigmenMI , Title = 'branch ID AssigmenMI, Dist Iterations  '+str(ix)); 
            
            
            
            
            
            
            
            
            
        Assign_filabment_2previous(PreImage = Image_ID_OverTime[T-1], PreProp = Props_ID_OverTime[T-1], Image = Image_ID_OverTime[T], CurrentProp = Props_ID_OverTime[T])
        
        PreImage = Image_ID_OverTime[T-1]
        
        
        '''  
        Crops = Im[:,:,Z,C,T-1][1:200,1:200]
        Imshow(Image = Crops , Title = 'T1'); 
        Crops = Im[:,:,Z,C,T][1:200,1:200]
        Imshow(Image = Crops , Title = 'MI');


        fig, ax = plt.subplots(1,3)
        ax[0].imshow(Image_ID_OverTime[T-1])
        ax[1].imshow(Image_ID_OverTime[T])
        ax[2].imshow(NewAssigment)
        plt.show()
        '''
        
        
#%%      
        for ip in range(1 , len(Props_ID_OverTime[T]['area'])):
            PxVal1 = NewAssigment[Image_ID_OverTime[T] == ip]
            PxVal2 = NewAssigmenMI[Image_ID_OverTime[T] == ip]
            Unique1 = unique(PxVal1, 0)
            Unique2 = unique(PxVal2, 0)
            Unique_1n2 = list(Unique1[:,0]) + list(Unique2[:,0])
            Unique_1n2 = unique(Unique_1n2, 0)
            if len(Unique_1n2) == 1:
                if Unique1[0,2] > 50 and Unique2[0,2] > 75 :
                    ID = Unique_1n2[0,0] 
                    if Track_ID['T'+str(T+1)][ID] != 0:
                        Track_ID['T'+str(T+1)][ID] = Track_ID['T'+str(T+1)][ID] + list(ip)
                    else:    
                        Track_ID['T'+str(T+1)][ID] = [ip, 'temp']
                        Track_ID['T'+str(T+1)][ID] = [ip]
                        
            elif  len(Unique_1n2) > 1:
                blablabls
                        
                

#%% 
'''    
        for ip in range(1 , len(Props_ID_OverTime[T]['area'])):
            #temp = np.zeros(Image_ID_OverTime[T-1].shape)
            print(ip)
      #      PossTracks[ip]
      #      PossTracks.add(ip)
            PxVal = Image_ID_OverTime[T-1][Expanded_ID==ip]
            Unique1 = unique(PxVal, 0)
            print("\n",Unique1)
            

            

            
            
            
            if len(Unique1[:,0]) == 1:
                ID = Unique1[0,0]
                PxVal = Expanded_ID[Image_ID_OverTime[T-1]==ID]
                Unique2 = unique(PxVal, 0)
                for i2 in range(0, len(Unique2[:,2])):
                    if Unique2[i2,2] > 50 and Unique2[i2,0] == ID   :
                        Track_ID['T'+str(T+1)][ID]=ip
            elif  len(Unique1[:,0]) > 1   :
                ApprovedID = np.zeros(0)
                for ID in Unique1[:,0]:
                    PxVal = Expanded_ID[Image_ID_OverTime[T-1]==ID]
                    Unique2 = unique(PxVal, 0)
                    for ix in Unique2[:,2] > 25:
                        
                        if ix:
                            print('\n\n\nis it this that is wrong???')
                            print(type(Track_ID['T'+str(T+1)][ID]).__name__)
                            print('\n\n\n')
                            if type(Track_ID['T'+str(T+1)][ID]).__name__ == 'int64':
                                print('\n\n\n this successful? \n')
                                Track_ID['T'+str(T+1)][ID] = [ip, 'temp']
                                Track_ID['T'+str(T+1)][ID] = [ip]
                                print('\n Yes i think? \n')
                                
                            else:    
                                print('\n\n\n 2nd add on \n\n\n')
                                Track_ID['T'+str(T+1)][ID] = np.append(Track_ID['T'+str(T+1)][ID],ip)
                 
                        if T==1 and ID == 17:
                            exit()
                 
                    #           ApprovedID = np.insert(ApprovedID,len(ApprovedID),ID, axis=0) 
                #PossTracks.add(ip)
                #PossTracks[ip] : ApprovedID
                #PossTracks[ip]['Multi'] = ApprovedID
            else:
                #PossTracks.add((ip,'NaN'))
                ID_len = ID_len+1
                Track_ID.loc[ID_len]=0
                Track_ID['T'+str(T+1)][ID_len]=ID_len
                # Track_ID = Track_ID.drop(76) delete Lb row
            
            
            
            
            #input(str(ip)+"Press Enter to continue...\n"+str(ID_Ind)) 
            
            
            #SkelPx = np.where(Expanded_ID==ip)            
            #Track_ID = np.insert(Track_ID,len(Track_ID),0, axis=0)

            '''
    
        #%%
         

       
  
    
for T in range(0,3):  
    Imshow(Image = Image_ID_OverTime[T] , Title = 'branch_ID-'+str(T+1));
    
#%%
'''
sigma_est = np.mean(estimate_sigma(Im2, channel_axis=None)) # need to check if each frame is independant or if Zstack of same channel can to treated as 3D and not independant slices 
nlm = denoise_nl_means(Im2, h=1.15*sigma_est, fast_mode=False, patch_size=5, patch_distance=6, channel_axis=None)

# Alternative to denoise_nl_means using packaged variables

#  ## Packaged variables (dict)
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)
denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
#                     Here ^^ the ** means unpack from "dictionary" (dict() function, previous defoned above)
"""

### histogram segmentation to create differennt masks 
mask = img_as_ubyte(nlm)
plt.show(); plt.title("mask"); skio.imshow(mask); # plot binary image
array = mask[mask>0]
maxPx = np.amax(array); minPx = np.amin(array); # find max and min value within image
mean = np.mean(array);  std_lowerInt = np.std(array[array<np.mean(array)])
minTheshold = mean-(mean-std_lowerInt)/2
mask2 = mask > minTheshold

X, Y = np.histogram(array, bins=50, range=(minTheshold,maxPx))
BinCentre = []
for i in range(X.size):
#    print(i, Y[i],  "\n                         " , (Y[i]+Y[i+1])/2)
    BinCentre = np.append(BinCentre,(Y[i]+Y[i+1])/2)
#print(i+1, Y[i+1])

"""

ax1 = plt.subploMIgrid((2, 2), (0, 0))
ax1.imshow(mask)
ax2 = plt.subploMIgrid((2, 2), (0, 1))
ax2.imshow(mask2)
ax3 = plt.subploMIgrid((2, 2), (1, 0),  colspan=2)
ax3.plot(BinCentre, X)
ax3.scatter(minTheshold/2, np.sum(mask>minTheshold), marker="s", s=BinCentre[0]*2, c = 'orange')
ax3.bar(minTheshold/2, np.sum(mask>minTheshold), width = BinCentre[0], fill=False, edgecolor = 'red' )
ax3.annotate('Ignored threshold', xy=(minTheshold,  np.sum(mask>minTheshold)), xytext=((BinCentre[-1]-BinCentre[0])/2, np.amax(X)),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
   
       #%%  
# mask3 = convF(mask, 20, 'same')

sigma_est = np.mean(estimate_sigma(Im2, channel_axis=None)) # need to check if each frame is independant or if Zstack of same channel can to treated as 3D and not independant slices 
nlm = denoise_nl_means(Im2, h=1.15*sigma_est, fast_mode=False, patch_size=5, patch_distance=6, channel_axis=None)
mask = img_as_ubyte(nlm)
Imshow(Image = mask, Title = 'mask?', Min = 0, Max = np.amax(mask))
'''

'''
     plot image (*Image, Title, Min, Max)  \n_________ * indicates require inputs
     \ninput types accepted: Array(numpy.ndarray), Min(int), Max(int))
     example:  Imshow(Image = Org, Title = 'Org', Min = 0, Max = np.amax(Org), Colorbar = False, ColorMap = 'Blues')
     
     accepted_input_types = {
                             'Image':    'ndarray', \n
                             'Title':    'str', \n
                             'Min':     {'int', 'uint8', 'uint16', 'uint32', 'float16', 'float32', 'float64'}, \n
                             'Max':     {'int', 'uint8', 'uint16', 'uint32', 'float16', 'float32', 'float64'}, \n
                             'Colorbar': 'bool', \n
                             'ColorMap': 'str' \n
                             'returnFig':'bool'
                             }
''' 
 



#%%
'''
fig, ax = plt.subplots(1,3)
ax[0].imshow(mask)
ax[1].imshow(mask5)
ax[2].imshow(mask6)
plt.show()
'''
#%%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

image = np.zeros((600, 600))

rr, cc = ellipse(300, 350, 100, 220)
image[rr, cc] = 1

image = rotate(image, angle=15, order=0)

rr, cc = ellipse(100, 100, 60, 50)
image[rr, cc] = 1

label_img = label(image)
regions = regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()

#%%



