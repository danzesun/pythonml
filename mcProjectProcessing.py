# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:32:25 2017

@author: Jason
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
#from mcWordProcessor import mcWordProcessor as mcWord


import seaborn as sns

from matplotlib import style
from mcControlFile import mcControlFile
from mcCorrmatProcessing import mcCorrmatProcessing as mcc
from mcDataManagerBase import mcDataManagerBase

from mcExploreAnalyze import mcExploreAnalyze
from mcCategoryDataHandler import mcCategoryDataHandler  as mcCategoryHandler

from mcMissingDataProcessor import mcMissingDataProcessor as mcMissing

from mcUtility import mcUtility as mcUtility
from scipy import stats
from mcFeatureEngineering import mcPCAAnlysis as mcPCA
from mcTracking import mcProjTracking as mcProjTracking
from mcProcessRegressor import mcProcessRegressor as mcProcessRegressor
from mcProcessClassfier import mcProcessClassfier as mcProcessClassfier
from mcDataManagerBase import mcDataManagerBase
from mcDataManagerBase import mcDataConfig
from mcAutoProcessing import mcAutoProcessing
from xgboost import plot_importance
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import sys
from imblearn.over_sampling import SMOTE

class  mcDataManagerNPL(mcDataManagerBase):
 
    def preSetupTargetField(self, dataConfig):
        return super(mcDataManagerNPL, self).preSetupTargetField(dataConfig)
    def postSetupTargetField(self, dataConfig):
         return super(mcDataManagerNPL, self).postSetupTargetField(dataConfig)
    def preSetupFeatureFields(self, dataConfig):
        return super(mcDataManagerNPL, self).preSetupTargetField(dataConfig)
    def postSetupFeatrureFields(self, dataConfig):
         return super(mcDataManagerNPL, self).postSetupTargetField(dataConfig)
  
    
    
print('\nSetting project enviroment....\n' )

dataConfig = mcDataConfig()
dataConfig.m_ctrFileName =  "E:/kaggle/Jason Development/mcproject\mcprocessing-2.0\controlfile.ini"
dataConfig.m_trainDataPath = "E:/kaggle/talkdata/train_sample.csv"
dataConfig.m_testDataPath=""
dataConfig.m_test_size_ratio = 0.38
dataConfig.m_doTrainTestSplitted = True
dataConfig.m_targetFieldName = "is_attributed"
dataConfig.m_intputFormat = "csv"
dataConfig.m_doOverSample = True
 
   
controlFile = mcControlFile(dataConfig.m_ctrFileName )


##################Override this ############################################
dataManager = mcDataManagerNPL(dataConfig)


#####################################################################3


m_AutoProcessing =  mcAutoProcessing()

print('\nPreprocessing data....\n' )
#############Preprocessing###############################################################

fieldDropList =  controlFile.getPropertylist("project", "fieldexcluded")
onlyIncludeList =  controlFile.getPropertylist("project", "fieldsincludeonly")
dataConfig.m_excludeFeatureList= fieldDropList;
dataConfig.m_onlyIncludeList = onlyIncludeList
dataManager.setupData(dataConfig)


projectTracking = mcProjTracking()

#projectTracking.ctrFileName = dataConfig.m_ctrFileName

m_AutoProcessing.setDataConfig(dataConfig)
m_AutoProcessing.setDataManager(dataManager)  
m_AutoProcessing.setProjectTracking(projectTracking)  
m_AutoProcessing.setControlFile(controlFile)
 
#m_AutoProcessing.AttachdataManager( dataConfig, projectTracking, dataManager, controlFile)
#m_AutoProcessing.convertFieldToCategoryIfNeed()

#######################################################################################



print('\nReading data....\n' )
 
print('\nDisplayong data structure....\n' )

m_AutoProcessing.displayDataframeInfo()

####################Handling of Categorical Data####################################333
categoryThred = controlFile.getIntegerProperties("project", "categorythredshold")
displagcategorylist = controlFile.getBooleanProperties("project", 'displagcategorylist')

print('\nProcess category data....\n' )

CategoryHandler = mcCategoryHandler(dataManager.m_dataFrame, categoryThred, displayCategoryCountList=displagcategorylist)


#####################Handling of Missing Data #############
print('\n\nProcess missing data....\n' )
m_AutoProcessing.processMissingData()
#Exploration analysisdz


#if controlFile.getBooleanProperties("exploreText", "process") == True:
#    textFieldList = controlFile.getPropertylist("exploreText", "fieldlist")
#    for textField in textFieldList:  
#        mcWord.createWordCountField(textField, dataManager.m_dataFrame, dataManager.m_targetFieldName)
            
    
#####################Explore of Correlactionship #############    
if controlFile.getBooleanProperties("corrmat", "process") == True:
    print('\n\nProcess corrlatonmap....' )
    m_AutoProcessing.processCorrmat()

#####################Feature Selection  - PCA  #############  
if controlFile.getBooleanProperties("pca", "process") == True:
    print('\n\nPerforming PCA ....' )
    m_AutoProcessing.processPCA()

#####################Exploratory Analysis    #############  
if controlFile.getBooleanProperties("exploreanalysis", "process") == True:
    print('\n\nExploring analysis....' )
    m_AutoProcessing.exploreAnalysis()

#####################Regression Modelling    #############  
if controlFile.getBooleanProperties("regression", "process") == True:
    print('\nRunning Regression....\n' )
    m_AutoProcessing.runRegression()

#####################Classification Modelling    #############  
if controlFile.getBooleanProperties("classification", "process") == True:
    print('\n\nRunning classification....')    
    m_AutoProcessing.runClassification()
     
if controlFile.getBooleanProperties("tensorprocess", "process") == True:
    print('\n\nRunning Tensorflow analysis....\n\n' )
    m_AutoProcessing.runTensorAnalysis()

    