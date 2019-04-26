# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:51:56 2018

@author: Jason
"""

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

from mcExploreAnalyze import mcExploreAnalyze
from mcCategoryDataHandler import mcCategoryDataHandler  as mcCategoryHandler

from mcMissingDataProcessor import mcMissingDataProcessor as mcMissing

from mcUtility import mcUtility as mcUtility
from scipy import stats
from mcFeatureEngineering import mcPCAAnlysis as mcPCA
from mcTracking import mcProjTracking as mcProjTracking
from mcProcessRegressor import mcProcessRegressor as mcProcessRegressor
from mcProcessClassfier import mcProcessClassfier as mcProcessClassfier
from mcTsRegession import tensorDataColumnManager
from mcDataManagerBase import mcDataManagerBase
from mcDataManagerBase import mcDataConfig

from sklearn.feature_extraction.text import TfidfVectorizer
import string


    
class mcAutoProcessing:
    def __init__(self, ):
        self.m_CanProcess = False;
     

#####################################################################################   
######################################################################################     
      
    def setDataConfig(self, dataConfig):
       self.m_dataConfig = dataConfig  
       
    def setDataManager(self, dataManager):
       self.m_dataManager = dataManager

    def setProjectTracking(self, projectTracking):
       self.m_projectTracking = projectTracking
       self.m_projectTracking.setupFeatureTracking(self.m_dataManager.m_dataFrame)
  
    def setControlFile(self, controlFile):
       self.m_controlFile = controlFile;
    
#######################################################################################################3     
    def convertFieldToCategoryIfNeed(self): 
       CategoryToConvertList = self.m_controlFile.getPropertylist("project", "converttocategorylist")
       for field in CategoryToConvertList:
           self.m_dataManager.m_dataFrame[field] = self.m_dataManager.m_dataFrame[field].astype('category')
   
 
                  
##########################################################################    
    def displayDataframeInfo(self):
        print (self.m_dataManager.m_dataFrame.dtypes)
        print(self.m_dataManager.m_dataFrame.head())
    
    
##########################################################################    

    def processMissingData(self):
        
        Missing = mcMissing(self.m_dataManager.m_dataFrame)
        missingTreashold = float(self.m_controlFile.getProperties("missing", "treadshold"))
        displayMissingList = self.m_controlFile.getBooleanProperties("missing", "displaylist") 
        displayMissingChart = self.m_controlFile.getBooleanProperties("missing", "displaychart") 
        toKeepList, toRemoveList = Missing.updateMissingDataList(missing_treadshold = missingTreashold, displayChart = displayMissingChart, displayList = displayMissingList)   
        
        RequireKeepList = self.m_controlFile.getPropertylist("missingdatadelete", "nullfieldkeep")
        RequireRemoveList = self.m_controlFile.getPropertylist("missingdatadelete", "nullfieldremove")   
    
        self.m_projectTracking.updateFeatureUsageStatusList(RequireRemoveList, "Missing", 1)
        self.m_projectTracking.updateFeatureUsageStatusList(RequireKeepList, "Missing", 2)
        self.m_projectTracking.updateFeatureUsageStatusList(toRemoveList, "Missing", 10)
        self.m_projectTracking.updateFeatureUsageStatusList(toKeepList, "Missing", 20)
    
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "Missing")
     #  we are not going to drop the feature here, intead , we will use the tracking Matrix   
    #    mcMissing.dropFeatures(Missing.m_dataFrame, Missing.m_MissingToRemoveList)
        ReplaceDict = self.m_controlFile.getDictInSection("missingdatareplace")
        fieldProcessed =[]
    
        for field in toKeepList:
            if field in list(ReplaceDict.keys()):
                mcMissing.fillMissingData(Missing.m_dataFrame, fieldToProcess=field, value=ReplaceDict[field])
                fieldProcessed.append(field)
                            
        for field in fieldProcessed:
            toKeepList.remove(field)
     
        mcUtility.clearAllNaField(self.m_dataManager.m_dataFrame[toKeepList])
            
       
        
    
    
    def getFinalToTrainFeatureList(self):
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "Missing")
        toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList)
    
        if self.m_controlFile.getBooleanProperties("corrmat", "process") == True:
            toKeepList2, toRemoveList2 = self.m_projectTracking.getCalcuatedToKeepRemoveList(self.m_dataManager.m_dataFrame, "Corr")
            toKeepList = mcUtility.addListToList(toKeepList, toKeepList2)
            toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList2)
    
        if self.m_controlFile.getBooleanProperties("pca", "process") == True:
            toKeepList3 = self.m_projectTracking.getAlllNoneZeroList(self.m_dataManager.m_dataFrame, "PCAComp1")
            toKeepList = mcUtility.addListToList(toKeepList, toKeepList3)
          
    
        if self.m_controlFile.getBooleanProperties("regression", "process") == True:
            toKeepList5 = self.m_controlFile.getPropertylist('regression', 'featureincluded')
            toRemoveList5 = self.m_controlFile.getPropertylist('regression', 'featureexcluded')      
            toKeepList = mcUtility.addListToList(toKeepList, toKeepList5)
            toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList5)
    
        toKeepList = mcUtility.removeIfInTheList(toKeepList, self.m_dataManager.m_targetFieldName)
        
        return toKeepList, toRemoveList
        
    
    def getExploreFeatureList(self):
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "Missing")
        toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList)
    
        if self.m_controlFile.getBooleanProperties("corrmat", "process") == True:
            toKeepList2, toRemoveList2 = self.m_projectTracking.getCalcuatedToKeepRemoveList(self.m_dataManager.m_dataFrame, "Corr")
            toKeepList = mcUtility.addListToList(toKeepList, toKeepList2)
            toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList2)    
    
        if self.m_controlFile.getBooleanProperties("pca", "process") == True:
            toKeepList3 = self.m_projectTracking.getAlllNoneZeroList(self.m_dataManager.m_dataFrame, "PCAComp1")
            toKeepList = mcUtility.addListToList(toKeepList, toKeepList3)
          
    #    if self.m_controlFile.getBooleanProperties("exploreanalysis", "process") == True:
        toKeepList4 = self.m_controlFile.getPropertylist('exploreanalysis', 'featureincluded')
        toRemoveList4 = self.m_controlFile.getPropertylist('exploreanalysis', 'featureexcluded')      
        toKeepList = mcUtility.addListToList(toKeepList, toKeepList4)
        toKeepList = mcUtility.removeListfromList(toKeepList, toRemoveList4)
            
        toKeepList = mcUtility.removeIfInTheList(toKeepList, self.m_dataManager.m_targetFieldName)
    
        return toKeepList, toRemoveList
    
    def getCategorialFeatureList(self):
        return CategoryHandler.getCategoryFeatureList()
    
    def exploreAnalysis(self):
     
        mcExplore = mcExploreAnalyze(self.m_dataManager.m_dataFrame)
        
        if self.m_controlFile.getBooleanProperties("exploreanalysis", "displayhistogram") == True:
            mcExplore.DisplayHistogramForField(self.m_dataManager.m_targetFieldName)
            
        toKeepList, toRemoveList = getExploreFeatureList()   
        
        work_dataframe = self.m_dataManager.m_dataFrame[mcUtility.addItemToList( toKeepList, self.m_dataManager.m_targetFieldName)]
            
        if self.m_controlFile.getBooleanProperties('exploreanalysis', 'numericanalysis') == True:
            mcExplore.AnalyzeTargetNumericFeatureList(work_dataframe, self.m_dataManager.m_targetFieldName, toKeepList)
        
        
        
        if self.m_controlFile.getBooleanProperties('exploreanalysis', 'nonnumericanalysis') == True:
    #        mcExplore.AnalyzeTargetNoneNumericFeatureList(work_dataframe, self.m_dataManager.m_targetFieldName, toKeepList)
            categoryList= getCategorialFeatureList()
            if (self.m_controlFile.getBooleanProperties('exploreanalysis', 'includeallcategoryfearture')== False):
                categoryList = mcUtility.removeListfromList(categoryList, toRemoveList)
            tempList = categoryList
            work_dataframe = self.m_dataManager.m_dataFrame[mcUtility.addItemToList(tempList, self.m_dataManager.m_targetFieldName)]
    
    
            mcExplore.AnalyzeTargetNoneNumericFeatureList(work_dataframe, self.m_dataManager.m_targetFieldName, categoryList)
        
        transformToKeepList = self.m_controlFile.getPropertylist("exploreanalysis", "featureincluded")
        transformToRemoveList = self.m_controlFile.getPropertylist("exploreanalysis", "featureexcluded")
    
        self.m_projectTracking.updateFeatureUsageStatusList(toKeepList, "transform", 20)
        self.m_projectTracking.updateFeatureUsageStatusList(toRemoveList, "transform", 10)
        
        self.m_projectTracking.updateFeatureUsageStatusList(transformToKeepList, "transform", 1)
        self.m_projectTracking.updateFeatureUsageStatusList(transformToRemoveList, "transform", 2)
            
    
        #retrieve the option for  explore analysis
        functionAnalysisList = self.m_controlFile.getPropertylist("exploreanalysis", "transformfunction")
        
        
        if self.m_controlFile.getBooleanProperties('exploreanalysis', 'showtransform') == True:    
            mcExplore.showTransformOption_FeatureList(work_dataframe, self.m_dataManager.m_targetFieldName, toKeepList, functionAnalysisList)
    
    def processPCA(self):
        #start PCA 
        mPCA = mcPCA(self.m_dataManager.m_dataFrame)
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "Missing")
        RequireKeepList = self.m_controlFile.getPropertylist("pca", "featureincluded")
        RequireRemoveList = self.m_controlFile.getPropertylist("pca", "featureexcluded")
    
        self.m_projectTracking.updateFeatureUsageStatusList(RequireRemoveList, "PCA", 1)
        self.m_projectTracking.updateFeatureUsageStatusList(RequireKeepList, "PCA", 2)
        self.m_projectTracking.updateFeatureUsageStatusList(toRemoveList, "PCA", 10)
        self.m_projectTracking.updateFeatureUsageStatusList(toKeepList, "PCA", 20)
    
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "PCA")
    
        
        NonNumericFieldList = mcUtility.getAllNoneNumericFields(self.m_dataManager.m_dataFrame)
        toKeepList = mcUtility.removeIfInTheList(toKeepList, self.m_dataManager.m_targetFieldName)
        toKeepList=  mcUtility.removeListfromList(toKeepList, NonNumericFieldList)
    #    print(toKeepList)
        
        mPCA.doPCAAnalysis(self.m_dataManager.m_dataFrame,toKeepList, self.m_dataManager.m_targetFieldName,self.m_projectTracking, 2 )
        #self.m_projectTracking.updateFeatureUsageStatusList(strongCorrFeatureList, "Corr", 20)  #calculate to keep
        print(self.m_projectTracking.FeatureUsageMatrix)
    
    
    
    def processCorrmat(self):
    #Process CorrMat
        mcorrMatAnalysis = mcc(self.m_dataManager.m_dataFrame)
        
        strongTargetCorr = float(self.m_controlFile.getProperties('corrmat', 'strongtargetfeaturecorr'))
        strongFeatureCorr = float(self.m_controlFile.getProperties('corrmat', 'strongfeaturecorr'))
        mediumFeatureCorr = float(self.m_controlFile.getProperties('corrmat', 'mediumfeaturecorr'))
        weakfeaturecorr = float(self.m_controlFile.getProperties('corrmat', 'weakfeaturecorr'))
        
        toKeepList, toRemoveList = self.m_projectTracking.getToKeepRemoveFeatureList(self.m_dataManager.m_dataFrame, "Missing")
            
        work_dataframe = self.m_dataManager.m_dataFrame[toKeepList]
        
        corrmat = mcorrMatAnalysis.getCorrmatFromDataframe(work_dataframe)
    #    print(corrmat)
        mcc.displayCorrelationMap(corrmat)
        
        strongCorrmat = mcc.getStrongCorrmat(corrmat, self.m_dataManager.m_targetFieldName, threshod=strongTargetCorr)
        mcc.displayCorrelationMap(strongCorrmat)
        
        strongCorrFeatureList = strongCorrmat.columns.tolist()
        self.m_projectTracking.updateFeatureUsageStatusList(strongCorrFeatureList, "Corr", 20)  #calculate to keep
       
        
    #    finalCorrList = mcc.generateUniqueCorrListFromCorrMat(strongCorrmat, self.m_dataManager.m_targetFieldName, mediumThredsold=mediumFeatureCorr)
    def runClassification(self):
        toKeepList = []
        toRemoveList = []
        
        if self.m_controlFile.getBooleanProperties("classification", "processonlymandatory") == True:     
            toKeepList = self.m_controlFile.getPropertylist("classification", 'featureToProcessOnly')
            
        else:
            toKeepList, toRemoveList = getFinalToTrainFeatureList()
        
        self.m_dataConfig.m_onlyIncludeFeatureList = toKeepList
        self.m_dataConfig.m_excludeFeatureList = toRemoveList
        
#        m_ProcessClassfier = mcProcessClassfier(self.m_dataManager.m_dataFrame, self.m_dataManager.m_targetFieldName, toKeepList)
        m_ProcessClassfier = mcProcessClassfier(self.m_dataManager, self.m_dataConfig )
           
#        m_ProcessClassfier.runLogisticClassifier()
#        m_ProcessClassfier.runBaggingClassifier()
#        m_ProcessClassfier.runDecisionTreeClassifier()
#        m_ProcessClassfier.runGaussianNBClassifier()
#        m_ProcessClassfier.runKNeighborsClassifier()
#        m_ProcessClassfier.runRandomForestClassifier()
#        m_ProcessClassfier.runSGDClassifier()
#        m_ProcessClassfier.runSVCClassifier()
#        m_ProcessClassfier.runSVMClassifier()
#        m_ProcessClassfier.runAdaBoostClassifier()
        m_ProcessClassfier.runxgBoostClassifier()
    
        m_ProcessClassfier.runKerasNNClassifier()
        
        
    def runRegression(self):
        
        toKeepList = []
        toRemoveList = []
        
        if self.m_controlFile.getBooleanProperties("regression", "processonlymandatory") == True:     
            toKeepList = self.m_controlFile.getPropertylist("regression", 'featureToProcessOnly')
            
        else:
            toKeepList, toRemoveList = self.getFinalToTrainFeatureList()
        
        
        mcRegressor = mcProcessRegressor(self.m_dataManager.m_dataFrame, self.m_dataManager.m_targetFieldName, toKeepList)
        
        mcRegressor.runLinearRegressor()
        mcRegressor.runBayesianRidgeRegressor()
    #    mcRegressor.runSGDRegressor()
    #    mcRegressor.runLassoRegressor()
        mcRegressor.runElasticNetRegressor()
    #    mcRegressor.runLarsRegressor()
    #    mcRegressor.runARDRegressor()
    #    mcRegressor.runLassoRegressor()
        mcRegressor.runMLPRegressor()
        mcRegressor.runOlsRegressor()
    
    def runTensorAnalysis(self):
        tensorfile = self.m_controlFile.getProperties("tensorprocess", "filename")
        tsManager = tensorDataColumnManager(tensorfile, CategoryHandler)
        tsManager.doTraining(self.m_dataManager.m_dataFrame, self.m_dataManager.m_targetFieldName)
        return