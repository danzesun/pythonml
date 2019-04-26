# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:02:05 2017

@author: Jason
"""
import pandas as pd
import numpy as np



class mcProjTracking:
    
    def __init__(self):
            
        self.NumberofField = 0
        self.TargetFieldName = ""
        self.NulLDeleteFields =[]
        self.CorrmatDeleteFields= []
        self.NullKeepList =[]
        self.NullRemoveList = []
        self.CorrKeepList =[]
        self.CorrRemoveList = []
        self.m_dataFrame=pd.DataFrame()
        
        self.Data_name =""
        self.numericFields =[]
        self.categoryFields = []
        self.FeatureUsageMatrix =  pd.DataFrame()
                 
        
        self.trackingListIndex = ["Missing", "Corr", "transform","PCA", "PCAComp1", "PCAComp2"]
        
        
    def setupFeatureTracking(self, dataframe):
    
        trackingList=[1,1,1,1]
        self.FeatureUsageMatrix =  pd.DataFrame(index = dataframe.columns, columns=self.trackingListIndex) 
        self.FeatureUsageMatrix[:][:]=0 
 
       # 0  original
        # 1  require to remove 
        # 2  require to Keep 
        # 00  original 
        # 10  caculate to remove 
        # 20  caculate to keep
     
    #status can be 0, 1, 2, 00, 11,12, 21, 22
    
    def updateFeatureUsageStatus(self, featureName, stage, status, override=False):
        if (override== True):
            self.FeatureUsageMatrix[stage][featureName] = status
            return        
            
        required = status % 10 
        calculated = status - (status % 10)
        oldRequired = self.FeatureUsageMatrix[stage][featureName] % 10
        oldCalculated = self.FeatureUsageMatrix[stage][featureName] - self.FeatureUsageMatrix[stage][featureName] % 10
                                               
        
        if  required >0  and calculated > 0:          
            self.FeatureUsageMatrix[stage][featureName]=calculated + required
                                  
        elif  required  <=0  and calculated > 0:
           self.FeatureUsageMatrix[stage][featureName] = oldRequired + calculated
        elif required > 0 and calculated <= 0:
           self.FeatureUsageMatrix[stage][featureName] = required + oldCalculated
                                  
    def getFeatureUsageStatus(self, featureName, stage):
        return self.FeatureUsageMatrix[stage][featureName]
    
    def addFeatureStage(self, dataframe, stage):
        self.FeatureUsageMatrix[stage] =stage
        ncount = 0
        for index in dataframe.columns:
            self.FeatureUsageMatrix.loc[ncount]=trackingList
            ncount=ncount+1
                          
    def updateFeatureUsageStatusList(self, featureList, stage, status,override=False):
#        featureList = transformToKeepList
#        stage = 'featuretransform'
#        status = 1        
        # 0  original
        # 1  require to remove 
        # 2  require to Keep 
        # 00  original 
        # 10  caculate to remove 
        # 20  caculate to keep
        
        if len(featureList) <= 0:
            return;
        if len(featureList) == 1:
            if featureList[0] =='':
                return
            
        
        for feature in featureList:
#            print (feature)
            self.updateFeatureUsageStatus(feature, stage, status, override)            
            

    def getToKeepRemoveFeatureList(self, dataframe, stage):
        iCount = 0
        statusList = self.FeatureUsageMatrix[stage]
        keepList= []
        removeList = []
        
        for status in statusList:
            required = status % 10
            calculated = status - (status % 10)

            if (required==2):
                keepList.append(dataframe.columns[iCount])
            elif (required == 1):
                removeList.append(dataframe.columns[iCount])
            else:
                if (calculated==10):
                    removeList.append(dataframe.columns[iCount])
                elif (calculated==20):
                    keepList.append(dataframe.columns[iCount])
                else:
                    keepList.append(dataframe.columns[iCount])

            iCount = iCount + 1
        
        return keepList, removeList

        # 10  caculate to remove 
        # 20  caculate to keep

            
    def getCalcuatedToKeepRemoveList(self, dataframe, stage):
        iCount = 0
        statusList = self.FeatureUsageMatrix[stage]
        keepList= []
        removeList = []
        
        for status in statusList:
            required = status % 10
            calculated = status - (status % 10)

            if (calculated==10):
                removeList.append(dataframe.columns[iCount])
            elif (calculated==20):
                keepList.append(dataframe.columns[iCount])

            iCount = iCount + 1
        
        return keepList, removeList
            
    def getAlllNoneZeroList(self, dataframe, stage):
        iCount = 0
        statusList = self.FeatureUsageMatrix[stage]
        keepList= []
        removeList = []
        
        for status in statusList:

            if (status != 0):
               keepList.append(dataframe.columns[iCount])

            iCount = iCount + 1
        
        return keepList
                