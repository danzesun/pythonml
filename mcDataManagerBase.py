# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:04:56 2018

@author: Jason
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:54:25 2017

@author: Jason
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
#base class for loading data and setting up the dataframe
#every new data should have its own unless using the default loading
#categorythredshold = 200
#fieldexcluded=click_time
#displagcategorylist=true

class mcDataConfig:
    def __init__(self):
        self.m_ctrFileName = ""
        self.m_trainDataPath =""
        self.m_testDataPath=""
        self.m_test_size_ratio = 0.38
        self.m_doTrainTestSplitted = True
        self.m_targetFieldName = ""
        self.m_intputFormat = "csv"
        self.m_excludeFeatureList =[]
        self.m_onlyIncludeFeatureList =[]
        self.m_doOverSample = False
        self.m_doMinMaxScaling =False
        
    def setTestSizeRatio(testSizeRatio):
        self.m_test_size_ratio = testSizeRatio
    
    def setTargetFieldName(self, targetFieldName):
        self.m_targetFieldName = targetFieldName
    
    def getTargetFieldName(self):
        return self.m_targetFieldName;
    
         
class  mcDataManagerBase:

    
    def __init__(self,  dataConfig):
        self.m_X_All_Org = np.ndarray(1)
        self.m_y_All_Org = np.ndarray(1)
        self.m_X_test_Org = np.ndarray(1)
        self.m_y_test_Org = np.ndarray(1)

        self.m_X_train = np.ndarray(1)
        self.m_X_test =np.ndarray(1)
        self.m_y_train = np.ndarray(1)
        self.m_y_test = np.ndarray(1)
        
 
        self.m_dataConfig = dataConfig   
        self.m_dataFrame_test = pd.DataFrame()
        self.m_dataFrame_train = pd.DataFrame()
 
        self.m_dataFrame_test_view = pd.DataFrame()
        self.m_dataFrame_train_view = pd.DataFrame()
                
        
        self.loadData(dataConfig) 
         
    def loadData(self, dataConfig):
        self.m_dataConfig = dataConfig   
        self.m_dataFrame = self.ReadDataToDataFrame(dataConfig.m_trainDataPath, dataConfig.m_intputFormat)   
 
        if (dataConfig.m_doTrainTestSplitted == False and len(dataConfig.m_testDataPath) > 0):
            self.m_dataFrame_test = self.ReadDataToDataFrame(dataConfig.m_testDataPath, dataConfig.m_intputFormat)
            self.m_dataFrame_train = self.m_dataFrame
        else:
            self.m_dataFrame_test = self.m_dataFrame.sample(frac=dataConfig.m_test_size_ratio, weights=None)  
            self.m_dataFrame_train = self.m_dataFrame.drop(self.m_dataFrame_test.index)
    
        return True
 
  
#Call this function evertime we need to setup the data in a certain way for analysis or modeling
        
    def setupData(self, dataConfig):  
        
        self.setupFeatureFields(dataConfig)
        self.setupTargetField(dataConfig)

        #setup everything from dataframe and dataframe_Test if exist
        self.m_X_train =  self.m_dataFrame_train_view.values
        self.m_X_test =  self.m_dataFrame_test_view.values      
        self.m_y_train, self.m_y_test = self.setupTargetField(dataConfig)
 
   
        if dataConfig.m_doOverSample == True:
            self.OverSamplingIfNeeded()

  
          

 ##########################################################################  
# This is a virtual function for derived data class to overwrite
    def setupTargetField(self, dataConfig):
        self.preSetupTargetField(dataConfig)
        self.m_targetFieldName = dataConfig.m_targetFieldName
        y_train = self.m_dataFrame_train_view[dataConfig.m_targetFieldName]
        y_test = self.m_dataFrame_test_view[dataConfig.m_targetFieldName]
        self.postSetupTargetField(dataConfig)
        return y_train, y_test
 
    def preSetupTargetField(self, dataConfig):
        return
    def postSetupTargetField(self, dataConfig):
       return
   
    def preSetupFeatureFields(self, dataConfig):
        return
    def postSetupFeatureFields(self, dataConfig):
        return
       
# This is a virtual function for derived data class to overwrite
    def setupFeatureFields(self, dataConfig):
        self.preSetupFeatureFields(dataConfig)

        columnList = list(self.m_dataFrame_train.columns)
        
        if (len(self.m_dataConfig.m_onlyIncludeFeatureList) > 0):
            columnList = self.m_dataConfig.m_onlyIncludeFeatureList
        else:
            for field in self.m_dataConfig.m_excludeFeatureList:
                columnList.remove(field)    
            
#        columnList.remove(dataConfig.m_targetFieldName)
        nTotalColumn = len(columnList)         
        self.m_dataFrame_train_view = self.m_dataFrame_train[columnList]
        self.m_dataFrame_test_view = self.m_dataFrame_test[columnList]
         
 
        if dataConfig.m_doMinMaxScaling:
   #        self.m_X= mu.clearAllNaField(dataframe[featureList])
            xColumnName = self.m_dataFrame_train_view.columns
            tempX = self.m_dataFrame_train_view.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(tempX)
            self.m_dataFrame_train_view = pd.DataFrame(x_scaled)      
            self.m_dataFrame_train_view.columns = xColumnName
     
            tempX = self.m_dataFrame_test_view.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(tempX)
            self.m_dataFrame_test_view = pd.DataFrame(x_scaled)      
            self.m_dataFrame_test_views = xColumnName

        self.postSetupFeatureFields(dataConfig)
           
    def OverSamplingIfNeeded(self):
#        ratio='auto', random_state=None, k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1
        over_samples = SMOTE(random_state=1234)  
        self.m_X_train, self.m_y_train  = over_samples.fit_sample(self.m_X_train , self.m_y_train )

    def UnderSamplingIfNeeded(self):
        return

    
    def ReadDataToDataFrame(self, dataFilePath, fileFormat):
        dataFrame = pd.DataFrame() 
        if (fileFormat == 'csv'):
            dataFrame = pd.read_csv(dataFilePath)
        elif (fileFormat == 'excel'):
            dataFrame = pd.read_excel(dataFilePath)    
        elif (fileFormat == 'JSON'):
           dataFrame = pd.read_json(dataFilePath)    
         
        return dataFrame
            
       
  