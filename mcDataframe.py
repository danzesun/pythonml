# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:31:12 2017

@author: Jason
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from scipy import stats
from scipy.stats import norm 

import seaborn as sns
from matplotlib.colors import ListedColormap
from mcCorrmatProcessing import mcCorrmatProcessing


class mcDataframe:
    
    def __init__(self, dataframe):
        self.m_Dataframe = dataframe
        
    def getFormulaOfField(fieldName, fomulaName):
        fomulaName = fomulaName.upper()
        
        #function list
        transformFunctionDict = {'LOG': np.log, 'SIN': np.sin, 'SQUARE': np.square,
                                 'SQUARE ROOT': np.sqrt, 'CUBE ROOT':np.cbrt, 'INVERSE':np.invert }
        
        return transformFunctionDict[fomulaName]
    
    def logarizeAField(dataframe, featureFieldName): #to add option
        newFeatureName = "log_"
        newFeatureName += featureFieldName
    
        dataframe["temp"] = pd.Series(len(dataframe[featureFieldName]), index=dataframe.index)
        dataframe["temp"] = 0
        dataframe.loc[dataframe[featureFieldName]>0, "temp" ] = 1      
        dataframe[newFeatureName] = dataframe[featureFieldName]
          
        #TransformData
        dataframe.loc[dataframe["temp"]==1,newFeatureName] = np.log(dataframe[newFeatureName])
        dataframe.drop("temp", axis=1 )
        return dataframe[newFeatureName]

    def logTransformTarget(dataframe, targetFieldName):
        logTargetFileName = "logof"
        logTargetFileName += targetFieldName;
        
        dataframe[logTargetFileName] = np.log(dataframe[targetFieldName])  
    #    sns.distplot(dataframe[targetFieldName], fit=norm, color='green');
        sns.distplot(dataframe[logTargetFileName], fit=norm, color='red');
        fig = plt.figure()
        res = stats.probplot(dataframe[logTargetFileName], plot=plt)
        plt.show()
               
    def getNumericFieldList(dataframe, targetField):   
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
    
        dataframe_num_Column = dataframe.select_dtypes(include=numerics)
        relatedFields = dataframe_num_Column.columns.values.tolist()
        if (relatedFields.count(targetField) >0):
            relatedFields.remove(targetField)
        return relatedFields
    
    def getNoneNumericFieldList(dataframe, targetField):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
        dataframe_non_num_Column = dataframe.select_dtypes(include=[object])    
        relatedFields = dataframe_non_num_Column.columns.values.tolist()
        
        if (relatedFields.count(targetField) >0):
            relatedFields.remove(targetField)

        return relatedFields
