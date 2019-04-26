# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:02:49 2017

@author: Jason
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from  mcTracking import mcProjTracking as mcProjTracking


import numpy as np
import pandas as pd
from sklearn import datasets
import mcUtility as mu

#This class takes care of 
class EncodingFieldManagement:
    
    def __init__(self, dataframe):
       self.m_dataframe = dataframe
       m_mappingDict = {}

    def getEncodm_edCategoryMapping(self, mappingDict, fieldName):
        mapping = self.m_mappingDict[fieldname]
        return mapping
    
    
    def registerNewField(fieldName):
        m_newFieldList.append(fieldName)
        
    def clearAllNewField(self, dataframe):
        for field in self.m_NewFieldList:
            dataframe.drop(field, inplace=True)
            
                
    def createLabelEncodeFields(dataframe, fieldName):
        labelEncoder = LabelEncoder()
        labelEncoder.fit(dataframe[fieldName])
        
        newFieldName = "LE_"
        newFieldName += fieldName
           
        newConvertedField = labelEncoder.transform(dataframe[fieldName]) 
        dataframe[newFieldName]=newConvertedField
        self.registerNewField(newFieldName)  
        
        mapping = dict(zip(labelEncoder.classes_, range(len(labelEncoder.classes_))))
        self.m_mappingDict[fieldname] = mapping
        
        return newFieldName, mapping

    def FE_oneHotEncodingDictory(dataframe, categoryDictFieldName):
        vec = DictVectorizer(sparse=False, dtype=int)
        vec.fit_transform(dataframe[categoryDictFieldName]).toarray()

    def convertCatergory_Text_to_Num(dataframe, categoryTextFieldName):    
       from sklearn.feature_extraction.text import CountVectorizer
#       dataframe = projectData.m_dataFrame
#       categoryTextFieldName= 'taxdelinquencyflag'
       vec = CountVectorizer()
       newVecMetrix = vec.fit_transform(dataframe[categoryTextFieldName])
       FeatureNameList = vec.get_feature_names()
       newFeatureNameList  = mcUtility.addPrefixstrToStringArray(FeatureNameList,categoryTextFieldName + "_" )

       newCatFrame = pd.DataFrame(newVecMetrix.toarray(), columns=newFeatureNameList)
       dataframe = pd.concat([dataframe, newCatFrame], axis = 1)
       
       print(newCatFrame)

               
class mcPCAAnlysis:

    def __init__(self, dataframe):
        self.m_dataframe = dataframe                   
    
     
    def clearAllNaField(self, dataframe):
        
        allNullColumns = dataframe.isnull().any()
        allNullColumns = allNullColumns[allNullColumns==True]
        
        for nullColumn in allNullColumns.index.tolist():
             dataframe[nullColumn].fillna(dataframe[nullColumn].mode()[0], inplace=True)
        return dataframe
    def getPCAComponentText(self, componentName, component, explained_variance_ratio, fieldNameList, mcProjTracking):
        nCount = 0
        FirstTime = True
        stage = componentName
        
        componentStr = componentName
        
        componentStr +=  "(varianace = "
        componentStr =  componentStr + str(round(explained_variance_ratio,3))
        componentStr =  componentStr + ") = "

        for weight in component:
            if round(weight,3) != 0.000:
                if FirstTime == False:                   
                    componentStr = componentStr + ' + '
                else:
                    FirstTime = False
                componentStr = componentStr + str(round(weight,3))
                componentStr = componentStr +  " * "
                componentStr = componentStr + fieldNameList[nCount]
                mcProjTracking.updateFeatureUsageStatus(fieldNameList[nCount], stage, weight, True)
            nCount = nCount + 1
      
        return componentStr
    def setupTrainingData(self, X, y):
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def doPCAAnalysis(self, dataframe, fieldNameList, targetFieldName, mcProjTracking, n_components=2):
        from sklearn.model_selection import train_test_split
        
#        fieldNameList = toKeepList
#        dataframe = projectData.m_dataFrame
#        targetFieldName = 'logerror'
        
        
        pcaDataframe =  dataframe[fieldNameList]
        print("\nPCA DataFrame before rowcount = {}".format(len(pcaDataframe)))
#        pcaDataframe.dropna(axis=0, how='any', inplace=True)
#        pcaDataframe = pcaDataframe.fillna(lambda x: x.median(), inplace=True)
        pcaDataframe = self.clearAllNaField(pcaDataframe)
        print("\PCA DataFrame after rowcount = {}".format(len(pcaDataframe)))

        n = len(pcaDataframe.columns)-1

        X_pca = pcaDataframe.values[:,0:n]  # 
        y_pca = dataframe[targetFieldName]

        X_pcaTrain, X_pcaTest, y_pcaTrain, y_pcaTest = \
        train_test_split(X_pca,  y_pca, test_size=.4, random_state=42)
        
#        X_pcaTrain, X_pcaTest, y_pcaTrain, y_pcaTest =  self.setupTrainingData(X_pca, y_pca)
#        print("original shape:   ", X_pcaTrain.shape)
     
        pca = PCA(n_components, svd_solver='randomized')
        X_pcaTrain_fit =pca.fit_transform(X_pcaTrain, y_pcaTrain)
        print('ValueError')
        print(self.getPCAComponentText("PCAComp1", pca.components_[0], pca.explained_variance_ratio_[0], fieldNameList, mcProjTracking))
        print(self.getPCAComponentText("PCAComp2", pca.components_[1], pca.explained_variance_ratio_[1], fieldNameList, mcProjTracking))

    
            
    

#catogorical data
#creates extra columns indicating the presence or absence of a category with a value of 1 or 0
# categoryDictFieldName  is a dictory column
# after onHotEncoding, 
            
#
    

       
    
