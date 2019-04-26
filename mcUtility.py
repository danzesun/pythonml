# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 02:35:17 2017

@author: Jason
"""
import numpy as np
import pandas as pd
from sklearn import datasets

class mcUtility:
    def removeIfInTheList(ListToCheck, itemToRemove):    
        try:
            ListToCheck.remove(itemToRemove)
        except ValueError:
            return ListToCheck
        return ListToCheck

    def removeFieldsFromDataframe(dataframe, removeList) :
        for field in removeList:
            try:
                dataframe.drop(field,  inplace=True, axis=1)
#                print("removing..", field)
            except ValueError:
                continue

    def removeFieldFromDataframe(dataframe, field) :
        
        try:
            dataframe.drop(field,  inplace=True, axis=1)
#           print("removing..", field)
        except ValueError:
            return
                
    def removeListfromList(mainList, listToDel):
        for field in listToDel:
            try:
                mainList.remove(field)
            except ValueError:
                continue
        return mainList

    def addListToList(mainList, listToAdd):
        for field in listToAdd:
            try:
                if (field in mainList) == False:
                    mainList.append(field)
            except ValueError:
                continue
        return mainList

    def addItemToList(mainList, itemToAdd):
        try:
            if (itemToAdd in mainList) == False:
                mainList.append(itemToAdd)
        except ValueError:
            return mainList
        return mainList

    def addPrefixstrToStringArray(stringArray, strPrefix):
        i=0  
        for strElement in stringArray:
           newElement = strPrefix + strElement
           stringArray[i] = newElement
           i=i+1
        return stringArray
    #extra a column from matric           
    def extractArrayFromList(A, column):
    
        finalArray=[]
    
        for i in range( 0, len(A)):
            finalArray.append(A[i][column])
    
        return finalArray

    #givn a feature with nXSample sample, and another feature of nYSample,
    #this will create a nxSample x  nYSample dots of sample point filling out 
    #the space
      
    def getall2DSpaceDots(x1, x2, nXSample, y1, y2, nYSample):
        # start at 3, end at 9, 100 sample
        xx = np.linspace(x1, x2, nXSample)
        # start at 1, end at 5, transpose- very important
        yy = np.linspace(y1, y2, nYSample).T
                    
                        #why meshgrip                
        xx, yy = np.meshgrid(xx, yy)
        
        #In the space of xx,  yy, create that many dots
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        
        return Xfull

    def addMappingDict(mappingDict, fieldName, mapping):
        mappingDict[fieldName] = mapping


    def getAllNumericFields(dataframe):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
        dataframe_num_Column = dataframe.select_dtypes(include=numerics)
        relatedFields = dataframe_num_Column.columns.values.tolist()
        return relatedFields

    def isNumericField(dataframe, field):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
        dataframe_num_Column = dataframe.select_dtypes(include=numerics)
        relatedFields = dataframe_num_Column.columns.values.tolist()
        
        if any(field in feature for feature in relatedFields):
            return True
        return False
 
    
    def getAllNoneNumericFields(dataframe):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
        dataframe_num_Column = dataframe.select_dtypes(exclude=numerics)
        relatedFields = dataframe_num_Column.columns.values.tolist()
        return relatedFields
            
    def getAllBooleanFields(dataframe):
        numerics = ['bool_']
        dataframe_num_Column = dataframe.select_dtypes(include=numerics)
        relatedFields = dataframe_num_Column.columns.values.tolist()
        return relatedFields
    
    def addMappingDict(mappingDict, fieldName, mapping):
        mappingDict[fieldName] = mapping
 
    def getEncodedCategoryMapping(mappingDict, fieldName):
        mapping = mappingDict[fieldname]
        return mapping

    def createlogarizeField1(dataframe, featureFieldName): #to add option
        newFeatureName = "log_"
        newFeatureName += featureFieldName
    
        #copy all non 0 over, and leave 0 intact
        dataframe["temp"] = pd.Series(len(dataframe[featureFieldName]), index=dataframe.index)
        dataframe["temp"] = 0
        dataframe.loc[dataframe[featureFieldName]>0, "temp" ] = 1      
        dataframe[newFeatureName] = dataframe[featureFieldName]
        dataframe.loc[dataframe["temp"]==1,newFeatureName] = np.log(dataframe[newFeatureName])
        dataframe.drop("temp", axis=1, inplace=True )
#       
        dataframe[newFeatureName][dataframe[featureFieldName] >0] = np.log(dataframe[featureFieldName][dataframe[featureFieldName] >0])
        
        #TransformData
        return dataframe[newFeatureName], newFeatureName
 
    
    def createlogarizeField(dataframe, featureFieldName): #to add option
        newFeatureName = "log_"
        newFeatureName += featureFieldName
    
        #copy all non 0 over, and leave 0 intact
        dataframe[newFeatureName] = 0
        dataframe.loc[dataframe[featureFieldName] >0,newFeatureName] = np.log(dataframe.loc[dataframe[featureFieldName] >0,featureFieldName])
        
        #TransformData
        return dataframe[newFeatureName], newFeatureName

    def createInverseField(dataframe, featureFieldName): #to add option
        newFeatureName = "Inverse_"
        newFeatureName += featureFieldName
    
#       
        dataframe[newFeatureName] = 0
#        dataframe[newFeatureName][dataframe[featureFieldName] >0] = np.divide(1,dataframe[featureFieldName][dataframe[featureFieldName] >0])
        dataframe.loc[dataframe[featureFieldName] >0,newFeatureName] = np.divide(1,dataframe[featureFieldName][dataframe[featureFieldName] >0])
       
        #TransformData
        return dataframe[newFeatureName], newFeatureName



    def isIntheList(a, alist):
        if (a in alist):
            return True
        else:
            return False
    
    def extractlistarray(A, nIndex):
    
        finalArray=[]
    
        for i in range( 0, len(A)):
            finalArray.append(A[i][nIndex])
    
        return finalArray

    def clearAllNaField(dataframe):
#        allNullColumns is a serie with len of columns of dataframe
        allColumns = dataframe.isnull().any()
        allNullColumns = list(allColumns[allColumns == True].index)
        
        if (len(allNullColumns) <= 0):
            return dataframe

        
        dataframe = dataframe[allNullColumns].fillna(dataframe[allNullColumns].mean())
 
        for nullColumn in allNullColumns:
            if mcUtility.isNumericField(dataframe, nullColumn):
                 dataframe[nullColumn].fillna(dataframe[[nullColumn]].mean(), inplace=True)
            else:
                 dataframe[nullColumn].fillna('missing', inplace=True)
        
        return dataframe
