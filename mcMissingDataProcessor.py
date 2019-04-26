    # -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 02:02:21 2017

@author: Jason
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#this class will take care of checking, processing and tracking of NULL value

class mcMissingDataProcessor:
  
    def __init__(self, dataframe):
        self.m_dataFrame = dataframe
        self.m_missingDataList = []
        self.m_MissingToRemoveList = []
        self.m_MissingToKeepList = []
        
        
    def updateMissingDataList(self, missing_treadshold = 0.15, displayChart = True, displayList = True):

        dataframe = self.m_dataFrame        
        total = dataframe.isnull().sum().sort_values(ascending=False)
 
       #isnull should have better name: hasnullvalue
        percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
        ListOnlyMissing = percent[percent >0]
    
        if (displayChart == True):            
            fg = plt.figure(figsize=(16, 22))      
            ax = fg.add_subplot(1,1,1)
            width = 0.9
            ypos = np.arange(len(ListOnlyMissing))
            plt.barh(ypos, ListOnlyMissing.tolist(), align='center', alpha=0.5)
            plt.yticks(ypos, ListOnlyMissing.index.tolist())
            plt.xlabel('Missing Percentage')
            plt.title('Missing Values')

        #ndarray
        nRow = 0
#        self.m_missingDataList = missing_data.index.values
        self.m_missingDataList = list(missing_data.index)
        
        ToKeepList = []
        ToRemoveList = []
        percentList = []
#        missingTable = pd.DataFrame(list(zip(columnnamelist, ResultArray)), columns=['Features', 'estimatedCoefficients'])

   
        for indexlabel in missing_data.index.values:
           if(missing_data[missing_data.columns[1]][nRow] > missing_treadshold):
#               print(indexlabel, " : ", '{:02.2f}'.format(missing_data[missing_data.columns[1]][nRow], '%'))
               ToRemoveList.append(indexlabel)
               percentList.append(missing_data[missing_data.columns[1]][nRow])               
           else:
               ToKeepList.append(indexlabel)
           nRow = nRow + 1
        
        if (displayList == True):
            if (len(percentList)):
                missingTable = pd.DataFrame(list(zip(ToRemoveList, percentList)), columns=['Features', 'Missing Percentage'])
                print("Feature with missing rate over ",missing_treadshold, "\n" )
                print(missingTable)

        self.m_MissingToRemoveList = ToRemoveList
        self.m_MissingToKeepList = ToKeepList
        
        return ToKeepList, ToRemoveList


    def getMissing_values_table(self): 
        mis_val = self.m_dataFrame.isnull().sum()
        mis_val_percent = 100 * self.m_dataFrame.isnull().sum()/len(self.m_dataFrame)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns 
    
        
    def fillMissingData(dataframe, fieldToProcess, value=None, method=None, axis=1, inplace=True):
        dataframe[fieldToProcess].fillna(value=None, method='ffill', axis=None, inplace=True)

    def fillMissingwithMean(dataframe, fieldList):
        for field in fieldList:
           mode = np.mean(dataframe[field])
           dataframe[field].fillna(value=mode, inplace=True)
 
    def fillNullwithMode(dataframe, fieldList):
        for field in fieldList:
           mode = np.mode(dataframe[field])
           dataframe[field].fillna(value=mode, inplace=True)
        
    def fillNullwithValue(dataframe, fieldList, value):
        for field in fieldList:
            dataframe[field].fillna(value, inplace=True)

    def dropAllNullFields(dataframe):
        dataframe.dropna(axis=1, how="all")
    
    def dropFeatures(dataframe, fieldList):
        for field in fieldList:
            dataframe.drop(field, 1, inplace=True)
    

    def getNullValuePercentage(dataframe, fieldToProcess):
        return dataframe[fieldToProcess].isnull().sum()/dataframe[fieldToProcess].isnull().count()
    
