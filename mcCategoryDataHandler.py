# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:28:58 2017

@author: Jason
"""
import numpy as np
import pandas as pd
import scipy as s


class mcCategoryDataHandler:
    def __init__(self, dataframe, categoryMaxCount = 50, displayCategoryCountList=False):
        self.m_dataFrame = dataframe
        self.m_categoryList = []
        self.m_categoryItemCountList = []
        self.m_FeatureUniqueCountSeries =  pd.Series(s.zeros(len(dataframe.columns)), list(dataframe.columns))
        self.identifyCategoryFeatureList(categoryMaxCount, displayCategoryCountList)
        
        
    def identifyCategoryFeatureList(self,categoryMaxCount = 50, displayCategoryCountList = True):
        self.m_itemCountList = []
       

        for feature in self.m_dataFrame.columns:
           self.m_FeatureUniqueCountSeries[feature] = len(self.m_dataFrame[feature].unique())
    
        self.m_categoryList=[]
        for feature in self.m_FeatureUniqueCountSeries.index:            
            if self.m_FeatureUniqueCountSeries[feature] <= categoryMaxCount:    
                   self.m_categoryList.append(feature)
                   self.m_categoryItemCountList.append(self.m_FeatureUniqueCountSeries[feature])
        
        if displayCategoryCountList == True:  
            categoryTable = pd.DataFrame(list(zip(self.m_categoryList, self.m_categoryItemCountList)), columns=['Features', 'Unique number Count'])
            
            print(categoryTable)

        return self.m_categoryList

    def getCategoryFeatureList(self):
        return self.m_categoryList
    
    def isCategoryFeature(self, feature):
        if (feature in self.m_categoryList) == True:
            return True
        else:
            return False
    
    
    # automatic generate 
    def getCategoryFieldVocList(self, feature, maxNum):
         self.m_dataFrame [feature].dropna(inplace=True)
         vocList = list(self.m_dataFrame [feature].unique())
         vocList = [x for x in vocList if x != float('NaN')]
         if (len(vocList) > maxNum):
             vocList.sort(reverse=True)
             vocList = vocList[0:(maxNum-1)]
         vocList.sort(reverse=False)
         return vocList
    