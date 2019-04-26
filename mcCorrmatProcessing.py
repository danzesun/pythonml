# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:26:16 2017

@author: Jason
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

import seaborn as sns


class mcCorrmatProcessing:
    def __init__(self, dataframe):
        self.m_dataframe = dataframe
        self.m_corrmat = []
    
    def displayCorrelationMap(corrmat):
       #correlation matrix
       
       f, ax = plt.subplots(figsize=(12, 9))
       sns.heatmap(corrmat, vmax=.8, square=True); 
       plt.show()
    #def getCorrmatFromDataframe(dataframe):
    #    #the following are all DataFrame
    #    pMap = getPearsonCorrmatFromDataframe(dataframe)
    #    sMap = getSpearmanCorrmatFromDataframe(dataframe)
    #    kMap = getKendallCorrmatFromDataframe(dataframe)
    #    return (pMap + sMap + kMap)/3
    
    def getCorrmatFromDataframe(self, dataframe):
       self.m_corrmat =  dataframe.corr(method='pearson')
       return self.m_corrmat
    
    def getPearsonCorrmatFromDataframe(self, dataframe):
       self.m_corrmat =  self.m_dataframe.corr(method='pearson')
       return self.m_corrmat
    
    #Spearman rather than comparing means and variances, 
    #Spearman's coefficient looks at the relative order of values 
    #for each variable. This makes it appropriate to use with 
    #both continuous and discrete data
    
    def getSpearmanCorrmatFromDataframe(dataframe):
       self.m_corrmat = dataframe.corr(method='spearman')
       return self.m_corrmat
    
    def getKendallCorrmatFromDataframe(dataframe):
       self.m_corrmat = dataframe.corr(method='kendall')
       return self.m_corrmat
    
    def getStrongCorrmat(fullcorrmat, targetFieldName,  threshod=0.02):
        finalMat = fullcorrmat.copy()   
    
        for column in fullcorrmat.columns:
            if ((abs(fullcorrmat[targetFieldName][column]) < threshod) or (fullcorrmat[targetFieldName][column] == np.nan )):
                finalMat.drop(column, axis = 1, inplace=True)
                finalMat.drop(column, axis = 0, inplace=True)
        return finalMat    

    def generateUniqueCorrListFromCorrMat(corrmat, targetFieldName, mediumThredsold=0.8):
        
         
        targetFieldCorrSerie = corrmat[:][targetFieldName]
        targetFieldCorrSerie.drop(targetFieldName, inplace=True)
        WorkingCorrSerie = targetFieldCorrSerie
#        print ("WorkingCorrSerie：", WorkingCorrSerie)
        FinalCorrSerie = WorkingCorrSerie
    
        for curFeature in targetFieldCorrSerie.keys():
            #look for other high corr feature
            WorkingCorSerie = FinalCorrSerie
            for featureToCompare in WorkingCorrSerie.keys():
                if (featureToCompare != curFeature):
                    if (abs(corrmat[featureToCompare][curFeature]) >= mediumThredsold):
                        #this is a highly corrected feature so we can remove it out from the list
                        #remove from FinalCorList
                        if (featureToCompare in FinalCorrSerie.keys() ):
                            FinalCorrSerie.drop(featureToCompare, inplace = True)
#                            print("Removing ",featureToCompare, " becuase it is highly correlated to ", curFeature)
                  
        print(FinalCorrSerie.keys().tolist())               
        return FinalCorrSerie.keys().tolist()            

    def doCorrAnalysisForTwoFeature(dataframe, feature1, feature2, weak=0.3, medium = 0.6):
        printing("doCorrAnalysisForTwoFeature")
        pCorrmat = dataframe.corr(method='pearson')
        kCorrmat = dataframe.corr(method='kendall')
        sCorrmat = dataframe.corr(method='spearman')
      
        PearsonCorr = pCorrmat.get_value(feature1, feature2)
        KendallCorr = kCorrmat.get_value(feature1, feature2)
        SpearmanCorr = sCorrmat.get_value(feature1, feature2)
        
         
        #Show Plot for two Feature
        figure1 = plt.figure(figsize=(10,10))
        ax1 = figure1.add_subplot(1,1,1)
        titletext = "Correlation Analysis for" + feature1 + " and " + feature2
        ax1.set_title(titletext, fontsize='xx-large')
        
    
    #    plt.plot(dataframe[feature1], dataframe[feature2])
        plt.ylabel(feature1)
        plt.xlabel(feature2)
        plt.scatter(dataframe[feature1], dataframe[feature2])
        
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        CorrText = "Pearson = " 
        CorrText += str(PearsonCorr)
        CorrText += "\n\nKendall = " 
        CorrText += str(KendallCorr )
        CorrText +="\n\nSpearman = "
        CorrText += str(SpearmanCorr)
        if (abs(PearsonCorr + KendallCorr + SpearmanCorr)/3 <weak):
              CorrText +="\nCorrelation is week "  
        elif (abs(PearsonCorr + KendallCorr + SpearmanCorr)/3  < medium):
              CorrText +="\nCorrelation is medium"
        else: 
              CorrText +="\nCorrelation is strong"
            
        plottext = plt.text(xmin+ 3/5* (xmax - xmin ), ymin+4/5 * (ymax- ymin), CorrText)
        plottext.set_fontsize('x-large')
        plottext.set_fontname('Times New Roman')
        plottext.set_color('blue')
        plottext.set_weight('bold')
        
        plt.legend()
        plt.show()
