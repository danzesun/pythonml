# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:10 2017

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

from mcDataframe import mcDataframe
from mcUtility import mcUtility as mcUtility

    
class mcExploreAnalyze:
 
    def __init__(self,  dataframe):
        self.m_dataFrame = dataframe
        
        
    def DisplayHistogramForField(self, Field):

        Fieldinfo = self.m_dataFrame[Field].describe()
        print(Fieldinfo)
        
        #histogramHistogram
        print ("Purpose of scatter plot:  to see the distribution of a field and if there are any outlier")
        print ("Purpose the distribution plot:  to see if the target field is normally distributed" )
        figure1 = plt.figure(figsize=(12, 12))
        try:
            tempF = self.m_dataFrame[Field]
#            tempF.dropna(inplace=True)
            sns.distplot(tempF)     
            res = stats.probplot(tempF, plot=plt)
            plt.show()
        except:
           print("Unexpected error:", Field)       
           return
    
    #    sns.set()
    #    sns.pairplot(dataframe[relatedFields], size = 3.5, kind='reg')
    #    plt.show();
    
        
    #relatedFields a List
    def AnalyzeTargetFeatureRelation(self, dataframe, targetField,  featureFieldList, isTargetDecreted = False):
        
        if ((featureFieldList is None)):
            featureFieldList = dataFrame.columns.values.tolist()
            featureFieldList.remove(targetField)
    
       
       
        for featureField in featureFieldList:
            figure1 = plt.figure(figsize=(16, 8))
            ax1 = figure1.add_subplot(1, 2, 1)
            ax1.set_xlabel(featureField, fontsize='large')
            ax1.set_ylabel(targetField, fontsize='large')
            self.AnalyzeTargetToNumericFeature(dataframe, targetField, featureField)
            
            ax2 = figure1.add_subplot(1, 2, 2)
            ax2.set_ylabel(targetField, color='red')
            ax2.set_xlabel(featureField, color='blue')
            self.AnalyzeTargetTocategoricalFeature(dataFrame, targetField, featureField)
            mcCorrmatProcessing.doCorrAnalysisForTwoFeature(dataFrame, targetField, featureField )
            plt.legend()
            plt.show()
  
    #    plt.tight_layout()
        
    def AnalyzeTargetToNumericFeature(self, dataframe, targetField, relatedField):
        plt.scatter(dataFrame[relatedField], dataFrame[targetField])
       
    
    def AnalyzeTargetTocategoricalFeature(self, dataFrame, targetField, relatedField):
        #box plot 
        data = pd.concat([dataFrame[targetField], dataFrame[relatedField]], axis=1)
        sns.boxplot(x=relatedField, y=targetField, data=data)
    #This function will display the scatter Plot between Target and all feature
    
    def justNumericFields(self, dataframe, relatedFields, targetField, isNumeric= True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8', 'uint16', 'uint32', 'uint64','bool_', 'int_', 'complex_', 'complex64', 'complex128','intc', 'intp']
        dataframe_num_Column = dataframe.select_dtypes(include=numerics)
        numericFields = dataframe_num_Column.columns.values.tolist()
        
        workingList = relatedFields[:] 
        for field in relatedFields:
            if field == targetField:
                 workingList.remove(field)
                 continue
            if any(field in feature for feature in numericFields) == isNumeric :
                continue
            else:
                workingList.remove(field)
        return workingList
        
        
    def AnalyzeTargetNumericFeatureList(self, dataframe,  targetField, relatedFields):   
         #Each Plot to be 3 chart
        print("Scatter plot for numeric feature/target field")

        featureFieldList = self.justNumericFields(dataframe, relatedFields, targetField, True)
        print('featureFieldList', featureFieldList)
       
        sns.pairplot(dataframe, x_vars=featureFieldList, y_vars=targetField, diag_kind="kde", kind='reg', size=7, aspect=0.7)
        return
       
        
        plotPerRow = 2
        plotWidth = 8
        plotHeight = 8
        totalPlot = len(featureFieldList)
        totalRow =  int(totalPlot/plotPerRow + 0.5)
        totalPlot = int(totalRow * plotPerRow)
        plotID =  1
 
        figure1 = plt.figure(figsize=(plotPerRow * plotWidth, totalRow * plotHeight))
        figure1.suptitle("Target Feature Analysis")
        
        for featureField in featureFieldList:
#            print(featureField)
            ax1 = figure1.add_subplot(totalRow, plotPerRow, plotID)
            plotID = plotID + 1
            Title = targetField
            Title += " vs "
            Title += featureField
            ax1.title.set_text(Title) 
            
            data = pd.concat([dataframe[targetField], dataframe[featureField]], axis=1)
            data.columns = [targetField, featureField]
            data.dropna(how='any', inplace=True)
            
            try:
                plt.scatter(data[featureField], data[targetField])
            except ValueError:
                print("Error in", featureField )
                continue
            except:
                print("Unexpected error:", featureField)       
                continue

#            plt.tight_layout()
        plt.show()
        
#        plt.legend()
   
            
    def AnalyzeTargetAllNumericFeatures(self, targetField):   

        relatedFields = mcDataframe.getNumericFieldList(self.m_dataFrame, targetField)
        self.AnalyzeTargetNumericFeatureList(self.m_dataFrame, targetField, relatedFields )

    def AnalyzeTargetNoneNumericFeatureList(self, dataframe, targetField, relatedFields):
        print("This is to display the boxplot plot for every categorial feature against the target field\n")
        #Each Plot to beNoneNone 3 chart
#        featureFieldList = self.justNumericFields(dataframe, relatedFields, targetField, False)
        featureFieldList = relatedFields
        print(featureFieldList)
        
        
        plotWidth = 8
        plotHeight = 8 
        plotPerRow= 3
        totalPlot = len(featureFieldList)
        totalRow = int(totalPlot/plotPerRow + 0.5)
        totalPlot = int(totalRow * plotPerRow)
        plotID =  1
        
        figure1 = plt.figure(figsize=(plotPerRow * plotWidth, totalRow * plotHeight))
        
        for featureField in featureFieldList:
            if (featureField == targetField ):
                continue
            ax1 = figure1.add_subplot(totalRow, plotPerRow, plotID)
            print(plotID, " ", featureField)
            plotID = plotID + 1
            
#        self.AnalyzeTargetTocategoricalFeature(dataframe, targetField, iFeature)
          #box plot 
            data = pd.concat([dataframe[targetField], dataframe[featureField]], axis=1)
            data.dropna(inplace=True)
            try:
               sns.stripplot(x=featureField, y=targetField, data=data, jitter=True)
#               sns.barplot(x=featureField, y=targetField, data=data)
#               sns.swarmplot(x=featureField, y=targetField, data=data)
            except ValueError as err:        
               print ("Value Error :", featureField, " ", err)
               continue
            except OverflowError as err:
               print("OverflowError ",featureField , " ", err)
               continue 
            except:
               print("Unexpected error1:", featureField )  
               continue             
            
            Title = targetField
            Title += " vs "
            Title += featureField
            ax1.title.set_text(Title) 

#        plt.tight_layout()
        plt.show()
        
#        plt.legend()

    
    def AnalyzeTargetAllNoneNumericFeatures(self, targetField):
        relatedFields = mcDataframe.getNoneNumericFieldList(self.m_dataFrame, targetField)
        AnalyzeTargetNoneNumericFeatureList  (self.m_dataFrame, relatedFields, targetField)
        
        
    def AnalyzeTargetWithAllFeatures(self, targetField,  type="all"):
    
        if (type=="all" or type == "numeric"):
            self.AnalyzeTargetAllNumericFeatures(targetField)
        if (type=="all" or type == "nonnumeric"):
            self.AnalyzeTargetAllNoneNumericFeatures(targetField)
    #The following are non Numeric Data    
        
        
            
    def showTransformOption_FeatureList(self, dataframe, targetField, featureFieldList, functionAnalysisList):

#        featureFieldList =['landtaxvaluedollarcnt' ]
        for feature in featureFieldList:
            if (feature == targetField):
                continue
            if (mcUtility.isNumericField(dataframe, feature)==False):
                continue
            
            self.showTransformOption_Feature(dataframe, targetField, feature, functionAnalysisList)
        
        
        #   dataframe[logTargetFileName] = np.log(dataframe[targetFieldName])  
    def showTransformOption_Feature1(self, dataframe, targetField, featureField, 
                                    functionAnalysisList):
        if (targetField.upper() == featureField.upper()):
            return

        data = pd.concat([dataframe[targetField], dataframe[featureField]], axis=1)
        data.dropna(inplace = True, how='any')
        data.columns = [targetField, featureField]
      
 
        newField = ''
        locIndex = -1
        for f in functionAnalysisList:
            locIndex = locIndex+1
            tranformFieldData = data[featureField]
            
            if (f.upper() == 'LOG'):
                tranformFieldData, newField = mcUtility.createlogarizeField(data, featureField)
            elif (f.upper() =='INVERSE'):
                tranformFieldData, newField = mcUtility.createInverseField(data, featureField)                
            else:
                 newField =''
                 if (f.upper() == 'ORIGINAL'):
#                     functionAnalysisList[locIndex] = featureField
#                    tranformFieldData = data[featureField]
                     continue
                 else:
                    transformFunction = getFormulaFunction(f)
                    tranformFieldData = transformFunction(data[featureField])
            data[f.upper()] = tranformFieldData

        fig1 = plt.figure(figsize=(len(functionAnalysisList) * 6,  12 ))
        fig1.suptitle(featureField)
        ax0 = fig1.add_subplot(1, 1, 1)
        sns.pairplot(data,  dropna=True, hue=targetField)
        
        plt.show()
        
        if len(newField) > 0:
            if mcUtility.isIntheList(newField, list(dataframe.columns)):
                dataframe.drop(newField, axis=1, inplace=True)

    def showTransformOption_Feature(self, dataframe, targetField, featureField, 
                                    functionAnalysisList):
        if (targetField.upper() == featureField.upper()):
            return

        data = pd.concat([dataframe[targetField], dataframe[featureField]], axis=1)
        data.dropna(inplace = True, how='any')
        data.columns = [targetField, featureField]
      
        plotPerRow = 2
        totalPlot =    len(functionAnalysisList) + 1
#        totalRow = int((totalPlot + plotPerRow - totalPlot % plotPerRow) / plotPerRow)
        totalRow = int(totalPlot/plotPerRow + 0.5)
#        functionAnalysisList.insert(0,'Original')
        tranformFieldData = data[featureField]
        
        
        plotID = 1
        fig1 = plt.figure(figsize=(plotPerRow * 8,  totalRow * 8 ))
#        fig1 = plt.figure(figsize=(16,  12 ))
        fig1.suptitle(featureField)
        ax0 = fig1.add_subplot(totalRow, plotPerRow, plotID)
        plt.title('Density Distribution')

        plotID = plotID + 1
        
        try:
            sns.distplot(tranformFieldData, color='blue', ax=ax0);
        except ValueError as err:  
           print("Value Error 3:" , featureField , " - " , err) 
#               plt.title ("Value Error 3:" + featureField + " - " + err)
#           plt.show()
#           continue
        except OverflowError as err:
           print("OverflowError 3:" , featureField ,  " - " , err) 
#                plt.title("OverflowError 3:" + featureField +  " - " + err)
#           plt.show()
#           continue 
        except:
           print("Unexpected error 3:", featureField )  
#           plt.show()
#           continue
#
#        plt.show()
 
        newField = ''
        
        for f in functionAnalysisList:
            tranformFieldData = data[featureField]
            
            if (f.upper() == 'LOG'):
                tranformFieldData, newField = mcUtility.createlogarizeField(data, featureField)
            elif (f.upper() =='INVERSE'):
                tranformFieldData, newField = mcUtility.createInverseField(data, featureField)                
            else:
                 newField =''
                 if (f.upper() == 'ORIGINAL'):
                    tranformFieldData = data[featureField]
                 else:
                    transformFunction = getFormulaFunction(f)
                    tranformFieldData = transformFunction(data[featureField])
           
            ax2 = fig1.add_subplot(totalRow, plotPerRow, plotID)            
            plotID = plotID + 1             

            plt.scatter(tranformFieldData, data[targetField])
            plt.title(f + ': ' + featureField + " vs " +  targetField )
            plt.ylabel(targetField)
            plt.xlabel(f + ': ' + featureField)
#            plt.show()
        
    
        plt.legend()  
        plt.show()
        
        if len(newField) > 0:
            if mcUtility.isIntheList(newField, list(dataframe.columns)):
                dataframe.drop(newField, axis=1, inplace=True)


def getFormulaFunction(fomulaName):
    fomulaName = fomulaName.upper()
    
    #function listFunction
    transformFunctionDict = {'LOG': np.log, 'SIN': np.sin, 'SQUARE': np.square,
                             'SQUARE ROOT': np.sqrt, 'CUBE ROOT':np.cbrt, 'INVERSE':np.invert, 'TAN':np.tan}
    
    return transformFunctionDict[fomulaName]
