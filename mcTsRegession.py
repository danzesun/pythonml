# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:37:55 2017

@author: Jason
"""

import tempfile
import urllib
import pandas as pd
import tensorflow as tf
from tensorflow.python import *
from mcUtility import mcUtility as mu
import numpy as np

from mcControlFile import mcControlFile as mcTensorControl
from mcCategoryDataHandler import mcCategoryDataHandler  as mcCategoryHandler

import math

import numpy as np

from tensorflow.python.estimator import estimator


#This Column manager will manage all the column
#1) All the original column list
#2) Tensor that are associated wilthe original column
#3) 
class tensorDataColumnManager:
    def __init__(self, controlFilePath, CategoryHandler):
        self.m_feature_cols ={}
        self.m_CategoryHandler = CategoryHandler
        

        self.tensorControl = mcTensorControl(controlFilePath)
        self.loadControlFile()
        self.processTensor()
        
    #load controlFilePath
    
    def loadControlFile(self):
        
        self.m_original_all_column_list = self.tensorControl.getPropertylist("columnlist", "original_all_column_list")
        self.m_categorial_single_column_list = self.tensorControl.getPropertylist("columnlist", "categorial_single_column_list")        
        self.m_categorial_mixed_column_list = self.tensorControl.getPropertylist("columnlist", "categorial_mixed_column_list")
        print ("self.m_categorial_mixed_column_list =", self.m_categorial_mixed_column_list )
        self.m_numeric_column_List = self.tensorControl.getPropertylist("columnlist", "numeric_column_list")

        self.m_numeric_column_bucket_vocabulary_list = self.tensorControl.getSectionKeys("numeric_column_bucket_vocabulary")
        self.m_categorial_single_column_vocabulary_list = self.tensorControl.getSectionKeys("categorial_single_column_vocabulary")
        self.m_categorial_mixed_column_vocabulary_list = self.tensorControl.getSectionKeys("categorial_mixed_column_vocabulary")
        
 
        self.m_finalBaseColumnList = self.tensorControl.getPropertylist("final_tensor_list","base_column_tensor_list")
        self.m_finalCrossedColumnList = self.tensorControl.getPropertylist("final_tensor_list","crossed_column_tensor_list")
        self.m_finalBaseColumnTensorList =[]
        self.m_finalCrossedColumnTensorList = []

    def processTensor(self):

        for field in self.m_numeric_column_List:
            self.m_feature_cols[field]= self.createNumericColumn(field)

        #all  field in the numeric_bucket need to be in numeric_column Tensor
        for field in self.m_numeric_column_bucket_vocabulary_list:
            newFieldName = field + "_buckets"
            vocList = self.tensorControl.getPropertylist("numeric_column_bucket_vocabulary", field)

            if (len(vocList) >= 1):
                if (vocList[0].upper()=="AUTO"):
                    maxNum = 100
                    if (len(vocList) == 2):
                        maxNum = int(vocList[1])
                    
                    vocList = self.getCategoryVocList(field, maxNum)

            count = 0
            for i in vocList:
                vocList[count] = float(i)    
                count = count + 1

            self.m_feature_cols[newFieldName]= self.createNumericBucketColumn (self.m_feature_cols[field],  vocList)


        for field in  self.m_categorial_single_column_list:
            vocList = self.tensorControl.getPropertylist("categorial_single_column_vocabulary", field)
            if (len(vocList) >= 1):
                if (vocList[0].upper()=="AUTO"):
                    maxNum = 20
                    if (len(vocList) == 2):
                        maxNum = int(vocList[1])
                    
                    vocList = self.getCategoryVocList(field, maxNum)
                     
            self.m_feature_cols[field] = self.createCategorialColumnWithVocList(field,vocList)

#crossed_columns = [
#    tf.feature_column.crossed_column(
#        ["education", "occupation"], hash_bucket_size=1000),
#    tf.feature_column.crossed_column(
#        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
#    tf.feature_column.crossed_column(
#        ["native_country", "occupation"], hash_bucket_size=1000)
#]

#    there are 3 type of crossed_column  1) all single column 2) single column + tensor  
# 3) single + tensor  + CrossedColumn
            # Mix Category can either take a single field, a categorial field, 
            # a bucket field, and/or numeric field            
#categorial_mixed_column_list=bedroomcnt_buckets, fullbathcnt_buckets,  garagecarcnt_buckets
        print(self.m_categorial_mixed_column_list)
        for field in self.m_categorial_mixed_column_list:
            vocList = self.tensorControl.getPropertylist("categorial_mixed_column_vocabulary", field)
            tensorList =[]
            strList = []
           
            for voc in vocList:
                if (voc in self.m_feature_cols.keys()):                
                    tensorList.append(self.m_feature_cols[voc])
                else:
                    strList.append(voc)
                
#            strList.append(field)       
            self.m_feature_cols[field]=self.createCrossedColumnfromStrTensor(strList, tensorList)

        for field in self.m_finalBaseColumnList:        
            self.m_finalBaseColumnTensorList.append(self.m_feature_cols[field])

        for field in self.m_finalCrossedColumnList:
            self.m_finalCrossedColumnTensorList.append(self.m_feature_cols[field])


                                                
    def getCategoryVocList(self,  categoryField, maxNum):
       return self.m_CategoryHandler.getCategoryFieldVocList(categoryField, maxNum)
        
        

#Buckets include the left boundary, and exclude the right boundary. 
#   Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).

    def createNumericBucketColumn(self, field_numeric_Tensor, boundariesNumericList ):
        return tf.feature_column.bucketized_column(field_numeric_Tensor, boundaries = boundariesNumericList)

#Use this when your inputs are in string or integer format, and you have an in-memory vocabulary 
#mapping each value to an integer ID
    def createCategorialColumnWithVocList(self, fieldName, vocList):
        vocList = [x for x in vocList if x != np.nan]
        vocList = [int(x) for x in vocList if x != float('nan')]
#        
        return tf.feature_column.categorical_column_with_vocabulary_list(fieldName, vocList)
 
#       return tf.feature_column.categorical_column_with_vocabulary_list(fieldName, vocList)

    def createNumericColumn(self, fieldname):
        return tf.feature_column.numeric_column(fieldName)

    def createNumericColumn(self, fieldName):
        return tf.feature_column.numeric_column(fieldName)

    def createWeighted_categorical_column(self, fieldName):
        return tf.weighted_categorical_column(fieldName)
    

    def createCrossedColumn(self, columnList):
        return tf.feature_column.crossed_column(columnList, hash_bucket_size=1000)

    def createCrossedColumnfromStrTensor(self, columnStrList, columnTensorList):
        for tensor in columnTensorList:
            columnStrList.append(tensor) 
        return self.createCrossedColumn(columnStrList)


#Use this when your sparse features are in string or integer format, and you want to distribute 
#your inputs into a finite number of buckets by hashing. output_id = Hash(input_feature_string) % bucket_size#  "occupation", hash_bucket_size=1000)
    def createCategorialColumnHashBucket(self, fieldName, hash_bucket_size=1000):
        return tf.feature_column.categorical_column_with_hash_bucket("fieldName", hash_bucket_size)

    def addToBaseColumns(self, column):
        self.m_baseColumnList.append(column)    

    def addToCrossColumnList(self, crossColumn):
        self.m_crossColumnList.append(crossColumn)
    
    def loadOriginalColumnList(self, originalColumnList):
        self.m_OriginalColumnList = originalColumnList
    

#df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
#df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
#
#train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    def input_fn_fromDataFrame(self, dataframe, targetFieldName,  
                               batch_size=100, num_epochs=1,   
                               shuffle=None, num_threads=5):
    
        df = dataframe
#        mu.removeFieldFromDataframe(df, targetFieldName)
        labels = dataframe[targetFieldName]
#        labels = mu.removeIfInTheList(labels,  targetFieldName)
    
        return tf.estimator.inputs.pandas_input_fn(x=df, y = labels,    
           batch_size=batch_size, num_epochs=num_epochs,   shuffle=shuffle, 
           num_threads=num_threads )
        
#    def input_fn_fromCSV(data_file, num_epochs, shuffle):
#      """Input builder function."""
#      df_data = pd.read_csv(
#          tf.gfile.Open(data_file),
#    #      names=CSV_COLUMNS,
#          skipinitialspace=True,
#          engine="python",
#          skiprows=1)
#      # remove NaN elements
#    #  df_data = df_data.dropna(how="any", axis=0)
#      labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
#      return tf.estimator.inputs.pandas_input_fn(
#          x=df_data,
#          y=labels,
#          batch_size=100,
#          num_epochs=1,
#          shuffle=None,
#          num_threads=5)    
#    
    
    
    def doTraining(self, dataFrame,targetFieldName):
        model_dir = tempfile.mkdtemp()
    
        estimator  = tf.contrib.learn.LinearRegressor(
                model_dir=model_dir, 
                feature_columns=self.m_finalCrossedColumnTensorList + self.m_finalBaseColumnTensorList,
                )
#                optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
#                                          l1_regularization_strength=1.0,
#                                          l2_regularization_strength=1.0))
        
        # set num_epochs to None to get infinite stream of data.
        estimator.fit(input_fn=self.input_fn_fromDataFrame(dataFrame, targetFieldName, batch_size=100,
                                                num_epochs=None, shuffle=True))
    
        results = regression.evaluate( input_fn=self.input_fn_fromDataFrame(dataFrame, targetFieldName, batch_size=100,num_epochs=None, shuffle=True),steps=None)
 
        print("model directory = %s" % model_dir)
    
        for key in sorted(results):
           print("%s: %s" % (key, results[key]))
    
   
#    
