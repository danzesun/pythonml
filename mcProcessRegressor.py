# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:46:58 2017

@author: Jason
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import ARDRegression
from sklearn.neural_network import MLPRegressor

from sklearn import preprocessing
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from mcUtility import mcUtility as mu
import plotly.plotly as py
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std



    
class mcProcessRegressor:
    
    def __init__(self,dataframe, targetFieldName, featureList):
#       self.m_dataFrame = dataframe
#       self.m_targetFieldName = targetFieldName
#       self.m_featureList = featureList
#       self.m_y = dataframe[targetFieldName]#.values
#       self.m_X= mu.clearAllNaField(dataframe[featureList])
#       tempX = self.m_X.values #returns a numpy array
#       min_max_scaler = preprocessing.MinMaxScaler()
#       x_scaled = min_max_scaler.fit_transform(tempX)
#       self.m_X = pd.DataFrame(x_scaled)      
#  
#       self.m_X_train, self.m_X_test, self.m_y_train, self.m_y_test =  train_test_split(self.m_X, self.m_y)
       self.setupTrainingData(dataframe, targetFieldName, featureList) 
        
        
    def setupTrainingData(self, dataframe, targetFieldName, featureList):
       self.m_dataFrame = dataframe
       self.m_targetFieldName = targetFieldName
       self.m_featureList = featureList
       self.m_y = dataframe[targetFieldName]#.values
#       print("Previous ", dataframe[featureList].columns)
       self.m_X= mu.clearAllNaField(dataframe[featureList])
#       print("After ", self.m_X.columns)
#       self.m_X= mu.clearAllNaField(self.m_X)
#      n = len(m_T.columns)-1
#       self.m_X = m_T.values[:,0:n] 
       xColumnName = self.m_X.columns
       tempX = self.m_X.values #returns a numpy array
       min_max_scaler = preprocessing.MinMaxScaler()
       x_scaled = min_max_scaler.fit_transform(tempX)
       self.m_X = pd.DataFrame(x_scaled)      
       self.m_X.columns = xColumnName
       self.m_X_train, self.m_X_test, self.m_y_train, self.m_y_test =  train_test_split(self.m_X, self.m_y)
       
       


    def dispalyModelResult(self, lm, predictY, score):
#        columnnamelist = list(self.m_X_train.columns)
        columnnamelist = self.m_X_train.columns
        ResultArray= lm.coef_


        modelTable = pd.DataFrame(list(zip(columnnamelist, ResultArray)), columns=['Features', 'estimatedCoefficients'])
        #show Result
        print(modelTable)
 
        if (lm.fit_intercept == True):
#           if (len(lm.intercept_)>0):
            print("Intercept = ", lm.intercept_)


        meanError = np.mean((self.m_y_test - predictY) ** 2)
        print("Mean Error = ", meanError)
        print("Score = ", score)
#        print("rsquared = ", lm.r)
        

    def displayPredictPlot(self, predictY):
        plt.scatter(self.m_y_test,predictY)
        plt.xlabel(self.m_targetFieldName)
        plt.ylabel("predicted " + self.m_targetFieldName)
        print("r2_score is ", r2_score(self.m_y_test,predictY))

  
    def displayResidualPlot(self, predictY, predictTraingY):
        plt.scatter(predictTraingY, predictTraingY - self.m_y_train, c='b', s=40, alpha=0.5)
        plt.scatter(predictY, predictY - self.m_y_test, c='g', s=40)
        plt.hlines(y=0, xmin=0, xmax=50)
        plt.title("Residual Plot")
        plt.show()

    def runLinearRegressor(self):
        lm =  LinearRegression()
        
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)

        print("Linear Regression")
        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)
       
    def runBayesianRidgeRegressor(self):
        lm =  BayesianRidge(n_iter=300, compute_score = True, fit_intercept = True, normalize  = True)
        
        print("Ridge Regression")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)


        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)

    def runSGDRegressor(self):
        lm =  SGDRegressor(loss= 'squared_loss', penalty='l2', 
                            fit_intercept = True)
        
        print("SGDRegressor\n")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)


        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)


    def runLassoRegressor(self):
        lm =  Lasso(alpha=1.0, fit_intercept=True, normalize=True)
        
        print("Lasso Regressor\n")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)

        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)

    def runLarsRegressor(self):
        lm =  Lars(fit_intercept=True, normalize=True)
        
        print("Lars Regressor\n")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)

        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)


    def runElasticNetRegressor(self):
        lm =  ElasticNet(fit_intercept=True, normalize=True)
        
        print("ElasticNet Regressor\n")
        reg = lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)
        

        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)


    def runARDRegressor(self):
        lm =  ARDRegression(fit_intercept=True, normalize=True)
        
        print("runARDRegressor\n")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)

        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)

    def runLassoRegressor(self):
        lm =  Lasso(alpha = 0.1, fit_intercept=True, normalize=True)
        
        print("Lasso\n")
        lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)

        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)

    def runOlsRegressor(self):
        lm = sm.OLS(self.m_y_train, self.m_X_train)
        
        print("OLS\n")
        reg = lm.fit()
#        predictY = lm.predict(self.m_X_test)
#        score = lm.score(self.m_X_test,self.m_y_test )
#        predictTraingY = lm.predict(self.m_X_train)

        print("Summary = ", reg.summary())
#        self.displayPredictPlot(predictY)
#        self.displayResidualPlot(predictY, predictTraingY)
#        self.dispalyModelResult(lm, predictY, score)

    def runMLPRegressor(self):
        lm =  MLPRegressor(hidden_layer_sizes=(250, ), activation='tanh', solver='adam', 
                           alpha=0.0001, batch_size='auto', learning_rate='constant',
                           learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                           shuffle=True, random_state=None, tol=0.0001, 
                           verbose=False, warm_start=False, momentum=0.9, 
                           nesterovs_momentum=True, early_stopping=False, 
                           validation_fraction=0.1, beta_1=0.9, beta_2=0.999)
                           
        
        print("MLPRegressor\n")
        reg= lm.fit(self.m_X_train,self.m_y_train )
        predictY = lm.predict(self.m_X_test)
        score = lm.score(self.m_X_test,self.m_y_test )
        predictTraingY = lm.predict(self.m_X_train)
        
        
        lm.coef_ = lm.coefs_
        lm.fit_intercept = lm.intercepts_ 
        self.displayPredictPlot(predictY)
        self.displayResidualPlot(predictY, predictTraingY)
        self.dispalyModelResult(lm, predictY, score)
