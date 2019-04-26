00# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:26:42 2017

@author: Jason
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from mcUtility import mcUtility as mu
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D, Conv2D
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
# Generate dummy data
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from mcDataManagerBase import mcDataConfig
from mcDataManagerBase import mcDataManagerBase

class mcProcessClassfier:
    def __init__(self, dataManager, dataConfig):
       self.m_X_train = dataManager.m_X_train
       self.m_y_train = dataManager.m_y_train
       self.m_X_test = dataManager.m_X_test
       self.m_y_test = dataManager.m_y_test
       self.m_dataConfig = dataConfig
#    def __init__():
#    def __init__(self,dataframe, targetFieldName, featureList):
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
#        self.setupTrainingData(dataframe, targetFieldName, featureList) 
        
          
       
   
       
       
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
       
           

    def DisplayLinearModelingPlot( self, clf, title):
        w = clf.coef_[0]
        a = -w[0] / w[1]
    
        minX = min(extractArrayFromList(self.m_X, 0)) - 0.5
        maxX = max(extractArrayFromList(self.m_X, 0)) + 0.5
    #    print("minX", minX)
    #    print("maxX",  maxX)
        xx = np.linspace(minX,maxX, 50)
        yy = a * xx - clf.intercept_[0] / w[1]
        h0 = plt.plot(xx, yy, 'k-', label=title)
     #   print(xx)
     #   print(yy)
        plt.scatter(extractArrayFromList(X, 0),extractArrayFromList(self.m_X, 1),  c = self.m_y)
        plt.legend()
        plt.show() 
    

    
    def DisplayNoneLinearModelingPlot(self, clf, title):
        return
        w = clf.coef_[0]
        a = -w[0] / w[1]
    
        minX = min(extractArrayFromList(self.m_X_train, 0))
        maxX = max(extractArrayFromList(self.m_X_train, 0))
    #    print("minX", minX)
    #    print("maxX",  maxX)
        xx = np.linspace(minX,maxX, 50)
        yy = a * xx - clf.intercept_[0] / w[1]
        h0 = plt.plot(xx, yy, 'k-', label=title)
     #   print(xx)
     #   print(yy)
        plt.scatter(extractArrayFromList(X, 0),extractArrayFromList(self.m_X_train, 1),  c = self.m_y_train)
        plt.legend()
        plt.show() 
    

    def DisplayClassifierPlot(self, clf, title):
        return
        plotFigure = plt.figure(figsize=(3 * 3, 6))
        plt.subplots_adjust(bottom=.2, top=.95)
       
        y_pred = clf.predict(self.m_X_train)
    #    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
       
        minX1 = min(extractArrayFromList(self.m_X_train, 0))
        maxX1 = max(extractArrayFromList(self.m_X_train, 0))
        minX2 = min(extractArrayFromList(self.m_X_train, 1))
        maxX2 = max(extractArrayFromList(self.m_X_train, 1))
        
        Xfull = getall2DSpaceDots(minX1, maxX1, 100, minX2, maxX2, 100)
        probas = clf.predict_proba(Xfull)
        
        plt.suptitle(title)
      
        n_classes = np.unique(y_pred).size
        print(n_classes)
    
        for k in range(n_classes):
            plt.subplot(1, n_classes,  k + 1)
            plt.title("Class %d" % k)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                       extent=(minX1, maxX1, minX2, maxX2), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(self.m_X_train[idx, 0], self.m_X_train[idx, 1], marker='o', c='k')
    
    #    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    #    plt.title("Probability")
    #    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    
    #    plt.show()    
    #    plt.scatter(extractArrayFromList(X, 0),extractArrayFromList(X, 1),  marker='o', c='k')
    #    plt.legend()
        plt.show() 

    def DisplayClassifierMap(self, clf, title):
     
        figure = plt.figure(figsize=(12, 12))
#        X_train, X_test, y_train, y_test = \
#            train_test_split(self.m_X, self.m_y, test_size=.4, random_state=42)
    
        minX1, maxX1 = self.m_X[:, 0].min() - .5, X[:, 0].max() + .5
        minX2, maxX2 = self.m_X[:, 1].min() - .5, X[:, 1].max() + .5
    
        h = .02  # step size in the mesh
        Xfull = getall2DSpaceDots(minX1, maxX1, 100, minX2, maxX2, 100)
        xx, yy = np.meshgrid(np.arange(minX1, maxX1, h),
                             np.arange(minX2, maxX2, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FFFF00', '#9900FF'])
        ax = plt.subplot(1, 2, 1)
        ax.set_title(title)
        ax.scatter(self.m_X_train[:, 0], self.m_X_train[:, 1], c=self.m_y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(self.m_X_test[:, 0], self.m_X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
    
     #   ax = plt.subplot(1, 2, 2)
        clf.fit(self.m_X_train, self.m_y_train)
        score = clf.score(self.m_X_test, self.m_y_test)
    
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
        print(Z.shape)
        print(xx.shape)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
        # Plot also the training points
        ax.scatter(self.m_X_train[:, 0], self.m_X_train[:, 1], c=self.m_y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(self.m_X_test[:, 0], self.m_X_test[:, 1], c=self.m_y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
    
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
    #    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #           size=15, horizontalalignment='right')
     
        plt.tight_layout()
        plt.show()
  


    
    def ClassifierDetailReport(self, y_true, y_pred):
        print("\nAccuracy classification score =  {} ".format(accuracy_score(y_true, y_pred)))
        print("\nAverage precision (AP) =  {} ".format(average_precision_score(y_true, y_pred)))
        print("\nMatthews correlation coefficient (MCC) =  {} ".format(matthews_corrcoef(y_true, y_pred)))
        print("\nAverage Hamming loss =  {} ".format(hamming_loss(y_true, y_pred)))
         
   
        
        
        
    def runSVMClassifier(self, bDetailReport = False):
        clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
        clf.fit(self.m_X_train, self.m_y_train)
        y = clf.predict(self.m_X_test)    
        print("\n\nSVMClassifier\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
        
#       DisplayLinearModelingPlot(X_train, y_train, clf, "SVM Traning")
#       DisplayClassifierPlot(X_train, y_train, clf, "SVM Traning")
    
    def runSGDClassifier(self, bDetailReport = False):
        clf = SGDClassifier(loss="hinge", penalty="l2")
        clf.fit(self.m_X_train, self.m_y_train)
        y = clf.predict(self.m_X_test)    
        print("\n\nSGDClassifier\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
                
#        DisplayLinearModelingPlot(X_train, y_train, clf, "SGD Classifier")
    
    def runLogisticClassifier(self, bDetailReport = False):
        clf = LogisticRegression(C=1, penalty='l1')
        clf.fit(self.m_X_train, self.m_y_train)
        y = clf.predict(self.m_X_test)    
        print("\n\nLogisticRegression\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayLinearModelingPlot(X_train, y_train, clf, "Logistic Regression")
#        DisplayClassifierPlot(X_train, y_train, clf,"Logistic Regression")
    
    def runSVCClassifier(self, bDetailReport = False):
        clf = SVC(kernel='linear', C=1, probability=True, random_state=0)
        clf.fit(self.m_X_train, self.m_y_train)
        y = clf.predict(self.m_X_test)    
        print("\n\nSVCClassifier\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayLinearModelingPlot(X_train, y_train, clf, "SVC")
#        DisplayClassifierPlot(X_train, y_train, clf,"SVC")
       
    
    def runKNeighborsClassifier(self, bDetailReport = False):
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("\n\nKNeighborsClassifier\n\n",  classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf,"KNeighbors")
    
    def runRadiusNeighborsClassifer(self, bDetailReport = False):
        clf = RadiusNeighborsClassifier(radius=1.0)
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("\n\nRadiusNeighborsRegressor\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "RadiusNeighborsRegressor")
     
    def runGaussianProcessClassifier(self):
        kernel = 1.0 * RBF([1.0, 1.0]) 
        clf = GaussianProcessClassifier(kernel) 
        clf.fit(self.m_X_train, self.m_y_train)
        y = clf.predict(self.m_X_test)    
        print("\n\nGaussianProcessClassifier\n\n",  classification_report(self.m_y_test, y))
        
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "GaussianProcessClassifier")
        
    def runGaussianNBClassifier(self, bDetailReport = False):
        clf = GaussianNB() 
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("\n\nGaussianNB\n\n",  classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "Gaussian Naive Bay Classifier")
#        DisplayClassifierMap(X_train, y_train, clf, "Gaussian Naive Bay Classifier ")
    def runAdaBoostClassifier(self, bDetailReport = False):
        dt = DecisionTreeClassifier()
        clf = AdaBoostClassifier(n_estimators=200, base_estimator=dt,learning_rate=1)
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)
        print(clf.algorithm)
        print(clf.feature_importances_)
        print("\n\nAdaBoostClassifier\n\n",  classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "Gaussian Naive Bay Classifier")
#        DisplayClassifierMap(X_train, y_train, clf, "Gaussian Naive Bay Classifier ")
    
    def runBaggingClassifier(self, bDetailReport = False):
        clf = BaggingClassifier(GaussianNB(),
                                n_estimators =10, max_samples=0.3, max_features=1)
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("\n\nBaggingClassifier\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "BaggingClassifier")
        
    def runDecisionTreeClassifier(self, bDetailReport = False):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("\n\nDecisionTreeClassifier\n\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "Decision Tree")
      
    def runRandomForestClassifier(self, bDetailReport = False):
        clf = RandomForestClassifier ()
        clf.fit(self.m_X_train, self.m_y_train)    
        y = clf.predict(self.m_X_test)    
        print("RandomForestClassifier\n", classification_report(self.m_y_test, y))
        
        if (bDetailReport):
            ClassifierDetailReport(self.m_y_test, y)
#        DisplayClassifierPlot(X_train, y_train, clf, "RandomForestClassifier ")
#        DisplayClassifierMap(X_train, y_train, clf, "RandomForestClassifier ")

    def runxgBoostClassifier(self, bDetailReport = False):
        print("m_X_train  size", len(self.m_X_train))
        
        boosters=['gbtree', 'gblinear']
        
        for  depth in range(3, 4):
            for rate in (range ( 2,  3,  1)):
                for estimator in (range(220,240, 20)):
#                    for bster in boosters:
                    clf = XGBClassifier(max_depth=depth, learning_rate=(float(rate)/10),  n_estimators=estimator,   silent=True, objective='binary:logistic', seed=400)
                    clf.fit(self.m_X_train, self.m_y_train)    
                    y = clf.predict(self.m_X_test)    
                    print("\nxgBoostClassifier depth={} rate={} estimator={}\n".format(depth, (float(rate)/10), estimator))
                    print(classification_report(self.m_y_test, y))
                    print(clf.feature_importances_)
                    # plot
                    pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
                    pyplot.show()       
                    
                    plot_importance(clf)
                    if (bDetailReport):
                        self.ClassifierDetailReport(self.m_y_test, y)
                        
        

    def runKerasNNClassifier(self, bDetailReport = False):
    
        model = Sequential()  
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
#        CHANGE THEM TO 3 DIMENSION
#        xDim = X_train.shape[1]
        
#        X_train =  np.expand_dims(X_train, axis=1)
#        y_train = np.expand_dims(y_train, axis=1)
#        y_train = np.expand_dims(y_train, axis=1)

        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(1, 30 )))

        model.add(Dropout(0.1))
        model.add(Conv1D(16, kernel_size=2, activation='elu', padding='same'))
#        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu', input_dim=self.m_X_train.shape[1]))
#        model.add(Dropout(0.1))
        
#        model.add(Dense(32, activation='relu'))
#        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu'))

#        model.add(Dropout(0.5))
#        model.add(Dense(10, activation='relu'))
#        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        
#        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

#        X_test =  np.expand_dims(X_test, axis=1)
#        y_test = np.expand_dims(y_test, axis=1)
#        y_test = np.expand_dims(y_test, axis=1)
       
        model.fit(self.m_X_train, self.m_y_train,
                  epochs=10,
                  batch_size=32)
        score = model.evaluate(self.m_X_test, self.m_y_test, batch_size=32)
        
        print(score)
        y = model.predict(self.m_X_test)
        print (y[self.m_y_test==1])
        if (bDetailReport):
            self.ClassifierDetailReport(self.m_y_test, y)
