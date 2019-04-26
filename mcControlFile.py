# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:22:19 2017

@author: Jason
"""
from configobj import ConfigObj

#JC Control File
class mcControlFile:
    filename = ""
    def __init__(self, filename):
        """Return a control object whose name is *name*.""" 
        self.filename = filename
#        configspec = ConfigObj(filename, interpolation=False, list_values=False, _inspec=True)
        self.config = ConfigObj(filename)
        
    def readControlFile(self, filename):
        configspec = ConfigObj(filename, interpolation=False, list_values=False, _inspec=True)
        self.config = ConfigObj(config_filename, configspec=configspec)

       
    def getProperties(self, block, propetyName):

        section1 = self.config[block]
        value1 = section1[propetyName]
        return value1
 
    def getIntegerProperties(self, block, propetyName):    
        section1 = self.config[block]
        value1 = section1[propetyName]
        return int(value1)

    def getFloat32Properties(self, block, propetyName):
    
        section1 = self.config[block]
        value1 = section1[propetyName]
        return float(value1)

    def getPropertylist(self, block, propetyName):
#        listvalue = self.config.list_values
#        self.config.list_values = True
        section1 = self.config[block]
        value1 = section1[propetyName]
        if isinstance(value1,  str):
               value2 = [value1]
               return value2
#        self.config.list_values = listvalue
        return value1

    def getPropertyIntegerList(self, block, propertyName):
        strList = self.getPropertylist(block, propertyName)
        intList =[]
        for strItem in strList:
            i = int(strItem)
            intList.append(i)
        return intList
    
        
    def setProperty(self, block, propertyName, value):
        self.config[block][propertyName] = value
                
    def getDictInSection(self, block):
        section1 = self.config[block]
        return section1.dict()   
    

    def getBooleanProperties(self, block, propetyName, default=True):
       
        section1 = self.config[block]
        value1 = section1[propetyName]

        if value1.upper() =='TRUE':
            return True
        elif value1.upper() == 'FALSE':
            return False
        else:
            return default

#    def getSectionDict(self,block):
#        section1 = self.config[block]
#        return section1.dict
    

    def getSectionKeys(self,block):
        section1 = self.config[block]
        return section1.dict().keys()
    
        
