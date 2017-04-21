# -*- coding: utf-8 -*-

import numpy as np
import math


class motherFeatures : 

	def __init__(self,data,nb_previous) : 

		"""
		On suppose que 'data' ne contient que les valeurs qui nous intéressent, 
		du temps le plan ancien au temps le plus récent
		"""

		self.data = data
		self.nbPrevious = nb_previous
		self.nbFeatures = nb_previous

	def basicFeatures(self,point) : 
		vec = []
		for i in range(point - self.nbPrevious,point) : 
			vec.append(float(self.data[i,0]))
		return vec

class model1 : 

	def __init__(self,classSup,dataWithFeatures,h_20,h_5,z) : 

		self.dataWithFeatures = dataWithFeatures
		self.classSup = classSup
		self.h_20 = h_20
		self.h_5 = h_5
		self.z = z
		self.nbFeatures = self.classSup.nbFeatures + 2 + h_5 + h_20 + 1

	def getFeatures(self,point) : 
		vec = self.classSup.basicFeatures(point)
		# add the MA 20
		vec.append(self.dataWithFeatures[point,1])
		# add the MA 5
		vec.append(self.dataWithFeatures[point,2])
		# add the MA 20 for the h states previous
		for i in range(point - self.h_20,point) : 
			vec.append(float(self.dataWithFeatures[i,1]))
		# add the MA 5 for the h states previous
		for i in range(point - self.h_5,point) : 
			vec.append(float(self.dataWithFeatures[i,2]))
		# add the trend for the last z days
		vec.append(history_point(point,self.dataWithFeatures,self.z))
		return vec

class model2 : 

	def __init__(self,classSup,dataWithFeatures,mean,sigma) : 

		self.dataWithFeatures = dataWithFeatures
		self.classSup = classSup
		self.nbFeatures = self.classSup.nbFeatures +  1
		self.mean = mean
		self.sigma = sigma

	def getFeatures(self,point) : 

		vec = self.classSup.basicFeatures(point)
		vec.append(np.random.normal(loc=self.mean, scale=self.sigma, size=1))
		return vec

"""
Colonnes Finance data:
Open,High,Low,Close,Volume,Dividend,Split,Adj_Open,Adj_High,Adj_Low,Adj_Close,Adj_Volume
"""

# know the trend from the past z days
def history_point(point,dataWithFeatures,z) : 
	liste1 = list(dataWithFeatures[point-z:z,0])

	liste2 = []
	for elt in range(1,len(liste1)) : 
		c = liste1[elt] - liste1[elt - 1]
		liste2.append(c)
	
	allpos = True
	allneg = True
	for i in liste2 : 
		if i < 0 :
			allpos = False
		if i > 0 : 
			allneg = False

	b = 0.
	if allpos == True : 
		b = 1.
	if allneg == True : 
		b = -1.

	return b

# add moving average to the data
def SMA(dataTable, period):

	table = np.zeros((np.shape(dataTable)[0],np.shape(dataTable)[1]+1))
	table[:,:-1] = dataTable
	for i in range(period,np.shape(dataTable)[0]) : 
		sma = np.mean(dataTable[i - period:i,0])
		table[i,-1] = sma

	return table

# ************* modify the data file
def modifyDataFile() : 

	dataFile = open('data_finance.csv','rb')
	dataFile_csv = csv.reader(dataFile,delimiter=',')
	dataFile_csv.next()
	with open('data_finance_openPrices.csv','wb') as csvfile : 

		for ligne in dataFile :

			spamwriter = csv.writer(csvfile,delimiter=";")
			new_str = ligne[11:]
			c = new_str.find(',')
			spamwriter.writerow(ligne[11:11+c])

	dataFile.close()

def SMA(df, period,column):

    sma = df[column].rolling(window=period, min_periods=period).mean()
    return df.join(sma.to_frame('SMA-'+str(period)))