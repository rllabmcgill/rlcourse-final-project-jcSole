# -*- coding: utf-8 -*-

import numpy as np
import csv
import math


"""
Code valable normalement quelque soit le data set, tant qu'on fournit un tableau (data) avec juste
les valeurs et un autre avec les features qui nous intéressent (comme le mois, etc)
Il faut adapter la définition des features en fonction des data qui nous intéressent. 
Voir le fichier features_weather pour un exemple de comment définir les features
"""

def transformData(datafile) : 
	# IMPORT DATA AVEC LES TEMPERATURES ET AJOUTER MOIS,ANNEES,SAISONS
	dataArray = np.zeros((3650,8))
	dataFile = open('data_weather.csv','rb')
	dataFile_csv = csv.reader(dataFile,delimiter=',')
	dataFile_csv.next()
	i = 0
	for ligne in dataFile : 
		if ligne.find(",") == 12 : 

			month = float((ligne[6:8]))
			day = float((ligne[9:11]))
			year = float((ligne[1:5]))
			temp = float((ligne[13:]))

			fall = 0.
			summer = 0.
			spring = 0.
			winter = 0.

			# spring
			if month == 4. or month == 5. : 
				spring = 1.
			if month == 3. and day >= 20. : 
				spring = 1.
			if month == 6. and day <= 20. : 
				spring = 1.

			# summer
			if month == 7. or month == 8. : 
				summer = 1.
			if month == 6. and day >= 21. : 
				summer = 1.
			if month == 9. and day <= 21. : 
				summer = 1.

			# fall
			if month == 10. or month == 11. : 
				fall = 1.
			if month == 9. and day >= 22. : 
				fall = 1.
			if month == 12. and day <= 20. : 
				fall = 1.	

			# winter
			if month == 1. or month == 2. : 
				winter = 1.
			if month == 12. and day >= 21. : 
				winter = 1.
			if month == 3. and day <= 19. : 
				winter = 1.			

			dataArray[i,0] = temp
			dataArray[i,1] = year
			dataArray[i,2] = month
			dataArray[i,3] = day
			dataArray[i,4] = summer
			dataArray[i,5] = fall
			dataArray[i,6] = winter
			dataArray[i,7] = spring

			i += 1
	dataFile.close()
	return dataArray

def getPrediction(weights,listFeatures) : 

	predi = np.dot(weights.T,np.array(listFeatures)[:,None])
	return float(predi)

class experimentsETDL : 

	def __init__(self,startingT,endingT,nbPrevious,dataComplete) :

		self.startingT = startingT
		self.endingT = endingT
		self.nbPrevious = nbPrevious
		self.data = dataComplete

	def predictWithETD(self,interest,
		coeffAlpha,gamma,
		modLDA,K,
		nbFeatures,functionFeatures) : 
		
		t = self.startingT

		finalTable = np.zeros((self.endingT - self.startingT,self.nbPrevious))

		w_t = 1./nbFeatures *  np.ones((nbFeatures,1))

		# select randomly 5 sequences and observe behavior
		random_sequences = list(np.random.choice(range(self.startingT,self.endingT),size = 15))
		observations = [] # tuples (time step,real value to predict,predictions)
		
		means_episodes = []

		i = 0
		while t < self.endingT : #self.endingT : 

			alpha = 1/(i+1) * 0.0001
			value6 = float(self.data[t+self.nbPrevious,0])

			sequence = range(t,t+self.nbPrevious) # contains indexes of interest
			predictions_episode = []

			F = 1
			e = np.zeros((nbFeatures,1))

			j = 1
			for time_step_episode in sequence[:-1] :  
 				
				feat_t = functionFeatures(time_step_episode)
				P_t = getPrediction(w_t,feat_t)

				predictions_episode.append(P_t)

				F = gamma * F + interest
				
				l = K
				if modLDA == 'model 1' : 
					l =  K / (len(sequence) - j)

				e = gamma * l * e + l * interest * np.array(feat_t)[:,None] + (1-l) * F * np.array(feat_t)[:,None]

				feat_tp1 = functionFeatures(time_step_episode + 1)
				P_tp1 = getPrediction(w_t,feat_tp1)

				w_tp1 = w_t + alpha * (P_tp1 - P_t) * e

				feat_t = np.copy(feat_tp1)
				P_t = P_tp1
				w_t = np.copy(w_tp1)

				j += 1

			# last element, P_tp1 = outcome
			F = gamma * F + interest
			e = gamma * K * e + K * interest * np.array(feat_t)[:,None] + (1-K) * F * np.array(feat_t)[:,None]
			w_t += alpha * (value6 - P_t) * e
			predictions_episode.append(P_t)

			# at the end of an episode, compute the difference with the true value for each step
			diff = [math.fabs(elt - value6) for elt in predictions_episode]

			finalTable[t - self.startingT] = np.array(diff)

			means_episodes.append(np.mean(diff))

			if t in random_sequences : 
				observations.append((t,value6,predictions_episode))

			t +=  1 
			i += 1


		return finalTable,observations,means_episodes	

	def predictWithETD_randomI(self,coeffAlpha,gamma,K,nbFeatures,functionFeatures) : 
		
		t = self.startingT

		finalTable = np.zeros((self.endingT - self.startingT,self.nbPrevious))

		w_t = 1./nbFeatures *  np.ones((nbFeatures,1))

		# select randomly 5 sequences and observe behavior
		random_sequences = list(np.random.choice(range(self.startingT,self.endingT),size = 15))
		observations = [] # tuples (time step,real value to predict,predictions)
		
		means_episodes = []

		i = 0
		while t < self.endingT : #self.endingT : 

			alpha = 1/(i+1) * 0.0001
			value6 = float(self.data[t+self.nbPrevious,0])

			sequence = range(t,t+self.nbPrevious) # contains indexes of interest
			predictions_episode = []

			F = 1
			e = np.zeros((nbFeatures,1))

			interests_sequence = list(np.random.uniform(low = 0., high = 100.0, size = len(sequence)))

			j = 1
			for time_step_episode in sequence[:-1] :  
 				
				feat_t = functionFeatures(time_step_episode)
				P_t = getPrediction(w_t,feat_t)

				predictions_episode.append(P_t)

				F = gamma * F + interests_sequence[j-1]

				l =  K / (len(sequence) - j)

				e = gamma * l * e + l * interests_sequence[j-1] * np.array(feat_t)[:,None] + (1-l) * F * np.array(feat_t)[:,None]

				feat_tp1 = functionFeatures(time_step_episode + 1)
				P_tp1 = getPrediction(w_t,feat_tp1)

				w_tp1 = w_t + alpha * (P_tp1 - P_t) * e

				feat_t = np.copy(feat_tp1)
				P_t = P_tp1
				w_t = np.copy(w_tp1)

				j += 1

			# last element, P_tp1 = outcome
			F = gamma * F + interests_sequence[-1]
			e = gamma * K * e + K * interests_sequence[-1] * np.array(feat_t)[:,None] + (1-K) * F * np.array(feat_t)[:,None]
			w_t += alpha * (value6 - P_t) * e
			predictions_episode.append(P_t)

			# at the end of an episode, compute the difference with the true value for each step
			diff = [math.fabs(elt - value6) for elt in predictions_episode]

			finalTable[t - self.startingT] = np.array(diff)

			means_episodes.append(np.mean(diff))

			if t in random_sequences : 
				observations.append((t,value6,predictions_episode))

			t +=  1 
			i += 1


		return finalTable,observations,means_episodes

	def predictWithETD_interestBumps(self,
		fonction_interest,
		coeffAlpha,gamma,K,
		nbFeatures,functionFeatures) : 
		
		t = self.startingT

		finalTable = np.zeros((self.endingT - self.startingT,self.nbPrevious))

		w_t = 1./nbFeatures *  np.ones((nbFeatures,1))

		# select randomly 5 sequences and observe behavior
		random_sequences = list(np.random.choice(range(self.startingT,self.endingT),size = 15))
		observations = [] # tuples (time step,real value to predict,predictions)
		
		means_episodes = []

		i = 0
		
		while t < self.endingT : #self.endingT : 

			alpha = 1/(i+1) * 0.0001
			value6 = float(self.data[t+self.nbPrevious,0])

			sequence = range(t,t+self.nbPrevious) # contains indexes of interest
			predictions_episode = []

			interests_sequence = fonction_interest(sequence,self.data,100)

			F = 1
			e = np.zeros((nbFeatures,1))

			j = 1
			for time_step_episode in sequence[:-1] :  
 				
				feat_t = functionFeatures(time_step_episode)
				P_t = getPrediction(w_t,feat_t)

				predictions_episode.append(P_t)

				F = gamma * F + interests_sequence[j-1]
				
				l =  K / (len(sequence) - j)

				e = gamma * l * e + l * interests_sequence[j-1] * np.array(feat_t)[:,None] + (1-l) * F * np.array(feat_t)[:,None]

				feat_tp1 = functionFeatures(time_step_episode + 1)
				P_tp1 = getPrediction(w_t,feat_tp1)

				w_tp1 = w_t + alpha * (P_tp1 - P_t) * e

				feat_t = np.copy(feat_tp1)
				P_t = P_tp1
				w_t = np.copy(w_tp1)

				j += 1

			# last element, P_tp1 = outcome
			F = gamma * F + interests_sequence[-1]
			e = gamma * K * e + K * interests_sequence[-1] * np.array(feat_t)[:,None] + (1-K) * F * np.array(feat_t)[:,None]
			w_t += alpha * (value6 - P_t) * e
			predictions_episode.append(P_t)

			# at the end of an episode, compute the difference with the true value for each step
			diff = [math.fabs(elt - value6) for elt in predictions_episode]

			finalTable[t - self.startingT] = np.array(diff)

			means_episodes.append(np.mean(diff))

			if t in random_sequences : 
				observations.append((t,value6,predictions_episode))

			t +=  1 
			i += 1


		return finalTable,observations,means_episodes
