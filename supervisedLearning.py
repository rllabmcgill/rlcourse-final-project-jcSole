# -*- coding: utf-8 -*-

import numpy as np
import csv
import math

def getPrediction(weights,listFeatures) : 

	predi = np.dot(weights.T,np.array(listFeatures)[:,None])
	return float(predi)


class experimentsSupervisedLearning : 

	def __init__(self,startingT,endingT,nbPrevious,dataComplete) :

		self.startingT = startingT
		self.endingT = endingT
		self.nbPrevious = nbPrevious
		self.data = dataComplete

	def supervisedlearning(self,
		coeffAlpha,gamma,
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

			j = 1
			sum_ = np.zeros((nbFeatures,1))
			for time_step_episode in sequence :  
 				
				feat_t = functionFeatures(time_step_episode)
				P_t = getPrediction(w_t,feat_t)

				predictions_episode.append(P_t)

				sum_ += alpha * (value6 - P_t) * np.array(feat_t)[:,None]

				j += 1

			# at the end of an episode, compute the difference with the true value for each step
			diff = [math.fabs(elt - value6) for elt in predictions_episode]

			finalTable[t - self.startingT] = np.array(diff)

			means_episodes.append(np.mean(diff))

			if t in random_sequences : 
				observations.append((t,value6,predictions_episode))

			t +=  1 
			i += 1


		return finalTable,observations,means_episodes	