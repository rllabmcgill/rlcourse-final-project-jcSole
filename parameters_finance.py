# -*- coding: utf-8 -*-

import numpy as np
import math

def model1_lambda(position,constante) : 

	return (10-position + 1)*constante / 10.


def model1_interest(sequence,data,relativeValue) : 

	# sequence is index times of a sequence
	interests = [0.1]*len(sequence)
	"""
	Pour chaque element de la sequence, on regarde ceux d'avant. Si ya un bump
	entre les deux d'avant, on met pas d'intérêt sur celui là
	"""
	amplify = False
	for elt in sequence : 
		index_considered = sequence.index(elt)
		# recupérer les deux valeurs d'avant
		vm1 = data[elt - 1,0]
		vm2 = data[elt - 2,0]

		if math.fabs(vm1 - vm2) > 40. or amplify == True : 

			interests[index_considered] = relativeValue * interests[index_considered]
			amplify = True

	return interests
