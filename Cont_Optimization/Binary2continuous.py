

import random
import numpy as np
# Apparently unused
#from keras.utils import to_categorical

import matplotlib.pyplot as plt

from math import ceil

import time

# input is a list of 0 and 1s of length precision*dim 
# it encodes a point of dimension dim in [-amplitude,amplitude]^dim
def Binary2continuous(x,precision,dim,amplitude):
	binary_coordinates = [x[i*precision:(i+1)*precision] for i in range(dim)]
	continuous_coordinates = [b2c(binary_coordinate)*2*amplitude - amplitude for binary_coordinate in binary_coordinates]
	return continuous_coordinates

def b2c(x):

	return sum([element*2**(-index-1) for index, element in enumerate(x)])

#print(Binary2continuous([1,1,1,1,0,0,0,1],4,2,10))
