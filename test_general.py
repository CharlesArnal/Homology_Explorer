
import random
import numpy as np

from math import comb, floor
import matplotlib.pyplot as plt
import time
import os
import sys


from utilities import read_first_experiment_param_file

from Current_Point import Current_Point
from Homology_Explorer import Homology_Explorer
from homology_objective_functions import b_0, b_0_p_a_b1, b_total, compute_homology



def ad_hoc_mean(mylist):
	max_length = max([len(my_array) for my_array in mylist])
	max_x_coordinate = max([my_array[-1,0] for my_array in mylist])
	new_length = max(100, max_length*2)
	x_step = max_x_coordinate/float(new_length)
	new_list = []
	for my_array in mylist:
		new_array = []
		for i in range(new_length+1):
			x_coordinate = i*x_step
			indices_of_entries_greater = [index for index, x_coord in enumerate(my_array[:,0])  if x_coord > x_coordinate]
			if indices_of_entries_greater == []:
				y_coordinate = my_array[-1, 1]
			else :
				next_index = indices_of_entries_greater[0]
				if next_index == 0:
					y_coordinate = 0
				else:
					previous_index = next_index-1
					y_coordinate = my_array[previous_index,1]
			new_array.append([x_coordinate,y_coordinate])
		new_array = np.array(new_array)
		new_list.append(new_array)
	new_list = np.array(new_list)
	return np.mean(new_list,axis = 0)


array_1 = np.array([[1, 1], [3,3], [5,5]])
array_2 = np.array([[1.5, 2], [2.5, 3], [3.5,4], [4.5,5], [5.5, 6]])
results = [array_1, array_2]

results_bis = ad_hoc_mean(results)
print(results_bis)

plt.plot(results_bis[:,0],results_bis[:,1])


plt.show()
