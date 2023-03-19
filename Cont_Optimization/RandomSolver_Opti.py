

import random
import numpy as np
# Apparently unused
#from keras.utils import to_categorical

import matplotlib.pyplot as plt

from math import ceil

import time

import os
import sys
import copy
import math

from RanaFun import RanaFun
from RosenbrockFun import RosenbrockFun
from AckleyFun import AckleyFun
from RastriginFun import RastriginFun	
from SineEnvFun import SineEnvFun
from RanaFunConstraint import RanaFunConstraint
from save_performance import save_performance

# search diversification simply starting with a new random point (hard to meaningfully divide the search space)

def generate_random_points(size_pop, dim,max_amplitude):
	return (np.random.rand(size_pop,dim) *max_amplitude*2 - np.ones((size_pop,dim))*max_amplitude).tolist()


def evaluate_moves(possible_moves,objective_function):
	scoring_starting_time = time.time()
	possible_rewards = [objective_function(move) for move  in possible_moves ]
	scoring_time = time.time()- scoring_starting_time
	return possible_rewards, scoring_time


def Random_Solver(objective_function, constraints_function, max_amplitude, dim, n_iter, batch_size, feedback_frequency,\
	n_best_solutions_to_display, starting_time, save_period, save_perf_file, stopping_obj_value =None, stopping_time = None,\
	n_solutions_saved = 0, saved_solutions_file = None):
	# guideline : everything is a list of lists (rather than numpy arrays)
	
	current_best_value = -math.inf
	i = 0
	# clear the file
	if n_solutions_saved != 0:
		with open(saved_solutions_file, 'w') as f:
			pass
	while True:
		iteration_starting_time = time.time()
		# [batch_size, dim]
		new_points = generate_random_points(batch_size,dim,max_amplitude)
		# list of length batch_size of lists of variable sizes of integers
		current_scores, scoring_time = evaluate_moves(new_points, objective_function)
		new_best_value = max(max(current_scores), current_best_value)
		if new_best_value > current_best_value:
			print(f"\nNew best value found : {new_best_value} ")
			print(f"Associated point : {sorted(zip(current_scores,new_points), key=lambda pair: pair[0], reverse= True)[0][1]}")
			current_best_value = new_best_value
		iteration_time = time.time() - iteration_starting_time
		if i%feedback_frequency == 0:
			print(f"\nCurrent best scores at iteration {i}:")
			sorted_scores = sorted(current_scores)
			print(sorted_scores[0:min(n_best_solutions_to_display,len(sorted_scores))])
			print(f"Best value encountered : {current_best_value}")
			print(f"Duration of the iteration = {iteration_time}, duration of the scoring phase = {scoring_time}")
			if n_solutions_saved != 0:
				sorted_zipped_list = sorted(zip(current_scores,new_points), key=lambda pair: pair[0], reverse= True)
				with open(saved_solutions_file, 'a') as f:
					sorted_population = [y for _, y in sorted_zipped_list]
					np.savetxt(f,sorted_population[0:min(n_solutions_saved,len(current_scores))],fmt='%d')
		save_performance(current_best_value,time.time() - starting_time,save_period,save_perf_file)
		if stopping_obj_value != None and stopping_obj_value < current_best_value:
			print("Objective reached")
			return 0
		if stopping_time != None and time.time()-starting_time>stopping_time:
			print("Time limit reached")
			return 0
		i+=1
		

if __name__ == "__main__":
	
	BATCH_SIZE, FEEDBACK_FREQUENCY, N_SOLUTIONS_SAVED, SAVED_SOLUTIONS_FILE = sys.argv[1:5]

	FUNCTION, DIMENSION, SEED, MAX_AMPLITUDE, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME,LOCAL_PATH,\
	SAVE_PERF_FILE, SAVE_PERIOD = sys.argv[-9:]
	



	random.seed(int(SEED))
	np.random.seed(int(SEED))

	print("\n\nRandom Solver.\n")

	print(f"\nOptimizing {FUNCTION} function in dimension {DIMENSION}\n")

	if FUNCTION == "Rana":
		objective_function = RanaFun
	elif FUNCTION == "Rosenbrock":
		def objective_function(x):
			return -1*RosenbrockFun(x)
	elif FUNCTION == "Rastrigin":
		def objective_function(x):
			return -1*RastriginFun(x)
	elif FUNCTION == "Ackley":
		def objective_function(x):
			return -1*AckleyFun(x)
	elif FUNCTION == "SineEnv":
		objective_function = SineEnvFun


	constraints_function = None
	max_amplitude = float(MAX_AMPLITUDE)
	batch_size = int(BATCH_SIZE)
	dim = int(DIMENSION)
	feedback_frequency = int(FEEDBACK_FREQUENCY)
	save_period = int(SAVE_PERIOD)
	save_perf_file = os.path.join(LOCAL_PATH,SAVE_PERF_FILE)
	stopping_obj_value = float(STOPPING_OBJ_VALUE)
	max_running_time = float(MAX_RUNNING_TIME)
	n_best_solutions_to_display = 5
	n_solutions_saved = N_SOLUTIONS_SAVED
	saved_solutions_file = SAVED_SOLUTIONS_FILE

	# currently not used
	n_iter = 0
	
	starting_time = time.time()

	Random_Solver(objective_function, constraints_function, max_amplitude, dim, n_iter, batch_size, feedback_frequency,\
	n_best_solutions_to_display, starting_time, save_period, save_perf_file, stopping_obj_value =None, stopping_time = max_running_time,\
	n_solutions_saved = 0, saved_solutions_file = None)
