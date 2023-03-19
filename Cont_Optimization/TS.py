

import random
import numpy as np
# Apparently unused
#from keras.utils import to_categorical

import matplotlib.pyplot as plt

from math import ceil
import math

import time

import os
import sys
import copy

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

def generate_moves(population, tabu_moves, current_step, max_amplitude, constraints_function, more_exploration = False, percent_more_exploration=None):
	# population of size [size_pop, dim]
	# returns a list of length size_pop of lists of variable sizes (due to the tabu moves) of 0/1 arrays of length dim	
	possible_moves = []
	if len(population) != 0 :
		dim = len(population[0])
	for element in population:
		# could maybe do it in a simpler way, but I had weird problems without this
		local_element = copy.deepcopy(element)
		local_moves = []
		for i in range(dim):
			very_local_element = copy.deepcopy(local_element)
			very_local_element[i] =  very_local_element[i] + current_step
			if very_local_element not in tabu_moves and constraints_function(very_local_element)== 1:
				local_moves.append(copy.deepcopy(very_local_element))
			very_local_element[i] = very_local_element[i] - 2*current_step
			if very_local_element not in tabu_moves and constraints_function(very_local_element)== 1:
				local_moves.append(copy.deepcopy(very_local_element))
		# To add more exploration, we construct sum(percent_more_exploration)*dim additional moves by randomly selecting entries and modifying them
		# there will be percent_more_exploration[0]*dim moves with 2 modifications, percent_more_exploration[1]*dim moves with 3 modifications, etc.
		if more_exploration:
			for n_of_modifs, percent_w_modifs in enumerate(percent_more_exploration):
				n_of_modifs +=2
				n_w_modifs = ceil(percent_w_modifs*dim)
				for i in range(n_w_modifs):
					very_local_element = copy.deepcopy(local_element)
					indices = random.sample(range(dim),min(n_of_modifs,dim))
					for i in indices :
						very_local_element[i] = very_local_element[i] + np.random.randint(0,2)*current_step
					if very_local_element not in tabu_moves and constraints_function(very_local_element)== 1:
						local_moves.append(very_local_element)

		# so that we don't get stuck
		if local_moves == []:
			local_moves = [(np.random.rand( dim) *max_amplitude*2 - np.ones((dim))*max_amplitude).tolist()]
		possible_moves.append(local_moves)
	return possible_moves


def evaluate_moves(possible_moves,objective_function):
	scoring_starting_time = time.time()
	possible_rewards = [[objective_function(move) for move in moves_list] for moves_list in possible_moves ]
	scoring_time = time.time()- scoring_starting_time
	return possible_rewards, scoring_time

# evaluate_moves([[[0,0],[1,0],[0,1]],[[1,1],[3,1]],[[1,1]]])) returns [[0, 1, 1], [2, 4], [2]]

def select_moves(possible_rewards, possible_moves):
	population = []
	current_values = []
	for index_element, set_of_rewards in enumerate(possible_rewards):
		index_best_move = np.argmax(set_of_rewards)
		population.append(possible_moves[index_element][index_best_move])
		current_values.append(max(set_of_rewards))
	return population, current_values

def update_tabu_list(tabu_moves, population, STM_length):
	# rmk : using a global tabu list (rather than one for each element)
	for element in population:
		if element not in tabu_moves:
			if len(tabu_moves)>=STM_length:
				tabu_moves.pop(0)
			tabu_moves.append(element)


def mix_points(points):
	if len(points)==1:
		return points[0]
	points = np.array(points)
	new_point = np.mean(points, axis =0).tolist()
	return new_point

def manage_SI_and_SD(population,current_values,MTM_memory,SI_counter,SD_counter,\
									MTM_length, SI_thresh, SD_thresh,max_amplitude):
	# MTM memory a list of length size_pop of pairs [values, coordinates], where values and coordinates are lists of length at most MTM_length
	if len(population) != 0:
		dim = len(population[0])
	for index_element, _ in enumerate(population):
		if MTM_memory[index_element][0] ==[] or current_values[index_element]>max(MTM_memory[index_element][0]):
			MTM_memory[index_element][0].sort()
			if len(MTM_memory[index_element][0])>= MTM_length:
				MTM_memory[index_element][0].pop(0)
				MTM_memory[index_element][1].pop(0)
			MTM_memory[index_element][0].append(current_values[index_element])
			MTM_memory[index_element][1].append(population[index_element])
			SI_counter[index_element] = 0
			SD_counter[index_element] = 0
		else:
			SI_counter[index_element] += 1
			SD_counter[index_element] += 1
			if SD_counter[index_element] >=SD_thresh:
				SD_counter[index_element] =0
				SI_counter[index_element] =0
				population[index_element] = (np.random.rand( dim) *max_amplitude*2 - np.ones((dim))*max_amplitude).tolist()
			elif SI_counter[index_element] >=SI_thresh:
				SI_counter[index_element] =0
				population[index_element] = mix_points(MTM_memory[index_element][1])

def Tabu_search(objective_function, constraints_function, max_amplitude, initial_step, dim, n_iter, size_pop, STM_length,MTM_length,\
	SI_thresh, SD_thresh,  step_reduction_thresh, feedback_frequency,\
	n_best_solutions_to_display, starting_time, save_period, save_perf_file, stopping_obj_value =None, stopping_time = None,\
		more_exploration = False, percent_more_exploration=None, n_solutions_saved = 0, saved_solutions_file = None):
	# guideline : everything is a python list (of lists...) as opposed to a numpy array
	# [size_pop, dim]
	population = generate_random_points(size_pop,dim,max_amplitude)
	MTM_memory = [[[],[]]]*size_pop
	SI_counter = [0]*size_pop
	SD_counter = [0]*size_pop
	current_step = initial_step
	step_reduction_count = 0
	current_best_value =  -math.inf
	tabu_moves = []
	# clear the file
	if n_solutions_saved != 0:
		with open(saved_solutions_file, 'w') as f:
			pass
	for i in range(n_iter):
		iteration_starting_time = time.time()
		# list of length size_pop of lists of variable sizes (due to the tabu moves) of 0/1 arrays of length dim
		possible_moves = generate_moves(population, tabu_moves, current_step, max_amplitude, constraints_function,\
			more_exploration, percent_more_exploration)
		# list of length size_pop of lists of variable sizes of integers
		possible_rewards, scoring_time = evaluate_moves(possible_moves, objective_function)
		population, current_values = select_moves(possible_rewards, possible_moves)
		new_best_value = max(current_values)
		update_tabu_list(tabu_moves, population, STM_length)
		manage_SI_and_SD(population,current_values,MTM_memory,SI_counter,SD_counter,\
									MTM_length, SI_thresh, SD_thresh, max_amplitude)
		if new_best_value > current_best_value:
			step_reduction_count = 0
			current_best_value = new_best_value
		else:
			step_reduction_count+=1
		if step_reduction_count>step_reduction_thresh:
			step_reduction_count=0
			current_step = current_step/2
			print("\n Reducing step size to "+str(current_step)+"\n")
		iteration_time = time.time() - iteration_starting_time
		if i%feedback_frequency == 0:
			print(f"\nCurrent best scores at iteration {i}:")
			sorted_zipped_list = sorted(zip(current_values,population), key=lambda pair: pair[0], reverse= True)
			current_values = [x for x,_ in sorted_zipped_list]
			print(current_values[0:min(n_best_solutions_to_display,len(current_values))])
			print(f"Associated point : {sorted_zipped_list[0][1]}")
			print(f"Best value encountered : {current_best_value}")
			print(f"Duration of the iteration = {iteration_time}, duration of the scoring phase = {scoring_time}")
			print(f"Current step size = {current_step}")
			if n_solutions_saved != 0:
				with open(saved_solutions_file, 'a') as f:
					sorted_population = [y for _, y in sorted_zipped_list]
					np.savetxt(f,sorted_population[0:min(n_solutions_saved,len(current_values))],fmt='%d')
			#for j in range(min(n_best_solutions_to_display,len(current_values))):
			#	print(current_values[j])
		save_performance(current_best_value,time.time() - starting_time,save_period,save_perf_file)
		if stopping_obj_value != None and stopping_obj_value < max(current_values):
			print("Objective reached")
			return 0
		if stopping_time != None and time.time()-starting_time>stopping_time:
			print("Time limit reached")
			return 0
		

if __name__ == "__main__":
	
	SIZE_POP, N_ITER, STM_LENGTH, MTM_LENGTH, INITIAL_STEP, STEP_REDUCTION_THRESH, \
	SI_THRESH, SD_THRESH, FEEDBACK_FREQUENCY, N_BEST_SOLUTIONS_TO_DISPLAY, MORE_EXPLORATION = sys.argv[1:12]
	PERCENT_MORE_EXPLORATION = [float(percent) for percent in sys.argv[12:-11]]
	N_SOLUTIONS_SAVED, SAVED_SOLUTIONS_FILE = sys.argv[-11:-9]

	FUNCTION, DIMENSION, SEED, MAX_AMPLITUDE, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME,LOCAL_PATH,\
	SAVE_PERF_FILE, SAVE_PERIOD = sys.argv[-9:]
	




	random.seed(int(SEED))
	np.random.seed(int(SEED))

	print("\n\nTabu Search.\n")

	# Rappel : Tabu search maximises the target function

	print(f"\nOptimizing {FUNCTION} function in dimension {DIMENSION}\n")
	


	max_amplitude = float(MAX_AMPLITUDE)

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

	def constraints_function(x):
		if np.max(np.abs(x)) > max_amplitude:
			return -1
		else:
			return 1

	initial_step = float(INITIAL_STEP)*float(MAX_AMPLITUDE)
	dim = int(DIMENSION)
	n_iter = int(N_ITER)
	size_pop = int(SIZE_POP)
	STM_length = int(STM_LENGTH)
	MTM_length = int(MTM_LENGTH)
	SI_thresh = int(SI_THRESH)
	SD_thresh = int(SD_THRESH)
	feedback_frequency = int(FEEDBACK_FREQUENCY)
	step_reduction_thresh = int(STEP_REDUCTION_THRESH)
	n_best_solutions_to_display = int(N_BEST_SOLUTIONS_TO_DISPLAY)
	n_solutions_saved = int(N_SOLUTIONS_SAVED)
	save_period = int(SAVE_PERIOD)
	saved_solutions_file = os.path.join(LOCAL_PATH,SAVED_SOLUTIONS_FILE)
	save_perf_file = os.path.join(LOCAL_PATH,SAVE_PERF_FILE)
	more_exploration =  True if MORE_EXPLORATION == "True" else False

	stopping_obj_value = float(STOPPING_OBJ_VALUE)
	max_running_time = float(MAX_RUNNING_TIME)
	
	starting_time = time.time()

	Tabu_search(objective_function, constraints_function, max_amplitude, initial_step, dim, n_iter, size_pop, STM_length,MTM_length,\
	SI_thresh, SD_thresh,  step_reduction_thresh, feedback_frequency,\
	n_best_solutions_to_display, starting_time, save_period, save_perf_file, stopping_obj_value = stopping_obj_value, stopping_time = max_running_time,\
		more_exploration = more_exploration, percent_more_exploration = PERCENT_MORE_EXPLORATION, n_solutions_saved = n_solutions_saved, saved_solutions_file = saved_solutions_file)

	




