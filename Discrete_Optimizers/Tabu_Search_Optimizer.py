
import random

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time

from math import ceil
import copy

from Discrete_Optimizer import Discrete_Optimizer


class Tabu_Search_Optimizer(Discrete_Optimizer):
	"""
	Optimizes with Tabu search
	""" 

	def get_parameters_from_strings(parameters):
		"""
		parameters is a list of strings
		size_pop STM_length  MTM_length  SI_thresh SD_thresh more_exploration percent_more_exploration
		    0         1          2              3       4           5               6-...
		"""
		more_exploration = True if parameters[5] == "True" else False 
		optimizer_parameters = [int(parameters[i]) for i in range(5)] + [more_exploration] + ([[float(parameter) for parameter in parameters[6:]]] if more_exploration else [[]])
		optimizer_name = "TS_"+"_".join(parameters[0:6]) + ("_".join(parameters[6:]) if more_exploration else "")
		return optimizer_parameters, optimizer_name

	def __init__(self, size_pop, STM_length, MTM_length, SI_thresh, SD_thresh, more_exploration, percent_more_exploration,\
		local_path,  saved_results_folder, exp_name, \
		optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
		saving_perf_period = 20, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):
		""" """
		self.size_pop = size_pop
		self.STM_length = STM_length
		self.MTM_length = MTM_length
		self.SI_thresh = SI_thresh
		self.SD_thresh = SD_thresh
		self.more_exploration = more_exploration
		self.percent_more_exploration = percent_more_exploration

		self.current_scores = []
		self.population = []
		self.MTM_memory = [[[],[]]]*self.size_pop
		self.SI_counter = [0]*self.size_pop
		self.SD_counter = [0]*self.size_pop
		self.tabu_moves = []
		
		super().__init__(local_path, saved_results_folder, exp_name, \
		optimizer_name = optimizer_name, n_solutions_to_display = n_solutions_to_display, feedback_period= feedback_period, \
		saving_perf_period = saving_perf_period, n_current_solutions_saved = n_current_solutions_saved, \
		saving_solutions_period = saving_solutions_period, n_all_time_best_solutions_saved = n_all_time_best_solutions_saved, random_seed=random_seed)



	def generate_random_points(self):
		""" Search diversification simply starting with a new random point (hard to meaningfully divide the search space)"""
		return np.random.randint(2, size = (self.size_pop,self.dim)).tolist()

	def generate_moves(self):
		""" self.population of size [size_pop, dim]
		returns a list of length size_pop of lists of variable sizes (due to the tabu moves) of 0/1 arrays of length dim """
		possible_moves = []
		for element in self.population:
			# could maybe do it in a simpler way, but I had weird problems without this
			local_element = copy.deepcopy(element)
			local_moves = []
			for i in range(self.dim):
				local_element[i] = 1- local_element[i]
				if local_element not in self.tabu_moves:
					local_moves.append(copy.deepcopy(local_element))
				local_element[i] = 1- local_element[i]
				# To add more exploration, we construct sum(percent_more_exploration)*dim additional moves by randomly selecting entries and modifying them
				# there will be percent_more_exploration[0]*dim moves with 2 modifications, percent_more_exploration[1]*dim moves with 3 modifications, etc.
				if self.more_exploration:
					for n_of_modifs, percent_w_modifs in enumerate(self.percent_more_exploration):
						n_of_modifs +=2
						n_w_modifs = ceil(percent_w_modifs*self.dim)
						for i in range(n_w_modifs):
							indices = random.sample(range(self.dim),min(n_of_modifs, self.dim))
							for i in indices :
								local_element[i] = 1- local_element[i]
							if local_element not in self.tabu_moves:
								local_moves.append(copy.deepcopy(local_element))
							for i in indices :
								local_element[i] = 1- local_element[i]
			# so that we don't get stuck
			if local_moves == []:
				local_moves = [list(np.random.randint(2,size = self.dim))]
			possible_moves.append(local_moves)
		return possible_moves


	def evaluate_moves(self, possible_moves):
		possible_rewards = []
		# save useful info
		n_moves_per_element = [len(element) for element in possible_moves]
		# flatten
		flat_list_of_moves = sum(possible_moves,[])
		scoring_starting_time = starting_CPU_and_wall_time()
		flat_list_of_scores = self.obj_function(flat_list_of_moves)
		scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
		# deflatten
		count = 0
		for n in n_moves_per_element:
			possible_rewards.append(flat_list_of_scores[count:count+n])
			count += n
		return possible_rewards, scoring_time

# evaluate_moves([[[0,0],[1,0],[0,1]],[[1,1],[3,1]],[[1,1]]])) returns [[0, 1, 1], [2, 4], [2]]


	def select_moves(self, possible_rewards, possible_moves):
		population = []
		current_scores = []
		for index_element, set_of_rewards in enumerate(possible_rewards):
			index_best_move = np.argmax(set_of_rewards)
			population.append(possible_moves[index_element][index_best_move])
			current_scores.append(max(set_of_rewards))
		return population, current_scores

	def update_tabu_list(self, population):
		# rmk : using a global tabu list (rather than one for each element)
		for element in population:
			if element not in self.tabu_moves:
				if len(self.tabu_moves)>=self.STM_length:
					self.tabu_moves.pop(0)
				self.tabu_moves.append(copy.deepcopy(element))


	def mix_points(self, points):
		if len(points)==1:
			return points[0]
		new_point = []
		random.shuffle(points)
		separators = np.random.randint(self.dim,size = len(points)-1)
		separators.sort()
		separators = np.concatenate(([0],separators,[self.dim]))
		for index in range(len(separators)-1) :
			new_point = new_point+points[index][separators[index]:separators[index+1]]
		return new_point

	def manage_SI_and_SD(self):
		# MTM memory a list of length size_pop of pairs [values, coordinates], where values and coordinates are lists of length at most MTM_length
		for index_element, _ in enumerate(self.population):
			if self.MTM_memory[index_element][0] ==[] or self.current_scores[index_element]>max(self.MTM_memory[index_element][0]):
				self.MTM_memory[index_element][0].sort()
				if len(self.MTM_memory[index_element][0])>= self.MTM_length:
					self.MTM_memory[index_element][0].pop(0)
					self.MTM_memory[index_element][1].pop(0)
				self.MTM_memory[index_element][0].append(self.current_scores[index_element])
				self.MTM_memory[index_element][1].append(self.population[index_element])
				self.SI_counter[index_element] = 0
				self.SD_counter[index_element] = 0
			else:
				self.SI_counter[index_element] += 1
				self.SD_counter[index_element] += 1
				if self.SD_counter[index_element] >=self.SD_thresh:
					self.SD_counter[index_element] =0
					self.SI_counter[index_element] =0
					# clear the MTM memory si that we start anew (otherwise past successes cut current search short)
					self.MTM_memory[index_element] = [[],[]]
					self.population[index_element] = list(np.random.randint(2,size = self.dim))
				elif self.SI_counter[index_element] >=self.SI_thresh:
					self.SI_counter[index_element] =0
					self.population[index_element] = self.mix_points(self.MTM_memory[index_element][1])
	
	def setup(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		
		# normally done in super().setup, but needed for self.generate_random_points
		self.dim = dim

		self.population = self.generate_random_points()
		if initial_solutions != None:
			self.population += initial_solutions
			self.size_pop += len(initial_solutions)
		
		self.current_scores = []
		self.SI_counter = [0]*self.size_pop
		self.SD_counter = [0]*self.size_pop
		self.tabu_moves = []
		self.MTM_memory = [[[],[]]]*self.size_pop

		
		super().setup(n_iter, dim, obj_function, self.population, stopping_condition, max_running_time, clear_log)
	
	def optimize(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		""" The main optimization function - optimizes with respect to obj_function

		stopping_condition is either None or a function that takes current solutions (a list of pairs (solution, score)) as input
		and outputs True if some stopping condition has been reached (and stops the optimization should it be the case)

		initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None"""

		self.setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)
		starting_time = starting_CPU_and_wall_time()
		iteration = 0
		stop = False
		while(stop == False):
			iteration_starting_time = starting_CPU_and_wall_time()
			# list of length size_pop of lists of variable sizes (due to the tabu moves) of 0/1 arrays of length dim
			possible_moves = self.generate_moves()
			# list of length size_pop of lists of variable sizes of integers
			possible_rewards, scoring_time = self.evaluate_moves(possible_moves)
			self.population, self.current_scores = self.select_moves(possible_rewards, possible_moves)
			self.update_tabu_list(self.population)
			self.manage_SI_and_SD()
			
			solutions = [(self.population[i],self.current_scores[i]) for i in range(len(self.population))]
			stop = super().end_of_iteration_routine(iteration, solutions, iteration_time = CPU_and_wall_time(iteration_starting_time)[0],\
				 scoring_time = scoring_time, current_running_time = CPU_and_wall_time(starting_time)[0])
			iteration += 1
			if stop:
				super().end_of_run_routine(starting_time)
				return super().get_all_time_best_solution_and_score()
			



			


if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"

	size_pop = 2
	STM_length = 300
	MTM_length = 10
	SI_thresh = 40
	SD_thresh = 50
	more_exploration = False
	percent_more_exploration = []

 

	parameters = Tabu_Search_Optimizer.get_parameters_from_strings(["1", "2", "1", "1", "1", "True", "3", "4"])

	opti = Tabu_Search_Optimizer(size_pop, STM_length, MTM_length, SI_thresh, SD_thresh, more_exploration, percent_more_exploration,\
		my_local_path, saved_files_folder, "test_TS",\
		optimizer_name = "Tabu search optimizer", n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=5,\
			feedback_period=1, saving_perf_period= 3, saving_solutions_period=6 )

	n_iter = 6
	dim = 4
	obj_function = lambda my_liste : [sum(x) for x in my_liste]
	initial_solutions = [[2,3,4,5],[1,2,3,4],[0,1,0,1]]
	opti.optimize(n_iter, dim, obj_function, initial_solutions = initial_solutions, stopping_condition = None, max_running_time = 10, clear_log = True)











# # I had to code it myself to make sure all evaluations are called at once


# def score_TS(signs):
# 	# signs must be a numpy array [sub_batch_size, n_signs]
# 	# or the name of a file containing all the signs
# 	return calc_score(LOCAL_PATH, TEMP_FILES_FOLDER, POLYMAKE_SCORING_SCRIPT, signs,\
# 			TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, OUTPUT_SCORING_FILE,\
# 			DEGREE, DIMENSION, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE, TEMP_HOMOLOGIES_FILE).tolist()



# print("\nUsing Tabu Search to optimize signs distribution.\n")


# SIZE_POP, N_ITER, STM_LENGTH, MTM_LENGTH, \
# 	SI_THRESH, SD_THRESH, FEEDBACK_FREQUENCY, N_BEST_SOLUTIONS_TO_DISPLAY,MORE_EXPLORATION = sys.argv[1:10]
# PERCENT_MORE_EXPLORATION = [float(percent) for percent in sys.argv[10:-20]]
# N_SOLUTIONS_SAVED, SAVED_SOLUTIONS_FILE = sys.argv[-20:-18]

# DEGREE, DIMENSION, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME, LOCAL_PATH, TEMP_FILES_FOLDER, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
# TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, STARTING_SIGNS_DISTRIBUTIONS_FILE,\
# TEMP_HOMOLOGIES_FILE, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE , SAVE_PERF_FILE, SAVE_PERIOD, OUTPUT_FILE = sys.argv[-18:]


# DEGREE = int(DEGREE)
# DIMENSION = int(DIMENSION)
# STOPPING_OBJ_VALUE = int(STOPPING_OBJ_VALUE)
# MAX_RUNNING_TIME = int(MAX_RUNNING_TIME)
# SIZE_POP = int(SIZE_POP)
# N_ITER = int(N_ITER)
# STM_LENGTH = int(STM_LENGTH)
# MTM_LENGTH = int(MTM_LENGTH)
# SI_THRESH = int(SI_THRESH)
# SD_THRESH = int(SD_THRESH)
# FEEDBACK_FREQUENCY = int(FEEDBACK_FREQUENCY)
# N_BEST_SOLUTIONS_TO_DISPLAY = int(N_BEST_SOLUTIONS_TO_DISPLAY)
# MORE_EXPLORATION = True if MORE_EXPLORATION == "True" else False
# N_SOLUTIONS_SAVED = int(N_SOLUTIONS_SAVED)



# FIND_NEW_TOPOLOGIES = True if FIND_NEW_TOPOLOGIES == "True" else False
# SAVE_PERIOD = int(SAVE_PERIOD)


# # Get the number of signs
# with open(os.path.join(LOCAL_PATH, RELEVANT_POINTS_INDICES_INPUT_FILE), 'r') as f:
# 	N_SIGNS =  len(f.readline().split(","))
# 	print(f"\nNumber of signs to generate : {N_SIGNS}\n")

# # Get starting sign distributions to add to the starting population
# starting_signs_distributions = []
# if STARTING_SIGNS_DISTRIBUTIONS_FILE !="None":
# 	with open(os.path.join(LOCAL_PATH, STARTING_SIGNS_DISTRIBUTIONS_FILE), 'r') as f:
# 		starting_signs_distributions = np.loadtxt(f,dtype = int)
# 		if len(np.shape(starting_signs_distributions)) ==1 :
# 			starting_signs_distributions = [starting_signs_distributions.tolist()]
# 		else:
# 			starting_signs_distributions = starting_signs_distributions.tolist()

	



# """
# population =[[1],[2]]
# current_values= [10,10]
# self.MTM_memory= [[[20,15],[[3],[-2]]],[[0,5,11],[[3],[-2],[0]]]]
# SI_counter = [1,1]
# SD_counter = [20,2]
# MTM_length = 3
# SI_thresh=  10
# SD_thresh = 20
# manage_SI_and_SD(population,current_values,self.MTM_memory,SI_counter,SD_counter,\
# 									MTM_length, SI_thresh, SD_thresh)
# print(population)
# print(self.MTM_memory)
# print(SI_counter)
# print(SD_counter)
# """



# STARTING_TIME = None

# Tabu_search(N_SIGNS, N_ITER, SIZE_POP, STM_LENGTH,MTM_LENGTH, SI_THRESH, SD_THRESH,\
# 	  FEEDBACK_FREQUENCY, starting_signs_distributions, N_BEST_SOLUTIONS_TO_DISPLAY, STARTING_TIME,SAVE_PERIOD,os.path.join(LOCAL_PATH,SAVE_PERF_FILE),\
# 		stopping_obj_value=STOPPING_OBJ_VALUE,stopping_time=MAX_RUNNING_TIME, output_file=os.path.join(LOCAL_PATH,OUTPUT_FILE),\
# 			more_exploration=MORE_EXPLORATION,percent_more_exploration=PERCENT_MORE_EXPLORATION, \
# 				n_solutions_saved = N_SOLUTIONS_SAVED, saved_solutions_file= os.path.join(LOCAL_PATH,SAVED_SOLUTIONS_FILE))