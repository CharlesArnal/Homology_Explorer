import os
import numpy as np

from Current_Point import Current_Point

from Discrete_Optimizers.GA_Optimizer import GA_Optimizer
from Discrete_Optimizers.MCTS_Optimizer import MCTS_Optimizer
from Discrete_Optimizers.Random_Search_Optimizer import Random_Search_Optimizer
from Discrete_Optimizers.RL_Optimizer import RL_Optimizer
from Discrete_Optimizers.Tabu_Search_Optimizer import Tabu_Search_Optimizer


def standard_parameters_for_signs_optimizer_for_walking_search(optimizer_type):
	if optimizer_type == "Tabu_Search_Optimizer":
		size_pop = 20
		STM_length = 1000
		MTM_length = 4
		SI_thresh = 30
		SD_thresh = 60
		more_exploration = True
		percent_more_exploration = [0.1,0.1,0.1,0.1]
		return [size_pop, STM_length, MTM_length, SI_thresh, SD_thresh, more_exploration, percent_more_exploration]
	if optimizer_type == "RL_Optimizer":
		n_layers = 3
		width_layers = [128,64, 4]
		learning_rate = 0.0002
		n_sessions = 400
		percentile = 93
		super_percentile = 94
		min_randomness = False
		alpha = 0.5
		return [n_layers, width_layers, learning_rate, n_sessions, percentile, super_percentile, min_randomness, alpha]
	if optimizer_type == "Random_Search_Optimizer":
		batch_size = 400
		return [batch_size]
	if optimizer_type == "MCTS_Optimizer":
		depth = 5
		n_MCR = 20
		return [depth, n_MCR]
	if optimizer_type == "GA_Optimizer":
		num_parents_mating = 50
		sol_per_pop = 100
		parent_selection_type = "sss"
		return [num_parents_mating, sol_per_pop, parent_selection_type,]



class Signs_Optimizer_for_Triang_Exploration():
	def __init__(self, optimizer_type, obj_function_for_signs_optimizer,  local_path, temp_files_folder, max_running_time, optimizer_parameters = None, random_seed = None):
		""" Input:	- obj_function_for_signs_optimizer(current_point, solutions)
			    - a type of optimizer
				- information relative to the triangulation and the experiment
				- parameters for the optimizer
				"""
		self.optimizer_type = optimizer_type
		self.obj_function_for_signs_optimizer = obj_function_for_signs_optimizer
		self.optimizer_parameters = optimizer_parameters
		self.local_path = local_path
		self.temp_files_folder = temp_files_folder
		self.max_running_time = max_running_time
		self.random_seed = random_seed

		# We are not saving anything from this run
		saved_results_folder = self.temp_files_folder
		exp_name = "signs_opti_subtask"
		optimizer_name = self.optimizer_type
		n_solutions_to_display = 5
		feedback_period = 100
		saving_perf_period = None
		n_current_solutions_saved = 0
		saving_solutions_period = None
		n_all_time_best_solutions_saved = 1
		general_arguments = [local_path, saved_results_folder, exp_name, optimizer_name, n_solutions_to_display, feedback_period, saving_perf_period, \
			n_current_solutions_saved, saving_solutions_period, n_all_time_best_solutions_saved, random_seed]

		if optimizer_parameters == None:
			optimizer_parameters = standard_parameters_for_signs_optimizer_for_walking_search(optimizer_type)

		if optimizer_type == "Tabu_Search_Optimizer":
			self.opti = Tabu_Search_Optimizer(*optimizer_parameters, *general_arguments)
		elif optimizer_type == "RL_Optimizer":
			self.opti = RL_Optimizer(*optimizer_parameters, *general_arguments)
		elif optimizer_type == "Random_Search_Optimizer":
			self.opti = Random_Search_Optimizer(*optimizer_parameters, *general_arguments)
		elif optimizer_type == "MCTS_Optimizer":
			# changing the feedback period
			general_arguments[5] = 1
			self.opti = MCTS_Optimizer(*optimizer_parameters, *general_arguments)
		elif optimizer_type == "GA_Optimizer":
			self.opti = GA_Optimizer(*optimizer_parameters, *general_arguments)
		else:
			print("Incorrect optimizer type provided to Signs_Optimizer_for_Triang_Exploration")
		

	def optimize(self, current_point : Current_Point):
		""" Input : a Current_Point
			
			Output: - a current_point whose signs distribution has been optimized

				The intermediate results of the optimization run are not saved"""

		initial_signs_distributions = None
		# Get starting sign distributions to add to the starting population
		initial_signs_distributions = []
		if current_point.signs_file !="None":
			with open(current_point.signs_file, 'r') as f:
				initial_signs_distributions = np.loadtxt(f,dtype = int)
				if len(np.shape(initial_signs_distributions)) == 1 :
					initial_signs_distributions = [initial_signs_distributions.tolist()]
				else:
					initial_signs_distributions = initial_signs_distributions.tolist()

		# Get the number of signs
		with open(current_point.current_points_indices_file, 'r') as f:
			n_signs =  len(f.readline().split(","))
			print(f"\nNumber of signs to generate : {n_signs}\n")

		# create an obj_function that computes scores relative to the current_point
		def specialized_obj_function(solutions):
			return self.obj_function_for_signs_optimizer(current_point, solutions)

		optimized_signs = self.opti.optimize(n_iter = 100000000, dim = n_signs, obj_function = specialized_obj_function,\
			initial_solutions = initial_signs_distributions, stopping_condition = None, max_running_time = self.max_running_time, clear_log = True)

		with open(current_point.signs_file,"w") as g:
			np.savetxt(g,np.array([optimized_signs[0]]),fmt='%d')

		return current_point



