
import numpy as np
from math import floor

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time

from Discrete_Optimizer import Discrete_Optimizer



class MCTS_Optimizer(Discrete_Optimizer):
	"""
	Optimizes with a Monte Carlo Tree Search

	Everything is a python list (of lists...) as opposed to a numpy array
	"""

	def get_parameters_from_strings(parameters):
		"""
		parameters is a list of strings
		depth	n_MCR
		  0       1 
		"""
		return [int(parameters[0]), int(parameters[1])], "MCTS_"+"_".join(parameters)

	def __init__(self, depth, n_MCR, \
		local_path, saved_results_folder, exp_name, \
		optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
		saving_perf_period = 20, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):
		"""  """

		self.depth = depth
		self.n_MCR = n_MCR

		super().__init__(local_path, saved_results_folder, exp_name, \
		optimizer_name = optimizer_name, n_solutions_to_display = n_solutions_to_display, feedback_period= feedback_period, \
		saving_perf_period = saving_perf_period, n_current_solutions_saved = n_current_solutions_saved, \
		saving_solutions_period = saving_solutions_period, n_all_time_best_solutions_saved = n_all_time_best_solutions_saved, random_seed=random_seed)
	


	def generate_random_seqs(self, size_pop, dim):
		""" dim is not self.dim"""
		return np.random.randint(2, size = (size_pop, dim)).tolist()

	def binary_combinations(self, n):
		combs = [[]]
		for i in range(n):
			combs = [x+[0] for x in combs] + [x+[1] for x in combs]
		return combs

	def end_of_sub_iteration_routine(self, sub_iteration, sequence, solutions, current_running_time):
		""" Solutions is not expected to be already sorted
			A modified (and lighter) version of the usual end_of_iteration_routine
		"""
		solutions.sort(key = lambda x : x[1], reverse = True)
		super().update_all_time_best_solutions(solutions, already_sorted= True)
		super().save_performance(solutions, current_running_time)
		if sub_iteration% floor(self.dim/3) == 0:
			print(f"\nMCTS sub-iteration : {sub_iteration}, best score of sub-iteration : {solutions[0][1]}, number of solutions evaluated : {len(solutions)}")
			print(f"Current sequence : {sequence}")
		

	def NMCS(self, sequence, level):
		""" sequence is a list of length i, representing a partial solution
			level is the level of recursion

			In the output, solutions and scores are only used for performance monitoring purposes
		"""
		combs = self.binary_combinations(min(level, self.dim-len(sequence)))
		solutions = []
		if len(sequence)+level >= self.dim:
			solutions = [sequence + comb for comb in combs]
			scoring_starting_time = starting_CPU_and_wall_time()
			scores = self.obj_function(solutions)
			scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
			best_index = np.argmax(scores)
			return solutions[best_index], scores[best_index], solutions, scores, scoring_time
		else :
			for comb in combs:
				rd_ends = self.generate_random_seqs(self.n_MCR, self.dim-level-len(sequence))
				solutions = solutions + [sequence+comb +end for end in rd_ends]
			scoring_starting_time = starting_CPU_and_wall_time()
			scores = self.obj_function(solutions)
			scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
			best_index = np.argmax(scores)
			best_score = scores[best_index]
			best_solution = solutions[best_index]
			#print(f"Best sub-solution {best_solution}")
			return best_solution[:len(sequence)+1], best_score, solutions, scores, scoring_time


	def optimize(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		"""The main optimization function - optimizes with respect to obj_function

		stopping_condition is either None or a function that takes current solutions (a list of pairs (solution, score)) as input
		and outputs True if some stopping condition has been reached (and stops the optimization should it be the case)

		initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None

		Due to the nature of the MCTS algorithm, the function is organized a bit differently compared to optimize in other child classes of Discrete_Optimizer
		"""
		super().setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)
		starting_time = starting_CPU_and_wall_time()
		iteration = 0
		stop = False
		while(stop == False):
			iteration_starting_time = starting_CPU_and_wall_time()
			total_scoring_time = 0
			sequence = []
			total_scoring_time = 0
			while(len(sequence)<dim):
				# a recursive function
				sequence, score, solutions, scores, scoring_time = self.NMCS(sequence, self.depth)
				total_scoring_time+=scoring_time
				
				solutions = [(solutions[i],scores[i]) for i in range(len(solutions))]
				self.end_of_sub_iteration_routine(len(sequence), sequence, solutions, current_running_time = CPU_and_wall_time(starting_time)[0])
			
			stop = super().end_of_iteration_routine(iteration, solutions, iteration_time = CPU_and_wall_time(iteration_starting_time)[0],\
				 scoring_time = total_scoring_time, current_running_time = CPU_and_wall_time(starting_time)[0])
			iteration += 1
			if stop:
				super().end_of_run_routine(starting_time)
				return super().get_all_time_best_solution_and_score()





if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"

	depth = 5
	n_MCR = 20

	opti = MCTS_Optimizer(depth, n_MCR, my_local_path, saved_files_folder, "test_MCTS",\
		n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=2, feedback_period=1, saving_perf_period= 10, saving_solutions_period=20 )

	n_iter = 1
	dim = 15
	obj_function = lambda my_liste : [sum(x) for x in my_liste]
	initial_solutions = [([2,1,1,1,1]*3, 18), ([0,0,0,0,0]*3,0)]
	
	opti.optimize(n_iter, dim, obj_function, initial_solutions = initial_solutions, stopping_condition = None, max_running_time = 200, clear_log = True)









# # I had to code it myself to make sure all evaluations are called at once
# def score_MCTS(signs):
# 	# signs must be a numpy array [sub_batch_size, n_signs]
# 	# or the name of a file containing all the signs
# 	return calc_score(LOCAL_PATH, TEMP_FILES_FOLDER, POLYMAKE_SCORING_SCRIPT, signs,\
# 			TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, OUTPUT_SCORING_FILE,\
# 			DEGREE, DIMENSION, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE, TEMP_HOMOLOGIES_FILE).tolist()



# print("\nUsing Monte Carlo Tree Search to optimize signs distribution.")



# DEPTH, N_MCR, FEEDBACK_FREQUENCY, N_ITER, \
# N_SOLUTIONS_SAVED, SAVED_SOLUTIONS_FILE = sys.argv[1:7]

# DEGREE, DIMENSION, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME,LOCAL_PATH, TEMP_FILES_FOLDER, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
# TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, STARTING_SIGNS_DISTRIBUTIONS_FILE,\
# TEMP_HOMOLOGIES_FILE, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE , SAVE_PERF_FILE, SAVE_PERIOD, OUTPUT_FILE = sys.argv[-18:]


# DEPTH = int(DEPTH)
# N_MCR = int(N_MCR)
# DEGREE = int(DEGREE)
# DIMENSION = int(DIMENSION)
# N_ITER = int(N_ITER)
# STOPPING_OBJ_VALUE = int(STOPPING_OBJ_VALUE)
# MAX_RUNNING_TIME = int(MAX_RUNNING_TIME)
# FEEDBACK_FREQUENCY = int(FEEDBACK_FREQUENCY)
# N_SOLUTIONS_SAVED = int(N_SOLUTIONS_SAVED)

# FIND_NEW_TOPOLOGIES = True if FIND_NEW_TOPOLOGIES == "True" else False
# SAVE_PERIOD = int(SAVE_PERIOD)

# # Get starting sign distributions - the function returns the max over these distributions and the ones found by MCTS
# starting_signs_distributions = []
# if STARTING_SIGNS_DISTRIBUTIONS_FILE !="None":
# 	with open(os.path.join(LOCAL_PATH, STARTING_SIGNS_DISTRIBUTIONS_FILE), 'r') as f:
# 		starting_signs_distributions = np.loadtxt(f,dtype = int)
# 		if len(np.shape(starting_signs_distributions)) ==1 :
# 			starting_signs_distributions = [starting_signs_distributions.tolist()]
# 		else:
# 			starting_signs_distributions = starting_signs_distributions.tolist()


# # Get the number of signs
# with open(os.path.join(LOCAL_PATH, RELEVANT_POINTS_INDICES_INPUT_FILE), 'r') as f:
# 	N_SIGNS =  len(f.readline().split(","))
# 	print(f"\nNumber of signs to generate : {N_SIGNS}\n")




# # search diversification simply starting with a new random point (hard to meaningfully divide the search space)

# def generate_random_seqs(size_pop, dim):
# 	return np.random.randint(2, size = (size_pop,dim)).tolist()

# def binary_combinations(n):
# 	combs = [[]]
# 	for i in range(n):
# 		combs = [x+[0] for x in combs] + [x+[1] for x in combs]
# 	return combs

# def NMCS(sequence, n_MCR, dim, level):
# 	""" sequence is a list of length i, representing a partial solution
# 		level is the level of recursion
# 	"""
# 	combs = binary_combinations(min(level,dim-len(sequence)))
# 	solutions = []
# 	if len(sequence)+level >=dim:
# 		solutions = [sequence + comb for comb in combs]
# 		scoring_starting_time = starting_CPU_and_wall_time()
# 		scores = score_MCTS(np.array(solutions))
# 		scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
# 		best_index = np.argmax(scores)
# 		return solutions[best_index], scores[best_index], scoring_time
# 	else :
# 		for comb in combs:
# 			rd_ends = generate_random_seqs(n_MCR,dim-level-len(sequence))
# 			solutions = solutions + [sequence+comb +end for end in rd_ends]
# 		scoring_starting_time = starting_CPU_and_wall_time()
# 		scores = score_MCTS(np.array(solutions))
# 		scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
# 		best_index = np.argmax(scores)
# 		best_score = scores[best_index]
# 		best_solution = solutions[best_index]
# 		#print(f"Best sub-solution {best_solution}")
# 		return best_solution[:len(sequence)+1], best_score, scoring_time


# STARTING_TIME = None


# MCTS(N_SIGNS, DEPTH, N_MCR, N_ITER, FEEDBACK_FREQUENCY, starting_signs_distributions, STARTING_TIME,SAVE_PERIOD,os.path.join(LOCAL_PATH,SAVE_PERF_FILE),\
# 		stopping_obj_value=STOPPING_OBJ_VALUE,stopping_time=MAX_RUNNING_TIME, output_file=os.path.join(LOCAL_PATH,OUTPUT_FILE),\
# 				n_solutions_saved = N_SOLUTIONS_SAVED, saved_solutions_file= os.path.join(LOCAL_PATH,SAVED_SOLUTIONS_FILE))