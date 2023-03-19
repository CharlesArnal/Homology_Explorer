
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time

from Discrete_Optimizer import Discrete_Optimizer


class Random_Search_Optimizer(Discrete_Optimizer):
	"""
    Randomly generates solutions
    """
    
	def get_parameters_from_strings(parameters):
		"""
		parameters is a list of strings
		batch_size
		    0      
		"""
		return [int(parameters[0])], "RS_"+parameters[0]


	def __init__(self, batch_size, local_path, saved_results_folder, exp_name, \
        optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
        saving_perf_period = 20, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):
		"""batch_size is the number of solutions randomly drawn in each iteration """
		self.batch_size = batch_size
		super().__init__(local_path, saved_results_folder, exp_name, \
        optimizer_name = optimizer_name, n_solutions_to_display = n_solutions_to_display, feedback_period= feedback_period, \
        saving_perf_period = saving_perf_period, n_current_solutions_saved = n_current_solutions_saved, \
		saving_solutions_period = saving_solutions_period, n_all_time_best_solutions_saved = n_all_time_best_solutions_saved, random_seed=random_seed)
	
	
	def generate_random_points(self):
		return np.random.randint(2, size = (self.batch_size,self.dim)).tolist()

	def evaluate_moves(self, possible_moves):
		scoring_starting_time = starting_CPU_and_wall_time()
		possible_rewards = self.obj_function(possible_moves)
		CPU_scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
		return possible_rewards, CPU_scoring_time

	def optimize(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		"""The main optimization function - optimizes with respect to obj_function

		stopping_condition is either None or a function that takes current solutions (a list of pairs (solution, score)) as input
		and outputs True if some stopping condition has been reached (and stops the optimization should it be the case)

		initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None
		"""
		super().setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)
		starting_time = starting_CPU_and_wall_time()
		iteration = 0
		stop = False
		while(stop == False):
			iteration_starting_time = starting_CPU_and_wall_time()
			# [batch_size, dim]
			new_points = self.generate_random_points()
			# list of length batch_size of lists of variable sizes of integers
			current_scores, scoring_time = self.evaluate_moves(new_points)
			solutions = [(new_points[i],current_scores[i]) for i in range(len(new_points))]
			stop = super().end_of_iteration_routine(iteration, solutions, iteration_time = CPU_and_wall_time(iteration_starting_time)[0],\
				 scoring_time = scoring_time, current_running_time = CPU_and_wall_time(starting_time)[0])
			iteration += 1
			if stop:
				super().end_of_run_routine(starting_time)
				return super().get_all_time_best_solution_and_score()


		
			
					


if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"


	batch_size = 6

	opti = Random_Search_Optimizer(batch_size, my_local_path, saved_files_folder, "test2",\
		optimizer_name = "Random optimizer", n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=5,\
			feedback_period=1, saving_perf_period= 3, saving_solutions_period=6 )

	n_iter = 6
	dim = 4
	obj_function = lambda my_liste : [sum(x) for x in my_liste]
	initial_solutions = None#[[2,3,4,5],[1,2,3,4],[0,1,0,1]]
	opti.optimize(n_iter, dim, obj_function, initial_solutions = initial_solutions, stopping_condition = None, max_running_time = 10, clear_log = False)



	
	# BATCH_SIZE, N_ITER, FEEDBACK_FREQUENCY, N_SOLUTIONS_SAVED, SAVED_SOLUTIONS_FILE = sys.argv[1:6]

	# DEGREE, DIMENSION, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME,LOCAL_PATH, TEMP_FILES_FOLDER, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
	# TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, STARTING_SIGNS_DISTRIBUTIONS_FILE,\
	# TEMP_HOMOLOGIES_FILE, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE , SAVE_PERF_FILE, SAVE_PERIOD, OUTPUT_FILE = sys.argv[-18:]


	# print("\nUsing Random Sampling to optimize signs distribution.")
	

	# #DEGREE = int(DEGREE)
	# #DIMENSION = int(DIMENSION)
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

	# # I had to code it myself to make sure all evaluations are called at once
	# def score_RS(signs):
	# 	# signs must be a numpy array [sub_batch_size, n_signs]
	# 	# or the name of a file containing all the signs
	# 	return calc_score(LOCAL_PATH, TEMP_FILES_FOLDER, POLYMAKE_SCORING_SCRIPT, signs,\
	# 			TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, OUTPUT_SCORING_FILE,\
	# 			DEGREE, DIMENSION, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE, TEMP_HOMOLOGIES_FILE).tolist()


	# #random.seed(int(SEED))
	# #np.random.seed(int(SEED))

	# objective_function = score_RS
	# batch_size = int(BATCH_SIZE)
	# dim = N_SIGNS
	# feedback_frequency = int(FEEDBACK_FREQUENCY)
	# save_period = int(SAVE_PERIOD)
	# save_perf_file = os.path.join(LOCAL_PATH,SAVE_PERF_FILE)
	# stopping_obj_value = float(STOPPING_OBJ_VALUE)
	# max_running_time = float(MAX_RUNNING_TIME)
	# n_best_solutions_to_display = 12
	# n_solutions_saved = N_SOLUTIONS_SAVED
	# saved_solutions_file = os.path.join(LOCAL_PATH,SAVED_SOLUTIONS_FILE)
	# n_iter = N_ITER
	# output_file = os.path.join(LOCAL_PATH,OUTPUT_FILE)
	# starting_signs_distributions = starting_signs_distributions
	
	# starting_time = None

	# discrete_Random_Solver(objective_function, dim, n_iter, batch_size, starting_signs_distributions, feedback_frequency,\
	# n_best_solutions_to_display, starting_time, save_period, save_perf_file, stopping_obj_value =STOPPING_OBJ_VALUE, stopping_time = max_running_time,\
	# output_file = output_file, n_solutions_saved = n_solutions_saved, saved_solutions_file = saved_solutions_file)

