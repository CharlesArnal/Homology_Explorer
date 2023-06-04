import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time


from Discrete_Optimizer import Discrete_Optimizer


from ES_2 import pygad





class GA_Optimizer(Discrete_Optimizer):

	"""
	Optimizes with a Genetic Algorithm

	I had to modify some code to make sure that the objective function is called a single time per generation
	Some things are rather mysterious - sometimes the number of elements in the population seems to collapse ?
	at least with small numbers of signs

	pygad expects numpy arrays, but my own functions expect lists of lists

	Warning ! I had to modify the adaptive_mutation_by_space function in the pygad code so that it saves good results obtained after crossover but before mutation

	Code is quite different from the other Discrete_Optimizer(s) due to using an already implemented Genetic Algorithm
	"""

	def get_parameters_from_strings(parameters):
		"""
		parameters is a list of strings
		num_parents_mating   sol_per_pop    parent_selection_type
		  			0              1 				2
		"""
		return [int(parameters[0]), int(parameters[1]), parameters[2]], "GA_"+"_".join(parameters)

	def __init__(self, num_parents_mating, sol_per_pop, parent_selection_type, \
		local_path, saved_results_folder, exp_name, \
		optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
		saving_perf_period = 20, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):
		
		"""  """

		self.num_parents_mating = num_parents_mating
		self.sol_per_pop = sol_per_pop
		self.parent_selection_type = parent_selection_type

		super().__init__(local_path, saved_results_folder, exp_name, \
		optimizer_name = optimizer_name, n_solutions_to_display = n_solutions_to_display, feedback_period= feedback_period, \
		saving_perf_period = saving_perf_period, n_current_solutions_saved = n_current_solutions_saved, \
		saving_solutions_period = saving_solutions_period, n_all_time_best_solutions_saved = n_all_time_best_solutions_saved, random_seed=random_seed)



	def optimize(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		"""The main optimization function - optimizes with respect to obj_function

		stopping_condition is either None or a function that takes current solutions (a list of pairs (solution, score)) as input
		and outputs True if some stopping condition has been reached (and stops the optimization should it be the case)

		initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None
		"""
		super().setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)
		starting_time = starting_CPU_and_wall_time()

		# for user feedback (I need to be a bit clever to work around pygad's specificities)
		self.iteration_starting_time = starting_CPU_and_wall_time()
		self.scoring_time = 0


		# inputs will be a numpy array
		# index is not used, but is required by pygad
		def modified_obj_function(inputs, index):
			inputs = inputs.tolist()
			if len(inputs)==0 :
				self.scoring_time = 0
				return np.array([])
			else:
				# rmk : the obj_function is called twice per iteration
				self.scoring_starting_time = starting_CPU_and_wall_time()
				scores =  np.array(self.obj_function(inputs))
				self.scoring_time += CPU_and_wall_time(self.scoring_starting_time)[0]
				return scores



		def on_generation_routine(ga):
			""" Automatically called by pygad at the end of each generation (=iteration)"""
			population = ga.population.tolist()
			scores = ga.last_generation_fitness.tolist()
			solutions = [(population[i],scores[i]) for i in range(len(population))]
			iteration = ga.generations_completed - 1
			stop = super(GA_Optimizer, self).end_of_iteration_routine(iteration, solutions, iteration_time = CPU_and_wall_time(self.iteration_starting_time)[0],\
				 scoring_time = self.scoring_time, current_running_time = CPU_and_wall_time(starting_time)[0])
			self.iteration_starting_time = starting_CPU_and_wall_time()
			self.scoring_time = 0
			if stop == True:
				return "stop"

		def custom_function_inside_adaptive_mutation(population, scores):
			""""
				Adaptive mutation takes the products of the crossovers, evaluates them then mutates them
				The products of the crossovers are not kept - the mutations are
				To be sure not to miss any good solutions, we store the best of them before the mutations (but after the crossover)
			"""
			population = population.tolist()
			scores = scores.tolist()
			solutions = [(population[i],scores[i]) for i in range(len(population))]
			solutions.sort(key = lambda x : x[1], reverse = True)
			super(GA_Optimizer, self).update_all_time_best_solutions(solutions, already_sorted= True)
			super(GA_Optimizer, self).save_performance(solutions, current_running_time = CPU_and_wall_time(starting_time)[0])

		
		def on_stop_routine(ga, list_of_fitness_values):
			""" Automatically called by pygad at the end of the whole process"""
			super(GA_Optimizer, self).end_of_run_routine(starting_time)
			
		# TODO for now, we don't use the initial_solutions (as pygad takes them as the full starting population, instead of adding them to it)

		# Read documentation 
		ga_instance = pygad.GA(num_generations = n_iter,
                       num_parents_mating = self.num_parents_mating,
					   initial_population = None, # TODO : initial_solutions,
					   sol_per_pop = self.sol_per_pop,
                       gene_space = [0,1],
                       gene_type = int,
					   num_genes = self.dim,
                       fitness_func = modified_obj_function,
					   keep_parents=-1,
					   parent_selection_type= self.parent_selection_type, #"sss", "tournament"
					   crossover_type="two_points",
					   mutation_type="adaptive",
					   mutation_percent_genes=[15,5], # percentage of entries of the low and high fitness solutions to be mutated
					   stop_criteria=None, # taken care of by on_generation_routine
                       save_best_solutions=False,
					   save_solutions=False,
                       on_generation=on_generation_routine,
		       		   custom_function_inside_adaptive_mutation = custom_function_inside_adaptive_mutation,
					   on_stop = on_stop_routine)

		ga_instance.run()

		return super().get_all_time_best_solution_and_score()

	


if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"

	num_parents_mating = 2
	sol_per_pop = 10
	parent_selection_type = "sss"

	opti = GA_Optimizer(num_parents_mating, sol_per_pop, parent_selection_type,\
		my_local_path, saved_files_folder, "test_ES",\
		optimizer_name = "Genetic Algorithm optimizer", n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=5,\
			feedback_period=1, saving_perf_period= 3, saving_solutions_period=6 )

	n_iter = 60
	dim = 6
	obj_function = lambda my_liste : [sum(x) for x in my_liste]
	initial_solutions = [[1, 1, 1, 1, 0, 1],[1, 0, 1, 1, 0, 1],[0, 1, 1, 1, 1, 1]]
	opti.optimize(n_iter, dim, obj_function, initial_solutions = initial_solutions, stopping_condition = None, max_running_time = 20, clear_log = True)





# def score_ES(signs, index):
# 	# signs must be a numpy array [sub_batch_size, n_signs]
# 	# index is useless but required by pygad
# 	if len(signs)==0 :
# 		return np.array([])
# 	else:
# 		return  calc_score(LOCAL_PATH, POLYMAKE_SCORING_SCRIPT, signs,\
# 				TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, OUTPUT_SCORING_FILE,\
# 				DEGREE, DIMENSION, FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE, TEMP_HOMOLOGIES_FILE)




# print("\n\nUsing evolution strategies to optimize signs distribution.\n")

# NUM_PARENTS_MATING, SOL_PER_POP, N_GENERATIONS, PARENT_SELECTION_TYPE, \
# 	DEGREE, DIMENSION, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME, LOCAL_PATH, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
# 	TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, STARTING_SIGNS_DISTRIBUTIONS_FILE,\
# 	TEMP_HOMOLOGIES_FILE,  FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE, SAVE_PERF_FILE, SAVE_PERIOD, OUTPUT_FILE  = sys.argv[1:]


# DEGREE = int(DEGREE)
# DIMENSION = int(DIMENSION)
# STOPPING_OBJ_VALUE = int(STOPPING_OBJ_VALUE)
# MAX_RUNNING_TIME = int(MAX_RUNNING_TIME)

# NUM_PARENTS_MATING = int(NUM_PARENTS_MATING)
# SOL_PER_POP = int(SOL_PER_POP)
# N_GENERATIONS =int(N_GENERATIONS)
# FIND_NEW_TOPOLOGIES = True if FIND_NEW_TOPOLOGIES == "True" else False
# SAVE_PERIOD = int(SAVE_PERIOD)


# # Get the number of signs
# with open(os.path.join(LOCAL_PATH, RELEVANT_POINTS_INDICES_INPUT_FILE), 'r') as f:
# 	N_SIGNS =  len(f.readline().split(","))
# 	print(f"\nNumber of signs to generate : {N_SIGNS}\n")


# fitness_function = score_ES 

# def on_gen(ga_instance):
# 	print("Generation : ", ga_instance.generations_completed)
# 	if ga_instance.pop_size != 0:
# 		print("Fitness of the best solution :", ga_instance.best_solution()[1])
# 		save_performance(ga_instance.best_solution()[1],time.time() - STARTING_TIME,SAVE_PERIOD,os.path.join(LOCAL_PATH,SAVE_PERF_FILE))
# 	if time.time() - STARTING_TIME > MAX_RUNNING_TIME:
# 		return "stop"
	

# # Read documentation : many user defined functions can be automatically called at various points ("on_generation", "on_mutation")
# ga_instance = pygad.GA(num_generations=N_GENERATIONS,
#                        num_parents_mating=NUM_PARENTS_MATING,
# 					   sol_per_pop=SOL_PER_POP,
#                        gene_space = [0,1],
#                        gene_type= int,
# 					   num_genes=N_SIGNS,
#                        fitness_func=fitness_function,
# 					   keep_parents=-1,
# 					   parent_selection_type= PARENT_SELECTION_TYPE, #"sss", "tournament"
# 					   crossover_type="two_points",
# 					   mutation_type="adaptive",
# 					   mutation_percent_genes=[15,5],#mutation_by_replacement=True,
# 					   stop_criteria="reach_"+str(STOPPING_OBJ_VALUE),
#                        save_best_solutions=True,
# 					   save_solutions=False,
#                        on_generation=on_gen)
                      
# STARTING_TIME = time.time()

# ga_instance.run()
# #ga_instance.plot_fitness()

# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# if ga_instance.best_solution_generation != -1:
#     print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

