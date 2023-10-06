import numpy as np
import os
import copy
import sys
import random


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time



class Discrete_Optimizer:
	"""Parent class for various discrete optimizers
	Typically takes as input for its optimize() function a list of lists of length dim of 0s and 1s (each list corresponding to a solution)
	Optimizes them with respect to the black blox obj_function
	obj_fun takes as input a list of solutions and outputs a list of scores
	"""

	def __init__(self, local_path, saved_results_folder, exp_name, \
		optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
		saving_perf_period = None, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):

		# Feedback and performance log-related attributes
		self.local_path = local_path
		self.saved_results_folder = saved_results_folder
		self.exp_name = exp_name
		self.optimizer_name = optimizer_name
		self.n_solutions_to_display = n_solutions_to_display
		self.feedback_period = feedback_period
		self.n_current_solutions_saved = n_current_solutions_saved
		self.n_all_time_best_solutions_saved = n_all_time_best_solutions_saved
		self.saving_perf_period = saving_perf_period
		self.saving_solutions_period = saving_solutions_period
		self.random_seed = random_seed

		self.best_score_attained_file = os.path.join(self.local_path,self.saved_results_folder,exp_name+"_scores")
		self.variance_solutions_file = os.path.join(self.local_path,self.saved_results_folder,exp_name+"_variance_solutions")
		self.variance_scores_file = os.path.join(self.local_path,self.saved_results_folder,exp_name+"_variance_scores")
		self.saved_current_solutions_file = os.path.join(self.local_path,self.saved_results_folder,exp_name+"_current_best_solutions")
		self.all_time_best_solutions_file = os.path.join(self.local_path,self.saved_results_folder,exp_name+"_all_time_best_solutions")

		# a list of pairs (solution, value) storing the best solutions found up till now
		# the list is ALWAYS sorted by score in decreasing order
		self.all_time_best_solutions = []

		# Done in wall time and not CPU time; doesn't matter much
		# Time is counted since the start of the optimization
		self.time_last_perf_save = 0
		self.time_last_solutions_save = 0

	def clear_log_files(self):
		""" clears log files """
		with open(self.best_score_attained_file,"w") as f:
			pass
		with open(self.variance_solutions_file,"w") as f:
			pass
		with open(self.variance_scores_file,"w") as f:
			pass
		with open(self.saved_current_solutions_file,"w") as f:
			pass
		with open(self.all_time_best_solutions_file,"w") as f:
			pass

	def setup(self, n_iter, dim, obj_function, initial_solutions, stopping_condition = None, max_running_time = None, clear_log = True):
		"""Saves a few internal variables, gives some feedback to the user 
			initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None
		"""
		if self.random_seed != None:
			random.seed(self.random_seed)
			np.random.seed(self.random_seed)
						
		# handles the initial solutions
		if initial_solutions != None and initial_solutions != []:
			self.all_time_best_solutions = initial_solutions
			# test if list of solutions or list of pairs (solution, score), makes it so if it isn't the case
			if isinstance(initial_solutions[0], tuple) == False:
				scores = obj_function(initial_solutions)
				self.all_time_best_solutions = [(self.all_time_best_solutions[i], scores[i]) for i in range(len(self.all_time_best_solutions))]
			self.all_time_best_solutions.sort(key = lambda x : x[1], reverse = True)
			# keep only the correct number
			self.all_time_best_solutions = self.all_time_best_solutions[:self.n_all_time_best_solutions_saved]

		self.best_score_attained_file = os.path.join(self.local_path,self.saved_results_folder,self.exp_name+"_scores")
		self.variance_solutions_file = os.path.join(self.local_path,self.saved_results_folder,self.exp_name+"_variance_solutions")
		self.variance_scores_file = os.path.join(self.local_path,self.saved_results_folder,self.exp_name+"_variance_scores")
		self.saved_current_solutions_file = os.path.join(self.local_path,self.saved_results_folder,self.exp_name+"_current_best_solutions")
		self.all_time_best_solutions_file = os.path.join(self.local_path,self.saved_results_folder,self.exp_name+"_all_time_best_solutions")

		if clear_log:
			self.clear_log_files()
		self.n_iter = n_iter
		self.dim = dim
		self.obj_function = obj_function
		self.stopping_condition = stopping_condition
		self.max_running_time = max_running_time
		# initialize clocks
		self.time_last_perf_save = 0
		self.time_last_solutions_save = 0
		print(f"\n-----\nStarting optimization with optimizer {self.optimizer_name} in experiment {self.exp_name}\n")

	def end_of_iteration_routine(self, iteration, solutions, iteration_time, scoring_time, current_running_time):
		""" Solutions is not expected to be already sorted"""
		solutions.sort(key = lambda x : x[1], reverse = True)
		self.update_all_time_best_solutions(solutions, already_sorted= True)
		self.save_performance(solutions, current_running_time)
		if iteration%self.feedback_period == 0:
			print(f"\nIteration {iteration} for optimizer {self.optimizer_name}")
			self.display_current_best_solutions(solutions[:self.n_solutions_to_display])
			self.display_mean_scores(solutions)
			self.display_all_time_best_score()
			print(f"Duration of the {self.optimizer_name} iteration = {'{:0.3}'.format(iteration_time)}, duration of the scoring phase = {'{:0.3}'.format(scoring_time)}")
		sys.stdout.flush()
		stop = self.test_stop(iteration, solutions, current_running_time)
		return stop

	def end_of_run_routine(self, starting_time):
		print(f"End of the run for optimizer {self.optimizer_name}")
		print(f"CPU time passed: { '{:0.6}'.format(CPU_and_wall_time(starting_time)[0])}, wall time passed: {'{:0.6}'.format(CPU_and_wall_time(starting_time)[1])}")
		print(f"Best score found : {self.all_time_best_solutions[0][1]} \n")
		self.save_all_time_best_solutions()
		sys.stdout.flush()

	def test_stop(self, iteration, solutions, current_running_time):
		"""Tests whether the run must be stopped due to either having reached the max number of iteration, the max running time or some stopping condition
			returns True if the run must be stopped
		"""
		stop = False
		if self.stopping_condition != None:
			if self.stopping_condition(solutions):
				stop = True
				print(f"Stopping condition reached for {self.optimizer_name}")
		if self.n_iter != None:
			if iteration == self.n_iter-1:
				stop = True
				print(f"Maximal number of iterations reached for {self.optimizer_name}")
		if self.max_running_time != None:
			if current_running_time > self.max_running_time:
				stop = True
				print(f"Maximal running time reached for {self.optimizer_name}")
		return stop


	def get_all_time_best_solution_and_score(self):
		"""returns the best solution ever found and its score"""
		return self.all_time_best_solutions[0][0], self.all_time_best_solutions[0][1]

	
	def update_all_time_best_solutions(self, solutions, already_sorted = True):
		"""Updates self.all_time_best_solutions, returns True if at least one solution in solutions is better than the worst solution in self.best_solutions
		Saves any new all time best solution in all_time_best_solutions and displays a warning that a new solution has been found
		Assumes that solutions are already sorted if already_sorted == True
		Always assume that self.all_time_best_solutions is already sorted (in decreasing order)"""
		# Not very computationaly efficient, but negligible (and safer)
		if not already_sorted:
			solutions.sort(key = lambda x : x[1], reverse = True)
		for solution in solutions :
			if len(self.all_time_best_solutions)<self.n_all_time_best_solutions_saved and (solution not in self.all_time_best_solutions):
				print(f"New good score found : {solution[1]}")
				self.all_time_best_solutions.append(copy.deepcopy(solution))
				with open(self.all_time_best_solutions_file, 'a+') as f:
					f.write(f"{solution[1]} | {solution[0]}\n")
				self.all_time_best_solutions.sort(key = lambda x : x[1], reverse = True)      
			elif solution[1]>self.all_time_best_solutions[-1][1] and (solution not in self.all_time_best_solutions):
				print(f"New good score found : {solution[1]}")
				self.all_time_best_solutions.append(copy.deepcopy(solution))
				with open(self.all_time_best_solutions_file, 'a+') as f:
					f.write(f"{solution[1]} | {solution[0]}\n")
				self.all_time_best_solutions.sort(key = lambda x : x[1], reverse = True)
				self.all_time_best_solutions = self.all_time_best_solutions[:self.n_all_time_best_solutions_saved]        
		self.all_time_best_solutions.sort(key = lambda x : x[1], reverse = True)
	 
	def display_current_best_solutions(self, solutions_to_display):
		current_best_scores = np.array([sol[1] for sol in solutions_to_display])
		print(f"Current scores : {current_best_scores}")

	def display_mean_scores(self, solutions):
		"""Solutions should already be sorted"""
		print(f"Mean score : {np.mean([solution[1] for solution in solutions])}, mean score among 20 best scores : {np.mean([solution[1] for solution in solutions[:20]])}")

	def display_all_time_best_score(self, n = 1):
		print(f"The current all time best score is {self.all_time_best_solutions[0][1]}")

	def save_performance(self, solutions, current_running_time):
		"""Assumes that self.all_time_best_solutions is up to date
	Assumes that solutions (and all_time_best_solutions) have been sorted (in decreasing order)"""
		if self.saving_perf_period != None and current_running_time - self.time_last_perf_save >= self.saving_perf_period :
			self.save_all_time_highest_score(current_running_time)
			self.save_variance_current_solutions(solutions, current_running_time)
			self.save_variance_current_scores(solutions, current_running_time)
			self.time_last_perf_save = current_running_time
		if self.saving_solutions_period != None and current_running_time - self.time_last_solutions_save >= self.saving_solutions_period :
			self.save_current_best_solutions(solutions, current_running_time)
			self.time_last_solutions_save = current_running_time

	def save_all_time_highest_score(self, current_running_time):
		"""Saves the highest score obtained up till now """
		best_score = self.all_time_best_solutions[0][1]
		with open(self.best_score_attained_file, 'a+') as f:
			f.write(f"{current_running_time} {best_score}\n")
		return 0

	def save_variance_current_solutions(self, solutions, current_running_time):
		"""Saves the variance of the current solutions"""
		variance = np.sum(np.var(np.array([sol[0] for sol in solutions]), axis=0))
		with open(self.variance_solutions_file, 'a+') as f:
			f.write(f"{current_running_time} {variance}\n")
		return 0

	def save_variance_current_scores(self, solutions, current_running_time):
		"""Saves the variance of the current scores"""
		variance = np.var(np.array([sol[1] for sol in solutions]), axis=0)
		with open(self.variance_scores_file, 'a+') as f:
			f.write(f"{current_running_time} {variance}\n")
		return 0

	def save_current_best_solutions(self, solutions, current_running_time):
		"""Saves the n_current_solutions_saved best solutions in the current solutions"""
		if self.n_current_solutions_saved != 0 :
			with open(self.saved_current_solutions_file, 'a+') as f:
				f.write(f"\n{current_running_time} :\n")
				for i in range(self.n_current_solutions_saved):
					f.write(f"{solutions[i][1]} | {solutions[i][0]}\n")
	
	def save_all_time_best_solutions(self):
		"""Saves the all time best solutions (to be used after the last iteration)"""
		with open(self.all_time_best_solutions_file, 'a+') as f:
			f.write("\n\nFinal all time best solutions :\n")
			for solution in self.all_time_best_solutions:
				f.write(f"{solution[1]} | {solution[0]}\n")
			f.write("\nBest score ever found :\n")
			f.write(str(self.all_time_best_solutions[0][1]))


	def optimize_testing_only(self, n_iter, dim, obj_function):
		self.setup(n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True)
		starting_time = starting_CPU_and_wall_time()
		for i in range(n_iter):
			iteration_starting_time = starting_CPU_and_wall_time()
			solutions = [(np.random.uniform(0,1,dim).tolist(), float(np.random.uniform(10,20))) for j in range(6)]
			stop = self.end_of_iteration_routine(i, solutions, iteration_time = CPU_and_wall_time(iteration_starting_time)[0], scoring_time = 0, current_running_time = CPU_and_wall_time(starting_time)[0])
			if stop:
				self.end_of_run_routine(starting_time)



   

if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"

	opti = Discrete_Optimizer(my_local_path, saved_files_folder, "test1",\
		n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=2, feedback_period=2, saving_perf_period= 10, saving_solutions_period=20 )
	opti.optimize_testing_only(6, 4, (lambda x : x**2), )
