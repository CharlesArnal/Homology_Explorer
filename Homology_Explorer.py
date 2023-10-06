
import random
import numpy as np

from math import comb, floor

import os
import sys
import subprocess

from utilities import CPU_and_wall_time, starting_CPU_and_wall_time, read_known_homology_profiles

from homology_objective_functions import  create_objective_function_for_signs_optimization, triangulation_growing_objective_function

from signs_optimizers_for_triang_exploration import Signs_Optimizer_for_Triang_Exploration
from move_generators import generate_moves_nb_triangs, generate_moves_nb_triangs_nb_signs
from move_selectors import create_move_selector, Random_Triang_Selector, Greedy_Selector,\
	 Greedy_Expanding_Selector, Greedy_Randomized_Expanding_Selector, Greedy_Randomized_Selector


from Current_Point import Current_Point


# Attention : en plus des fichiers de current_point, il y a des fichiers intermédiaires (nb_triangs, etc.)

class Homology_Explorer():
	"""Explores the graph of triangulations and the signs distributions.

		The information is stored as Current_Point(s), that must be provided to the Homology_Explorer or created by it

		When the exploration takes place, the Current_Point(s) are modified (as they are merely paths to files that are modified), hence copies must be created
	"""
	def __init__(self, dim, degree,  local_path, temp_files_folder, saved_results_folder, exp_name, \
		explorer_name = "Homology explorer", feedback_period = 1, saving_perf_period = None, random_seed = None, verbose = True):
		self.degree = degree
		self.dim = dim
		self.local_path = local_path
		self.temp_files_folder = temp_files_folder
		self.saved_results_folder = saved_results_folder
		self.exp_name = exp_name
		self.explorer_name = explorer_name
		self.feedback_period= feedback_period
		self.saving_perf_period = saving_perf_period
		self.random_seed = random_seed
		self.verbose = verbose
		
		
		self.reset_random_seed(random_seed)

		self.current_point = None

		self.save_perf_file = os.path.join(self.local_path,self.saved_results_folder,"scores_wrt_time_"+exp_name+".txt")
		self.observed_homologies_file = os.path.join(self.local_path,self.saved_results_folder,"homologies_"+exp_name+".txt")
		self.visited_homologies_file = os.path.join(self.local_path,self.saved_results_folder,"visited_homologies_"+exp_name+".txt")


	def update_files_paths(self):
		self.save_perf_file = os.path.join(self.local_path,self.saved_results_folder,"scores_wrt_time_"+self.exp_name+".txt")
		self.observed_homologies_file = os.path.join(self.local_path,self.saved_results_folder,"homologies_"+self.exp_name+".txt")
		self.visited_homologies_file = os.path.join(self.local_path,self.saved_results_folder,"visited_homologies_"+self.exp_name+".txt")



	def reset_random_seed(self, random_seed):
		if random_seed != None:
			random.seed(random_seed)
			np.random.seed(random_seed)


	def copy_and_load_current_point(self, copied_current_point : Current_Point, destination_current_point_folder):
		"""Copies the content of copied_current_point into a new self.current_point at desination_current_point_folder"""
		self.current_point = Current_Point(self.degree, self.dim,  self.local_path, destination_current_point_folder)
		self.current_point.copy_from_other_current_point(copied_current_point)
		return self.current_point

	def setup(self, n_iter, stopping_condition = None, max_running_time = None):
		# Just in case the saved files paths have changed
		self.update_files_paths()
		self.n_iter = n_iter
		self.stopping_condition = stopping_condition
		self.max_running_time = max_running_time
		self.time_last_perf_save = 0
		# to clear the file (it generally doesn't make sense to concatenate the results of several runs)
		with open(self.save_perf_file, "w") as f:
			pass
		with open(self.visited_homologies_file, "w") as f:
			pass
		if self.current_point == None:
			print(f"Error : the explorer {self.explorer_name} hasn't been been provided with a starting point yet")
		self.reset_random_seed(self.random_seed)
		print(f"\n-----\nStarting exploration with explorer {self.explorer_name} in experiment {self.exp_name}\n")

	def end_of_iteration_routine(self, iteration, current_value, iteration_time, current_running_time, move_selection_feedback_info):
		self.save_performance(current_value, current_running_time)
		if iteration%self.feedback_period == 0:
			self.display_feedback(iteration, iteration_time, current_value, move_selection_feedback_info)
		stop = self.test_stop(iteration, current_running_time)
		return stop
	
	def display_feedback(self, iteration, iteration_time, current_value, move_selection_feedback_info):
		scoring_time, selected_triang, selected_signs, selected_flip, scores, selected_homology = move_selection_feedback_info["selection time"], move_selection_feedback_info["selected triang"], \
		move_selection_feedback_info["selected signs"], move_selection_feedback_info["selected flips"], move_selection_feedback_info["scores"], move_selection_feedback_info["selected homology"]
		print(f"End of iteration {iteration} for explorer {self.explorer_name}")
		print(f"Current score for the objective function : {current_value}")
		print(f"Duration of the iteration : {'{:0.6}'.format(iteration_time)}, duration of the scoring phase for the local search : {'{:0.6}'.format(scoring_time)}")
		if self.verbose:
			if scores != None :
				scores.sort(reverse = True)
				print(f"Best scores considered for the objective function :\n{scores[:10]}")
			if selected_homology != None:
				print(f"Selected homology :\n{selected_homology}")
			print(f"Selected flip :\n{selected_flip}")
			print(f"New triang :\n{selected_triang}")
			print(f"New signs :\n{selected_signs}")


		sys.stdout.flush()

	
	def save_performance(self, current_value, current_running_time):
		if self.saving_perf_period != None and current_running_time - self.time_last_perf_save >= self.saving_perf_period :
			with open(self.save_perf_file, 'a+') as f:
				f.write(f"{current_running_time} {current_value}\n")
				self.time_last_perf_save = current_running_time
	
	def test_stop(self, iteration, current_running_time):
		"""Tests whether the run must be stopped due to either having reached the max number of iteration, the max running time or some stopping condition
			returns True if the run must be stopped
		"""
		stop = False
		# if self.stopping_condition != None:
		# 	if self.stopping_condition(solutions):
		# 		stop = True
		# 		print(f"Stopping condition reached for {self.explorer_name}")
		if self.n_iter != None:
			if iteration == self.n_iter-1:
				stop = True
				print(f"Maximal number of iterations reached for {self.explorer_name}")
		if self.max_running_time != None:
			if current_running_time > self.max_running_time:
				stop = True
				print(f"Maximal running time reached for {self.explorer_name}")
		sys.stdout.flush()
		return stop
	
	def end_of_run_routine(self, starting_time):
		print("\n")
		print(f"End of the run for explorer {self.explorer_name}")
		print(f"CPU time passed: { '{:0.6}'.format(CPU_and_wall_time(starting_time)[0])}, wall time passed: {'{:0.6}'.format(CPU_and_wall_time(starting_time)[1])}")
		sys.stdout.flush()

	def update_current_point(self, selected_move, current_point : Current_Point):
		selected_triang_file_path, selected_relevant_points_indices_file_path, selected_flip_file_path, selected_signs_file_path, local_path, temp_files_folder = selected_move
		# check if the triangulation has changed (as opposed to only a sign)
		triang_change = True
		selected_flip =""
		with open(selected_flip_file_path,"r") as f:
			# from [10,3:[[{7},{0,3,6}]->0]] to [{7},{0,3,6}]
			selected_flip = f.readline().split(":")[1].replace("\n","")[1:-5]
			if selected_flip == "[[]->0]]":
				triang_change = False
		# if the triangulation has changed, we update the flips
		if triang_change:
			# Necessary (for now), can't open the same file for reading and writing in the C++ file
			new_flips_file_path= os.path.join(local_path, temp_files_folder,"new_flips.dat")
			# Computes the new flips and replaces the old ones with the new ones
			list_files = subprocess.run(["./update_flips", current_point.chiro_file, current_point.triang_file, current_point.symmetries_file,\
				selected_triang_file_path, current_point.flips_file, selected_flip_file_path, new_flips_file_path])
			# Replaces the old flips with the new ones
			with open(current_point.flips_file,"w") as f:
				with open(new_flips_file_path,"r") as g:
					for line in g:
						f.write(line)
		# Replaces the old triangulation with the new selected one
		with open(current_point.triang_file,"w") as f:
			with open(selected_triang_file_path,"r") as g:
				for line in g:
					f.write(line)
		# Replaces the old signs with the new ones
		with open(current_point.signs_file,"w") as f:
			with open(selected_signs_file_path,"r") as g:
				signs = np.loadtxt(g,dtype = int)
				np.savetxt(f,np.array([signs]),fmt='%d')
		# Replaces the old relevant indices with the new ones
		with open(current_point.current_points_indices_file,"w") as f:
			with open(selected_relevant_points_indices_file_path,"r") as g:
				for line in g:
					f.write(line)
		return current_point


	def explore(self, n_iter, move_generator, move_selector, signs_optimizer = None, stopping_condition = None, max_running_time = None):
		"""Alternates between moving between triangulations and signs with the move_generator and the move_selector and applying the signs_optimizer

			The appropriate objective_functions are already "contained" in move_selector and signs_optimizer

			signs_optimizer is an instance of Signs_Optimizer_for_Triang_Exploration (or None) that also contains its parameters, including allowed running time
			
			self.current_point must have been initialized before

			Mostly for internal use (in more "user-friendly" fonctions like walking_search_on_triang_graph)
		"""
		self.setup(n_iter, stopping_condition, max_running_time)
		starting_time = starting_CPU_and_wall_time()

		if signs_optimizer != None:
			print("Initial signs optimization")
			self.current_point = signs_optimizer.optimize(self.current_point)
		iteration = 0
		stop = False
		while(stop == False):
			print("-------------")
			print(f"Starting iteration {iteration} for explorer {self.explorer_name}:\n")
			sys.stdout.flush()
			iteration_starting_time = starting_CPU_and_wall_time()
			# possible_moves is a list of file names containing all the necessary information
			# nb_triangs, nb_flips, nb_signs, local_path
			possible_moves = move_generator(self.current_point, self.temp_files_folder)
			# selected_move contains selected_triang, selected_flip, selected_sign, local_path
			# Note : self.visited_homologies_file is only updated here (not when optimizing the signs)
			selected_move, current_value, move_selection_feedback_info = move_selector(possible_moves, self.current_point.all_points_file, self.degree, self.dim)
			self.current_point = self.update_current_point(selected_move, self.current_point)
			sys.stdout.flush()
			if signs_optimizer != None:
				print("\nOptimizing signs separately...")
				self.current_point = signs_optimizer.optimize(self.current_point)
				sys.stdout.flush()
			stop = self.end_of_iteration_routine(iteration, current_value, iteration_time = CPU_and_wall_time(iteration_starting_time)[0],\
				 current_running_time = CPU_and_wall_time(starting_time)[0], move_selection_feedback_info = move_selection_feedback_info)
			iteration += 1
			if stop:
				self.end_of_run_routine(starting_time)
				return self.current_point
			

	def initialize_with_new_triangulation(self, final_size_or_num_iter, current_point_folder, look_while_growing = True) -> Current_Point:
		""" Creates and stores a Current_Point representing a new triangulation

			final_size_or_num_iter can be either "Trivial", "Medium", "Large", or a number of iterations
		"""
		self.current_point  = Current_Point(self.dim, self.degree, self.local_path, current_point_folder)
		self.current_point.standard_initialization(verbose = True)
		# Grows the triangulation if final_size_or_num_iter is not "Trivial" or 0
		if final_size_or_num_iter != 0 and final_size_or_num_iter != "Trivial":
			obj_function = triangulation_growing_objective_function
			move_generator = generate_moves_nb_triangs_nb_signs
			move_selector = create_move_selector(Greedy_Randomized_Selector, obj_function, must_compute_homology = look_while_growing, \
			 objective_function_takes_homology_as_input = False, observed_homologies_file = self.observed_homologies_file, visited_homologies_file = self.visited_homologies_file)

			if isinstance(final_size_or_num_iter, int):
				n_iter = final_size_or_num_iter
			# max number of vertices in triangulation is C(degree+dim,dim)
			elif final_size_or_num_iter == "Medium":
				n_iter = floor(float(comb(self.dim+self.degree, self.degree))*0.55)
			elif final_size_or_num_iter == "Large":
				n_iter = floor(float(comb(self.dim+self.degree, self.degree)))

			self.explore(n_iter, move_generator, move_selector)

		return self.current_point
	
	def initialize_with_Harnack_curve(self, current_point_folder) -> Current_Point:
		self.current_point  = Current_Point(self.dim, self.degree, self.local_path, current_point_folder)
		self.current_point.Harnack_curve_initialization(verbose = True)
		return self.current_point

		
	def initialize_with_random_triangulation_with_random_convex_hull(self, current_point_folder):
		self.current_point  = Current_Point(self.dim, self.degree, self.local_path, current_point_folder)
		self.current_point.random_convex_hull_initialization(verbose = True)
		return self.current_point


	def generate_random_triangulation_with_random_walk(self, initial_size, n_random_steps, current_point_folder, look_while_growing = False) -> Current_Point:
		"""Grows and saves as a Current_Point a new triangulation of size Trivial, Medium or Large, then takes n_random_steps random steps"""
		self.current_point  = self.initialize_with_new_triangulation(initial_size, current_point_folder, look_while_growing) 
		# Random walk
		if look_while_growing == False :
			move_generator = generate_moves_nb_triangs
		else :
			move_generator = generate_moves_nb_triangs_nb_signs

		move_selector = create_move_selector(Random_Triang_Selector, None, must_compute_homology = look_while_growing, \
			 objective_function_takes_homology_as_input = False, observed_homologies_file = self.observed_homologies_file, visited_homologies_file = self.visited_homologies_file)

		self.explore(n_random_steps, move_generator, move_selector)
		return self.current_point
		
	# def optimize_signs(self,  n_iter, polymake_scoring_script, optimizer_type, optimizer_max_running_time, optimizers_parameters):
	# 	""" Optimizes the signs relative to the polymake_scoring_script and the triangulation in self.current_point
		
	# 	self.current_point must have been initialized with at least triang_file, all_points_file and current_points_indices_file (the other files are not needed)"""
	# 	# TODO do something with the initial signs : add option to consider them, but also not to use them
		pass
		# obj_function_for_signs_optimizer = create_objective_function_for_signs_optimization(self.list_of_homologies_file, self.temp_files_folder, polymake_scoring_script)
		# 	signs_optimizer = Signs_Optimizer_for_Triang_Exploration(optimizer_type, obj_function_for_signs_optimizer,  self.local_path, self.temp_files_folder, optimizer_max_running_time, optimizers_parameters)
	
		# self.current_point = signs_optimizer(self.current_point)

	def make_function_of_the_homology_profiles_value_novelty(self, function_of_the_homology_profiles, value_novelty = 0):
		"""
			function_of_the_homology_profiles takes as input a list of lists of integers
			Returns a new objective function that takes as input a list of lists of integers and
			whose values are the same as those of function_of_the_homology_profiles,
			except +10000 to the score of any homology profile not yet seen if value_novelty == 1
			and +10000 to the score of any homology profile not yet visited if value_novelty == 2
		"""
		if value_novelty == 0:
			return function_of_the_homology_profiles
		else:
			stored_homologies_file = None
			if value_novelty == 1:
				stored_homologies_file = self.observed_homologies_file
			elif value_novelty == 2:
				stored_homologies_file = self.visited_homologies_file
			else :
				print("Invalid value_novelty argument")

			def modified_function(homology_profiles):
				# homology profiles a list of lists of integers
				scores = function_of_the_homology_profiles(homology_profiles).tolist()
				# test if file already exists :
				if not os.path.isfile(stored_homologies_file):
					with open(stored_homologies_file, 'w') as f:
						# create the file and do nothing
						pass
				# Set of strings
				known_hom = set(read_known_homology_profiles(stored_homologies_file))
				for index, profile in enumerate(homology_profiles):
					profile = " ".join([str(x) for x in profile])
					if profile not in known_hom:
						scores[index] += 10000
				return np.array(scores)

			return modified_function
			
	def walking_search_on_triang_graph(self, n_iter, function_of_the_homology_profiles, max_running_time, optimizer_type = None, optimizer_max_running_time = None, \
									optimizers_parameters = None, also_look_at_neighbouring_signs = False, value_novelty = 0):
		"""A special case of explore, where the objective function (for both move_selector and the signs_optimizer) comes from the function_of_the_homology_profiles
			and move_generator simply considers the neighbouring triangulations and signs distributions

			self.current_point must have been previously initialized

			value_novelty can be 0 (default), 1 (+10000 to the score of any homology profile not yet seen) or 2 (+10000 to the score of any homology profile not yet visited)
		"""
		print(f"Starting a walking search on the graph of triangulations")
		sys.stdout.flush()
		if value_novelty != 0:
			print(f"Modifying the objective function to reward novelty w.r.t. {'observed' if value_novelty==1 else 'visited'} homology profiles")
			function_of_the_homology_profiles = self.make_function_of_the_homology_profiles_value_novelty(function_of_the_homology_profiles, value_novelty)
		# most signs_optimizer should make looking at the neighbours superfluous, but it's important to look at neighbours when using value_novelty_persistently
		if optimizer_type != None:
			if also_look_at_neighbouring_signs:
				move_generator = generate_moves_nb_triangs_nb_signs
			else:
				move_generator = generate_moves_nb_triangs
			obj_function_for_signs_optimizer = create_objective_function_for_signs_optimization(self.observed_homologies_file, self.temp_files_folder, function_of_the_homology_profiles)
			signs_optimizer = Signs_Optimizer_for_Triang_Exploration(optimizer_type, obj_function_for_signs_optimizer,  self.local_path, self.temp_files_folder, optimizer_max_running_time, optimizers_parameters, random_seed = self.random_seed)
		else:
			signs_optimizer = None
			move_generator = generate_moves_nb_triangs_nb_signs

		move_selector = create_move_selector(Greedy_Randomized_Selector, function_of_the_homology_profiles, must_compute_homology = True, \
			 objective_function_takes_homology_as_input = True, observed_homologies_file = self.observed_homologies_file, visited_homologies_file = self.visited_homologies_file)


		self.explore(n_iter, move_generator, move_selector, signs_optimizer = signs_optimizer, stopping_condition = None, max_running_time = max_running_time)





	