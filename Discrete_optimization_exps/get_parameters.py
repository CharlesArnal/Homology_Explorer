import ast

import copy
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from Discrete_Optimizers.GA_Optimizer import GA_Optimizer
from Discrete_Optimizers.MCTS_Optimizer import MCTS_Optimizer
from Discrete_Optimizers.Random_Search_Optimizer import Random_Search_Optimizer
from Discrete_Optimizers.RL_Optimizer import RL_Optimizer
from Discrete_Optimizers.Tabu_Search_Optimizer import Tabu_Search_Optimizer

from homology_objective_functions import create_objective_function_for_signs_optimization, b_total, b_0, b_0_p_a_b1

from Objective_functions.Graph_objective_functions import obj_fun_2_1, obj_fun_2_3

from Current_Point import Current_Point



from utilities import waste_CPU_time


# the first line of params_file defines which experiments must be launched parallelly
# the second line contains various global parameters
# each other line gives the parameters for one experiment


# Structure of the parameters file :
"""
Exp_name
[[0,1,2], [3,4]]
local_path saved_results_folder
exp_num num_seeds dim obj_function max_running_time optimizer_type  specific_parameters
1 3 10 graph_2_1 3600 RS 30
"""
def test_function(vectors):
	waste_CPU_time(0.3)
	return [sum(vector) for vector in vectors]


def obj_function_from_string(name, local_path, exp_number = None, num_seeds = None, saved_results_folder = None, sub_exp_name = None):
	"""
	examples :
	graph_2.1_19 -> the graph function corresponding to conjecture 2.1 on 19 vertices

	homology_b0pab1_Harnack.10_2_10 -> the homology-based function b0a1 (b0 + alpha b1) on the triangulation "Harnack.10" of dim 2 and degree 10
	
	exp_number, num_seeds, saved_results_folder and sub_exp_name are needed for homology based functions (so that they can store their results)

	Returns a dictionary of objective functions (one for each seed in range(num_seeds), as homology based functions need the seed to define their saved results folder)
	"""
	if name == "test_fun":
		return test_function
	name_components = name.split("_")
	if name_components[0] == "graph":
		if name_components[1] == "2.1":
			N_vertices = int(name_components[2])
			def obj_fun(solutions):
				return [obj_fun_2_1(solution, N_vertices) for solution in solutions]
		elif name_components[1] == "2.3":
			N_vertices = int(name_components[2])
			def obj_fun(solutions):
				return [obj_fun_2_3(solution, N_vertices) for solution in solutions]
		else :
			print("Invalid objective function")
			return 1
		obj_fun_dict = dict.fromkeys(range(num_seeds),obj_fun)
		return obj_fun_dict
	elif name_components[0] == "homology":
		dim = int(name_components[3])
		degree = int(name_components[4])
		triangulation_name = name_components[2]
		current_point_folder = os.path.join(local_path, "Objective_functions", "Triangulations", triangulation_name, "current_point")
		current_point = Current_Point(dim, degree,  local_path, current_point_folder)
		if name_components[1] == "bt":
			function_of_the_homology_profiles = b_total
		elif name_components[1] == "b0":
			function_of_the_homology_profiles = b_0
		elif name_components[1] == "b0pab1":
			function_of_the_homology_profiles = b_0_p_a_b1
		else:
			print("Invalid objective function")
			return 1
		
		if not os.path.isdir(os.path.join(local_path, saved_results_folder)):
			os.mkdir(os.path.join(local_path, saved_results_folder))

		if not os.path.isdir(os.path.join(local_path, saved_results_folder, name)):
			os.mkdir(os.path.join(local_path, saved_results_folder, name))

		saved_homologies_folder = os.path.join(local_path, saved_results_folder, name, "homologies")
		if not os.path.exists(saved_homologies_folder):
			os.mkdir(saved_homologies_folder)
			
		temp_files_folder = os.path.join(local_path, saved_results_folder, name, "temp_files")
		if not os.path.exists(temp_files_folder):
			os.mkdir(temp_files_folder)

		# create a dir to store the temp files (one per exp_number and seed, so that there is no interference)
		obj_fun_dict = dict()
		# necessary to avoid the functions previously defined to be redefined by the loop (damn Python)
		def function_factory(saved_homologies_file, temp_files_subfolder, function_of_the_homology_profiles):
			obj_functions_for_signs_optimizer = create_objective_function_for_signs_optimization(saved_homologies_file, temp_files_subfolder, function_of_the_homology_profiles)
			def obj_fun(solutions):
				return  obj_functions_for_signs_optimizer(current_point, solutions)
			return obj_fun

		for seed in range(num_seeds):
			temp_files_subfolder = os.path.join(local_path, saved_results_folder, name, "temp_files", f"{sub_exp_name}_{name}_seed_{seed}_temp_files")
			if not os.path.exists(temp_files_subfolder):
				os.mkdir(temp_files_subfolder)
			saved_homologies_file = os.path.join(saved_homologies_folder, f"homologies_{sub_exp_name}_{name}_seed_{seed}")
			obj_fun_dict[seed] = function_factory(saved_homologies_file, temp_files_subfolder, function_of_the_homology_profiles)
		return obj_fun_dict
	else :
		print("Invalid objective function")
		return 0
	

	
def optimizer_from_name_and_parameters(optimizer_type, parameters, optimizer_general_arguments):
	"""
	optimizer_type is a string

	parameters is a list of strings

	general_arguments is a list of correctly formated arguments
	
	"""
	if optimizer_type == "TS":
		optimizer_parameters, optimizer_name = Tabu_Search_Optimizer.get_parameters_from_strings(parameters)
		opti = Tabu_Search_Optimizer(*optimizer_parameters, *optimizer_general_arguments)
	elif optimizer_type == "RL":
		optimizer_parameters, optimizer_name = RL_Optimizer.get_parameters_from_strings(parameters)
		opti = RL_Optimizer(*optimizer_parameters, *optimizer_general_arguments)
	elif optimizer_type == "RS":
		optimizer_parameters, optimizer_name = Random_Search_Optimizer.get_parameters_from_strings(parameters)
		opti = Random_Search_Optimizer(*optimizer_parameters, *optimizer_general_arguments)
	elif optimizer_type == "MCTS":
		# changing the feedback period
		optimizer_general_arguments[5] = 1
		optimizer_parameters, optimizer_name = MCTS_Optimizer.get_parameters_from_strings(parameters)
		opti = MCTS_Optimizer(*optimizer_parameters, *optimizer_general_arguments)
	elif optimizer_type == "GA":
		optimizer_parameters, optimizer_name = GA_Optimizer.get_parameters_from_strings(parameters)
		opti = GA_Optimizer(*optimizer_parameters, *optimizer_general_arguments)
	else :
		print("Invalid optimizer type")
	opti.optimizer_name = optimizer_name

	return opti


def get_parameters_exp_0(parameters_file):
	"""
	
	# parameters is a dictionary structured as follows :
	# {"exp_name" : "exp_1", "exp_batches":[["0","1","2"], ["3","4"]], ...other global parameters ...,  1 : dict_1, 2 : dict_2, ....}
	# dict_i contains the parameters for experiment i
	"""

	parameters = dict()
	with open(parameters_file, "r") as f:
		lines = f.readlines()
		# Oth line
		parameters["exp_name"] = lines[0].replace("\n","")
		# 1st line
		exp_batches = ast.literal_eval(lines[1])
		parameters["exp_batches"] = exp_batches
		# 2nd line
		text_global_params = lines[2].split()
		# NOTE simpler for now
		#parameters["local_path"] = text_global_params[0]
		if '/home/charles' in os.getcwd() :
			parameters["local_path"] = '/home/charles/Desktop/ML_RAG/Code/Discrete_optimization_exps'
		elif '/home/carnal/' in os.getcwd():
			parameters["local_path"] = '/user/carnal/home/ML_RAG/Code/Discrete_optimization_exps'


		parameters["saved_results_folder"] = text_global_params[1]
		parameters["n_solutions_to_display"] = 5
		parameters["feedback_period"] = 10
		parameters["saving_perf_period"] = 20	
		parameters["n_current_solutions_saved"] = 5
		parameters["saving_solutions_period"] = None
		parameters["n_all_times_best_solutions_saved"] = 5
	
		optimizer_general_parameters = [parameters["local_path"], parameters["saved_results_folder"], parameters["exp_name"], "", \
				  parameters["n_solutions_to_display"], parameters["feedback_period"], parameters["saving_perf_period"], parameters["n_current_solutions_saved"],\
					parameters["saving_solutions_period"], parameters["n_all_times_best_solutions_saved"]]
		
		# skip the 4th line, which is only for humans
		for line in lines[4:]:
			exp_parameters = dict()
			text_parameters = line.split()
			exp_number = int(text_parameters[0])
			exp_parameters["num_seeds"] = int(text_parameters[1])
			exp_parameters["dim"] = int(text_parameters[2])
			exp_parameters["obj_function_name"] = text_parameters[3]

			
			
			exp_parameters["max_running_time"] = int(text_parameters[4])
			optimizer_type = text_parameters[5]
			exp_parameters["n_iter"] = 100000000000
			# the optimizer_name will be of the shape "RS_400"
			exp_parameters["optimizer"] = optimizer_from_name_and_parameters(optimizer_type, text_parameters[6:], copy.deepcopy(optimizer_general_parameters))
			exp_parameters["sub_exp_name"] = parameters["exp_name"]+"_"+exp_parameters["optimizer"].optimizer_name

			# The obj_function is (potentially) different for each seed, as for each seed the homology profiles found (saved by the obj_function) are saved in a different folder
			# Hence exp_parameters["obj_function"] is a dictionary whose keys are the seeds and whose values are the functions
			# This is not great, but I don't see how to do it better without extensive changes
			exp_parameters["obj_function"] = obj_function_from_string(text_parameters[3], parameters["local_path"], exp_number, exp_parameters["num_seeds"], parameters["saved_results_folder"], exp_parameters["sub_exp_name"])
			# print(f'Test {exp_parameters["obj_function"][0]()}')
			# print(f'Test {exp_parameters["obj_function"][1]()}')
			# print(f'Test {exp_parameters["obj_function"][2]()}')
			parameters[exp_number] = exp_parameters

	return parameters
