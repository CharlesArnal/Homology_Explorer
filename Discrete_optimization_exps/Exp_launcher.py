
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import copy

from get_parameters import get_parameters_exp_0


exp_name = sys.argv[1]
exp_numbers = [int(index) for index in sys.argv[2:]]

my_local_path = current_dir# "/home/charles/Desktop/ML_RAG/Code/Discrete_optimization_exps"


exps_param_function = get_parameters_exp_0
params_file = os.path.join(my_local_path, "Parameters_files",f"param_{exp_name}.txt")
parameters = exps_param_function(params_file)

# if exp_name == "exp_0":
# 	exps_param_function = get_parameters_exp_0
# 	params_file = os.path.join(my_local_path, "Parameters_files","param_exp_0.txt")
# 	parameters = exps_param_function(params_file)

# if exp_name == "exp_1_RS":
# 	exps_param_function = get_parameters_exp_0
# 	params_file = os.path.join(my_local_path, "Parameters_files","param_exp_1_RS.txt")
# 	parameters = exps_param_function(params_file)

# if exp_name == "exp_test_triangulations_RS":
# 	exps_param_function = get_parameters_exp_0
# 	params_file = os.path.join(my_local_path, "Parameters_files","param_exp_test_triangulations_RS.txt")
# 	parameters = exps_param_function(params_file)


# in params_file, the first line defines which experiments must be launched parallelly
# the second line contains various global parameters
# each other line gives the parameters for one experiment
# see get_parameters.py for details

# parameters is a dictionary structured as follows :
# {"exp_name" : "exp_1", "exp_batches":[["0","1","2"], ["3","4"]], ...other global parameters ...,  1 : dict_1, 2 : dict_2, ....}
# dict_i contains the parameters for experiment i
parameters = exps_param_function(params_file)

for exp_number in exp_numbers:
	exp_parameters = parameters[exp_number]
	opti_template = exp_parameters["optimizer"]
	if not os.path.isdir(os.path.join(my_local_path, parameters["saved_results_folder"])):
			os.mkdir(os.path.join(my_local_path, parameters["saved_results_folder"]))
	saved_results_folder = os.path.join(parameters["saved_results_folder"],f"{exp_parameters['obj_function_name']}")
	if not os.path.isdir(os.path.join(my_local_path, saved_results_folder)):
		os.mkdir(os.path.join(my_local_path, saved_results_folder))

	for seed in range(exp_parameters["num_seeds"]):
		opti = copy.deepcopy(opti_template)
		opti.exp_name = exp_parameters["sub_exp_name"]+ f"_{exp_parameters['obj_function_name']}" + f"_seed_{seed}"
		opti.random_seed = seed
		opti.saved_results_folder = saved_results_folder
		# one objective function for each seed, as the objective function is in charge of storing certain results in folders whose name depends on the seed
		opti.optimize(exp_parameters["n_iter"], exp_parameters["dim"], exp_parameters["obj_function"][seed], initial_solutions = None, \
			stopping_condition = None, max_running_time = exp_parameters["max_running_time"], clear_log = True)