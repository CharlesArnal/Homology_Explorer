
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import subprocess

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time

from Discrete_Optimizers.MCTS_Optimizer import MCTS_Optimizer
from get_parameters import get_parameters_exp_0


exp_name = sys.argv[1]

my_local_path = current_dir

#"/home/charles/Desktop/ML_RAG/Code/Discrete_optimization_exps"

# the first line of params_file defines which experiments must be launched parallelly
# the second line contains various global parameters
# each other line gives the parameters for one experiment
# see get_parameters.py for details

# parameters is a dictionary structured as follows :
# {"exp_name" : "exp_1", "exp_batches":[["0","1","2"], ["3","4"]], ...other global parameters ...,  1 : dict_1, 2 : dict_2, ....}
# dict_i contains the parameters for experiment i

if exp_name == "exp_0":
	exps_param_function = get_parameters_exp_0
	params_file = os.path.join(my_local_path, "Parameters_files","param_exp_0.txt")
	parameters = exps_param_function(params_file, exp_name)

if exp_name == "exp_1_RS":
	exps_param_function = get_parameters_exp_0
	params_file = os.path.join(my_local_path, "Parameters_files","param_exp_1_RS.txt")
	parameters = exps_param_function(params_file, exp_name)




for exp_batch in parameters["exp_batches"]:
	print(f"Starting batch {exp_batch}")
	#list_files = subprocess.run(["nohup","python3", "Exp_template_graphs.py", exp_name, *[str(index) for index in exp_batch], f"> {exp_name}_{exp_batch[0]}_to_{exp_batch[-1]}.out 2>&1 &"])
	print(" ".join(["nohup","python3", "Exp_template_graphs.py", exp_name, *[str(index) for index in exp_batch], f"> {exp_name}_{exp_batch[0]}_to_{exp_batch[-1]}.out 2>&1 &"]))
	os.system(" ".join(["nohup","python3", "Exp_template_graphs.py", exp_name, *[str(index) for index in exp_batch], f"> {exp_name}_{exp_batch[0]}_to_{exp_batch[-1]}.out 2>&1 &"]))
		
	


