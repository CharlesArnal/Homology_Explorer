import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)


from utilities import read_first_experiment_param_file

from Current_Point import Current_Point
from Homology_Explorer import Homology_Explorer



# triangulation 1 and 2, Harnack degree 10 and 20

"""
degree = 20

exp_name = f"Harnack_{degree}"
explorer_name = "triangulation_generator"

dim = 2


if '/home/charles' in os.getcwd() :
	local_path = '/home/charles/Desktop/ML_RAG/Code/'

temp_files_folder = f"Discrete_optimization_exps/Objective_functions/General_test_temp_files/{exp_name}"

if not os.path.exists(os.path.join(local_path, temp_files_folder)):
	os.mkdir(os.path.join(local_path, temp_files_folder))

saved_results_folder = f"Discrete_optimization_exps/Objective_functions/General_test_saved_files/{exp_name}"
if not os.path.exists(os.path.join(local_path, saved_results_folder)):
	os.mkdir(os.path.join(local_path, saved_results_folder))

current_point_folder = f"Discrete_optimization_exps/Objective_functions/General_test_temp_files/{exp_name}/current_point"

explorer = Homology_Explorer(dim, degree, local_path, temp_files_folder, saved_results_folder, exp_name, explorer_name, saving_perf_period = 20, verbose= True)

explorer.initialize_with_Harnack_curve(current_point_folder)

print(f"Current homology {explorer.current_point.compute_homology()}")
"""

# triangulation 3 and 4, using previously found configurations 


#degree = 6
#dim = 3

degree = 5
dim = 4

exp_name = f"Dim_{dim}_deg_{degree}"
explorer_name = "triangulation_generator"


if '/home/charles' in os.getcwd() :
	local_path = '/home/charles/Desktop/ML_RAG/Code/'

temp_files_folder = f"Discrete_optimization_exps/Objective_functions/General_test_temp_files/{exp_name}"

if not os.path.exists(os.path.join(local_path, temp_files_folder)):
	os.mkdir(os.path.join(local_path, temp_files_folder))

saved_results_folder = f"Discrete_optimization_exps/Objective_functions/General_test_saved_files/{exp_name}"
if not os.path.exists(os.path.join(local_path, saved_results_folder)):
	os.mkdir(os.path.join(local_path, saved_results_folder))

current_point_folder = f"Discrete_optimization_exps/Objective_functions/General_test_temp_files/{exp_name}/current_point"

current_point = Current_Point(dim, degree,  local_path, current_point_folder)

current_point.complete_current_point(verbose = True, force_recompute = False)

print("Current point completed")
print(f"Current homology {current_point.compute_homology()}")



"""
if dim == 2 :
	explorer.current_point.visualize_triangulation()
if dim in {2,3}:
	explorer.current_point.visualize_hypersurface()

"""