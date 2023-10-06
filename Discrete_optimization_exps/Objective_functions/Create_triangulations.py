import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import random


from utilities import read_first_experiment_param_file

from Current_Point import Current_Point
from Homology_Explorer import Homology_Explorer

from homology_objective_functions import b_0_p_a_b1


# Harnack

"""
degree = 25

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

print(f"Current homology {explorer.current_point.compute_own_homology()}")

"""


# triangulation 3 and 4, using previously found configurations 


# degree = 6
# dim = 3

# degree = 4
# dim = 4

# degree = 5
# dim = 4

degree = 8
dim = 3

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
print(f"Current homology {current_point.compute_own_homology()}")

"""

degree = 8
dim = 3

exp_name = f"Dim_{dim}_deg_{degree}"
explorer_name = "triangulation_generator"


my_seed = random.randint(0,100)
print(f"seed = {my_seed}")
random.seed(my_seed)
np.random.seed(my_seed)

local_path = current_dir
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


explorer = Homology_Explorer(dim, degree, local_path, temp_files_folder, saved_results_folder, exp_name, explorer_name, saving_perf_period = 0.001, verbose= True)

explorer.current_point = current_point

from homology_objective_functions import  create_objective_function_for_signs_optimization, triangulation_growing_objective_function

from signs_optimizers_for_triang_exploration import Signs_Optimizer_for_Triang_Exploration
from move_generators import generate_moves_nb_triangs, generate_moves_nb_triangs_nb_signs
from move_selectors import create_move_selector, Random_Triang_Selector, Greedy_Selector,\
	 Greedy_Expanding_Selector, Greedy_Randomized_Expanding_Selector, Greedy_Randomized_Selector



obj_function = triangulation_growing_objective_function
move_generator = generate_moves_nb_triangs_nb_signs
move_selector = create_move_selector(Greedy_Randomized_Selector, obj_function, must_compute_homology = False, \
	objective_function_takes_homology_as_input = False, observed_homologies_file = None, visited_homologies_file = None)

n_iter = 60
explorer.explore(n_iter, move_generator, move_selector)

print(f"Current homology {explorer.current_point.compute_own_homology()}")

# New triangulation by random exploration
"""

"""
if dim == 2 :
	explorer.current_point.visualize_triangulation()
if dim in {2,3}:
	explorer.current_point.visualize_hypersurface()

"""