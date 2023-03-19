
import random
import numpy as np

from math import comb, floor
import matplotlib.pyplot as plt
import time
import os
import sys


from utilities import read_first_experiment_param_file

from Current_Point import Current_Point
from Homology_Explorer import Homology_Explorer


exp_name = "chiro_degree_6"
explorer_name = "my_first_explorer"
dim = 4
degree = 6

my_seed = random.randint(0,100)
print(f"seed = {my_seed}")
random.seed(my_seed)
np.random.seed(my_seed)

if '/home/charles' in os.getcwd() :
	local_path = '/home/charles/Desktop/ML_RAG/Code'

temp_files_folder = f"General_test_temp_files/{exp_name}"

if not os.path.exists(os.path.join(local_path, temp_files_folder)):
	os.mkdir(os.path.join(local_path, temp_files_folder))

saved_results_folder = f"General_test_saved_files/{exp_name}"
if not os.path.exists(os.path.join(local_path, saved_results_folder)):
	os.mkdir(os.path.join(local_path, saved_results_folder))

current_point_folder = f"General_test_temp_files/{exp_name}/current_point"

explorer = Homology_Explorer(dim, degree, local_path, temp_files_folder, saved_results_folder, exp_name, explorer_name, saving_perf_period = 20, verbose= True)

explorer.initialize_with_new_triangulation("Trivial", current_point_folder = current_point_folder, look_while_growing= False)
