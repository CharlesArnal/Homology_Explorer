
import random
import numpy as np

from math import comb, floor

import time
import os
import sys

from move_selectors import create_move_selector, Random_Triang_Selector, Greedy_Selector,\
	 Greedy_Expanding_Selector, Greedy_Randomized_Expanding_Selector, Greedy_Randomized_Selector
from move_generators import generate_moves_nb_triangs, generate_moves_nb_triangs_nb_signs
from signs_optimizers import TS_signs_optimizer, MCTS_signs_optimizer, RS_signs_optimizer
from Walking_search import Walking_search
from Initialization_functions import standard_triang_and_signs_initialization, custom_initialization_function_with_starting_triang_and_signs
from score_polymake_wrapper import calc_score

from utilities import read_first_experiment_param_file

# modified version, I directly tell it the experiment number (as opposed to a batch number), as each experiment is done separately

# give the experiment number as a parameter
experiment_num = int(sys.argv[1]) 



print(f"\nExperience number {experiment_num}")


with_symmetries = False


local_path = os.path.dirname(os.path.realpath(__file__))


find_new_topologies = True


SAVE_PERIOD = 20


save_period = 1
feedback_frequency = 1

# grows the triangulation
def expanding_objective_function(triangs_file, signs_file, points_file, points_indices_file):
	scores =[]
	with open(triangs_file,"r") as f:
		for line in f:
			scores.append(len(line.split("},{")))
	return np.array(scores)

# grows the triangulation and saves the homologies met along the way while doing so
def expanding_and_look_while_growing_objective_function(triangs_file, signs_file, points_file, points_indices_file):
	# we compute and record the homology, but don't use it as an objective function
	my_objective_function(triangs_file, signs_file, points_file, points_indices_file)
	
	scores =[]
	with open(triangs_file,"r") as f:
		for line in f:
			scores.append(len(line.split("},{")))
	return np.array(scores)



# start of the program

parameters = read_first_experiment_param_file("parameters_exps_0.1.txt",experiment_num)
dim, degree, signs_opti_time, total_time, signs_opti_alg, scoring_script, initial_triangulation_type, look_while_growing = \
	parameters["dim"], parameters["degree"], parameters["signs_opti_time"], parameters["total_time"], parameters["signs_opti_alg"], \
		parameters["scoring_script"], parameters["initial_triangulation_type"], parameters["look_while_growing_triangulation"]
if scoring_script == "Scoring/score_b_total.pl":
	obj_fun_name = "bt"
elif scoring_script == "Scoring/score_b_total_w_alpha_b_0.pl":
	obj_fun_name  = "btpa0"
elif scoring_script == "Scoring/score_b_total_w_alpha_b_1.pl":
	obj_fun_name = "btpa1"
elif scoring_script == "Scoring/score_b_0_w_alpha_b_1.pl":
	obj_fun_name = "b0pa1"
elif scoring_script == "Scoring/score_b_0.pl":
	obj_fun_name = "b0"
elif scoring_script == "value_novelty":
	obj_fun_name = "value_novelty"

# TODO 
if degree>5 : 
	signs_opti_time = signs_opti_time*2

NAME_EXP = f"exp_{experiment_num}_dim_{dim}_deg_{degree}_obj_{obj_fun_name}_sopttime_{signs_opti_time}_alg_{signs_opti_alg}_intriang_{initial_triangulation_type}_lookwhilegrow_{look_while_growing}"
polymake_scoring_script = scoring_script
temp_files_folder = f"Temp_files_exp_0.1/exp_num_{experiment_num}"

print("\n\n\n------------------------------------------------")
print("------------------------------------------------")
print(f"Experiment number {experiment_num}")
print(f"\nExperiment parameters : dim {dim}, degree {degree}, objective function: {obj_fun_name}, signs optimisation time: {signs_opti_time}, "+\
		f"signs optimisation algorithm: {signs_opti_alg}, initial triangulation type: {initial_triangulation_type}, look around while growing the triangulation: {look_while_growing}")



output_scoring_file = temp_files_folder+'/temp_score.txt'
temp_homologies_file = temp_files_folder+'/temp_homologies_file.txt'
list_of_homologies_file = f"Saved_files_exp_0.1/dim_{dim}_degree_{degree}/homologies_"+NAME_EXP+".txt"
save_perf_file = os.path.join(local_path,f"Saved_files_exp_0.1/dim_{dim}_degree_{degree}/perf_wrt_time_"+NAME_EXP+".txt") 


def my_objective_function(triangs_file, signs_file, points_file, points_indices_file):
	# needed because calc_score takes as input file names without the local_path
	triangs_file = triangs_file.replace(local_path+"/","")
	signs_file = signs_file.replace(local_path+"/","")
	points_file = points_file.replace(local_path+"/","")
	points_indices_file = points_indices_file.replace(local_path+"/","")

	return calc_score(local_path, temp_files_folder, polymake_scoring_script, signs_file,\
			triangs_file, points_file, points_indices_file, output_scoring_file,\
			degree, dim, find_new_topologies, list_of_homologies_file, temp_homologies_file)

random.seed(experiment_num)
np.random.seed(experiment_num)



if initial_triangulation_type != "Trivial":
	print("\n------------------------------------------------")
	print("Finding an initial triangulation with a large number of vertices")
	initialization_function = standard_triang_and_signs_initialization
	move_generator = generate_moves_nb_triangs_nb_signs#generate_moves_nb_triangs#
	move_selector = None
	if look_while_growing:
		move_selector = create_move_selector(Greedy_Randomized_Selector,expanding_and_look_while_growing_objective_function)
	else:
		move_selector = create_move_selector(Greedy_Randomized_Selector,expanding_objective_function)
	signs_optimizer = None
	if signs_opti_alg == "TS":
		signs_optimizer = TS_signs_optimizer
	elif signs_opti_alg == "MCTS":
		signs_optimizer = MCTS_signs_optimizer
	polymake_scoring_script = scoring_script
	optimize_signs_separately = False
	optimizer_stopping_time = 0
	initial_signs_opti_time = 0
	# max number of vertices in triangulation is C(degree+dim,dim)
	n_iter = 0
	if initial_triangulation_type == "Medium":
		n_iter = floor(float(comb(dim+degree, degree))*0.5)
	if initial_triangulation_type == "Large":
		n_iter = floor(float(comb(dim+degree, degree)))

	starting_time = time.time()

	# save_perf_file is None here, unlike below
	Walking_search(degree, dim, n_iter, initialization_function, move_generator, move_selector, local_path, temp_files_folder, feedback_frequency,\
	starting_time, save_period, None, list_of_homologies_file, temp_homologies_file , polymake_scoring_script = polymake_scoring_script,\
		optimize_signs_separately=optimize_signs_separately, signs_optimizer=signs_optimizer, optimizer_stopping_time= optimizer_stopping_time,\
		initial_signs_opti_time =initial_signs_opti_time, stopping_obj_value =None, stopping_time = None,\
		n_solutions_saved = 0, saved_solutions_file = None)

print("\n------------------------------------------------")
print("Optimizing the objective function\n")


if initial_triangulation_type != "Trivial":
	initialization_function = custom_initialization_function_with_starting_triang_and_signs(degree,dim,local_path, temp_files_folder,\
	temp_files_folder+"/all_points.dat",temp_files_folder+"/current_triang.dat",\
	temp_files_folder+"/current_signs.dat",temp_files_folder+"/chiro.dat",temp_files_folder+"/symmetries.dat")
else:
	initialization_function = standard_triang_and_signs_initialization

move_selector = create_move_selector(Greedy_Randomized_Selector,my_objective_function)
signs_optimizer = None
optimize_signs_separately = None
# Remark : all optimizers should do better than a neighborhood search
move_generator = None 
if signs_opti_alg =="TS":
	signs_optimizer = TS_signs_optimizer
	optimize_signs_separately = True
	move_generator = generate_moves_nb_triangs#generate_moves_nb_triangs_nb_signs#
elif signs_opti_alg == "MCTS":
	signs_optimizer = MCTS_signs_optimizer
	optimize_signs_separately = True
	move_generator = generate_moves_nb_triangs#generate_moves_nb_triangs_nb_signs#
elif signs_opti_alg == "None":
	signs_optimizer = None
	optimize_signs_separately = False
	move_generator = generate_moves_nb_triangs_nb_signs
polymake_scoring_script = scoring_script
optimizer_stopping_time = signs_opti_time
initial_signs_opti_time = signs_opti_time
stopping_time = total_time
n_iter = 1000000000

starting_time = time.time()

Walking_search(degree, dim, n_iter, initialization_function, move_generator, move_selector, local_path, temp_files_folder, feedback_frequency,\
	starting_time, save_period, save_perf_file, list_of_homologies_file, temp_homologies_file , polymake_scoring_script = polymake_scoring_script,\
		optimize_signs_separately=optimize_signs_separately, signs_optimizer=signs_optimizer, optimizer_stopping_time= optimizer_stopping_time,\
		initial_signs_opti_time =initial_signs_opti_time, stopping_obj_value =None, stopping_time = stopping_time,\
		n_solutions_saved = 0, saved_solutions_file = None)


