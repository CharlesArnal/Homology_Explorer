
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


exp_name = "test_general"
explorer_name = "my_first_explorer"
dim = 2
degree = 4

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



explorer.initialize_with_random_triangulation_with_random_convex_hull(current_point_folder)

# explorer.initialize_with_Harnack_curve(current_point_folder)


# explorer.initialize_with_new_triangulation("Medium", current_point_folder, look_while_growing = False)

# explorer.current_point.copy_into_other_current_point(f"General_test_temp_files/{exp_name}/medium_point_comparison")

# explorer.current_point.create_current_points_indices_file(verbose = True)

# explorer.generate_random_triangulation_with_random_walk("Medium", 3, f"General_test_temp_files/{exp_name}/current_point_random_triang", look_while_growing= True)

# explorer.current_point.copy_into_other_current_point(f"General_test_temp_files/{exp_name}/current_point_copy")


n_iter = 5
max_running_time = 600
optimizer_type = "MCTS_Optimizer"
polymake_scoring_script = "Scoring/score_b_total.pl"
explorer.walking_search_on_triang_graph(n_iter = n_iter, polymake_scoring_script= polymake_scoring_script, max_running_time = max_running_time, \
							optimizer_type= optimizer_type, optimizer_max_running_time= 10)



print(explorer.current_point.compute_homology())
if dim ==2 :
	explorer.current_point.visualize_triangulation()
if dim in {2,3}:
	explorer.current_point.visualize_hypersurface()




# # give the batch number as a parameter
# exp_batch_num = int(sys.argv[1]) 


# print(f"\nExperience batch number {exp_batch_num}")


# with_symmetries = False


# if '/home/charles' in os.getcwd() :
#     local_path = '/home/charles/Desktop/ML_RAG/Code'



# find_new_topologies = True


# SAVE_PERIOD = 20


# save_period = 1
# feedback_frequency = 1

# # grows the triangulation
# def expanding_objective_function(triangs_file, signs_file, points_file, points_indices_file):
# 	scores =[]
# 	with open(triangs_file,"r") as f:
# 		for line in f:
# 			scores.append(len(line.split("},{")))
# 	return np.array(scores)

# # grows the triangulation and saves the homologies met along the way while doing so
# def expanding_and_look_while_growing_objective_function(triangs_file, signs_file, points_file, points_indices_file):
# 	# we compute and record the homology, but don't use it as an objective function
# 	my_objective_function(triangs_file, signs_file, points_file, points_indices_file)
	
# 	scores =[]
# 	with open(triangs_file,"r") as f:
# 		for line in f:
# 			scores.append(len(line.split("},{")))
# 	return np.array(scores)



# # start of the program
# for experiment_num in range((exp_batch_num-1)*3+1,exp_batch_num*3+1):
# 	parameters = read_first_experiment_param_file("parameters_exps_0.1.txt",experiment_num)
# 	dim, degree, signs_opti_time, total_time, signs_opti_alg, scoring_script, initial_triangulation_type, look_while_growing = \
# 		parameters["dim"], parameters["degree"], parameters["signs_opti_time"], parameters["total_time"], parameters["signs_opti_alg"], \
# 			parameters["scoring_script"], parameters["initial_triangulation_type"], parameters["look_while_growing_triangulation"]
# 	if scoring_script == "Scoring/score_b_total.pl":
# 		obj_fun_name = "bt"
# 	elif scoring_script == "Scoring/score_b_total_w_alpha_b_0.pl":
# 		obj_fun_name  = "btpa0"
# 	elif scoring_script == "Scoring/score_b_total_w_alpha_b_1.pl":
# 		obj_fun_name = "btpa1"
# 	elif scoring_script == "Scoring/score_b_0_w_alpha_b_1.pl":
# 		obj_fun_name = "b0pa1"
# 	elif scoring_script == "Scoring/score_b_0.pl":
# 		obj_fun_name = "b0"


# 	# TODO : corriger
# 	signs_opti_alg = "RL"
# 	signs_opti_time = 10
# 	initial_triangulation_type = "Medium"

# 	# TODO 
# 	if degree>5 : 
# 		signs_opti_time = signs_opti_time*2
	
# 	NAME_EXP = f"exp_{experiment_num}_dim_{dim}_deg_{degree}_obj_{obj_fun_name}_sopttime_{signs_opti_time}_alg_{signs_opti_alg}_intriang_{initial_triangulation_type}_lookwhilegrow_{look_while_growing}"
# 	polymake_scoring_script = scoring_script
# 	temp_files_folder = f"General_test_temp_files/exp_batch_{exp_batch_num}"
# 	current_point_folder = f"General_test_temp_files/exp_batch_{exp_batch_num}/current_point"
# 	seed = experiment_num

# 	print("\n\n\n------------------------------------------------")
# 	print("------------------------------------------------")
# 	print(f"Experiment number {experiment_num}")
# 	print(f"\nExperiment parameters : dim {dim}, degree {degree}, objective function: {obj_fun_name}, signs optimisation time: {signs_opti_time}, "+\
# 		 f"signs optimisation algorithm: {signs_opti_alg}, initial triangulation type: {initial_triangulation_type}, look around while growing the triangulation: {look_while_growing}")
# 	print(f"Random seed : {seed}")


# 	output_scoring_file = temp_files_folder+'/temp_score.txt'
# 	list_of_homologies_file = f"General_test_saved_files/dim_{dim}_degree_{degree}/homologies_"+NAME_EXP+".txt"
# 	save_perf_file = os.path.join(local_path,f"General_test_saved_files/dim_{dim}_degree_{degree}/perf_wrt_time_"+NAME_EXP+".txt") 


	

# 	random.seed(seed)
# 	np.random.seed(seed)

# 	# create initial triangulation
# 	current_point = standard_triang_and_signs_initialization(degree, dim, local_path, current_point_folder, temp_files_folder)
	

# 	if initial_triangulation_type != "Trivial":
# 		print("\n------------------------------------------------")
# 		print("Finding an initial triangulation with a large number of vertices")
# 		move_generator = generate_moves_nb_triangs_nb_signs#generate_moves_nb_triangs#
# 		move_selector = None
# 		if look_while_growing:
# 			move_selector = create_move_selector(Greedy_Randomized_Selector,expanding_and_look_while_growing_objective_function)
# 		else:
# 			move_selector = create_move_selector(Greedy_Randomized_Selector,expanding_objective_function)
# 		signs_optimizer_type = None
# 		if signs_opti_alg == "TS":
# 			signs_optimizer_type = "Tabu_Search_Optimizer"
# 		elif signs_opti_alg == "MCTS":
# 			signs_optimizer_type = "MCTS_Optimizer"
# 		elif signs_opti_alg == "RL":
# 			signs_optimizer_type = "RL_Optimizer"
# 		elif signs_opti_alg == "RS":
# 			signs_optimizer_type = "Random_Search_Optimizer"
# 		elif signs_opti_alg == "GA":
# 			signs_optimizer_type = "GA_Optimizer"
# 		polymake_scoring_script = scoring_script
# 		optimize_signs_separately = False
# 		optimizer_stopping_time = 0
# 		initial_signs_opti_time = 0


# 		# max number of vertices in triangulation is C(degree+dim,dim)
# 		n_iter = 0
# 		if initial_triangulation_type == "Medium":
# 			n_iter = floor(float(comb(dim+degree, degree))*0.5)
# 		if initial_triangulation_type == "Large":
# 			n_iter = floor(float(comb(dim+degree, degree)))

# 		# TODO delete
# 		n_iter = 2

# 		# save_perf_file is None here, unlike below
# 		Walking_search(degree, dim, n_iter, current_point, move_generator, move_selector, temp_files_folder, feedback_frequency,\
# 	 	save_period, None, list_of_homologies_file, polymake_scoring_script = polymake_scoring_script,\
# 		 optimize_signs_separately=optimize_signs_separately, signs_optimizer_type=signs_optimizer_type, optimizer_stopping_time= optimizer_stopping_time,\
# 			initial_signs_opti_time =initial_signs_opti_time, stopping_obj_value =None, stopping_time = None,\
# 		 n_solutions_saved = 0, saved_solutions_file = None)

# 	print("\n------------------------------------------------")
# 	print("Optimizing the objective function\n")

# 	# devient inutile
# 	# if initial_triangulation_type != "Trivial":
# 	# 	initialization_function = custom_initialization_function_with_starting_triang_and_signs(degree,dim,local_path, temp_files_folder,\
# 	# 	temp_files_folder+"/all_points.dat",temp_files_folder+"/current_triang.dat",\
# 	# 	temp_files_folder+"/current_signs.dat",temp_files_folder+"/chiro.dat",temp_files_folder+"/symmetries.dat")
# 	# else:
# 	# 	initialization_function = standard_triang_and_signs_initialization
	
# 	move_selector = create_move_selector(Greedy_Randomized_Selector,my_objective_function)
# 	signs_optimizer = None
# 	optimize_signs_separately = False
# 	# Remark : all optimizers should do better than a neighborhood search
# 	move_generator = None 
# 	if signs_opti_alg =="TS":
# 		signs_optimizer_type = "Tabu_Search_Optimizer"
# 		optimize_signs_separately = True
# 		move_generator = generate_moves_nb_triangs#generate_moves_nb_triangs_nb_signs#
# 	elif signs_opti_alg == "MCTS":
# 		signs_optimizer_type = "MCTS_Optimizer"
# 		optimize_signs_separately = True
# 		move_generator = generate_moves_nb_triangs#generate_moves_nb_triangs_nb_signs#
# 	elif signs_opti_alg == "RL":
# 		signs_optimizer_type = "RL_Optimizer"
# 		optimize_signs_separately = True
# 		move_generator = generate_moves_nb_triangs
# 	elif signs_opti_alg == "RS":
# 		signs_optimizer_type = "Random_Search_Optimizer"
# 		optimize_signs_separately = True
# 		move_generator = generate_moves_nb_triangs
# 	elif signs_opti_alg == "GA":
# 		signs_optimizer_type = "GA_Optimizer"
# 		optimize_signs_separately = True
# 		move_generator = generate_moves_nb_triangs
# 	elif signs_opti_alg == "None":
# 		signs_optimizer_type = None
# 		optimize_signs_separately = False
# 		move_generator = generate_moves_nb_triangs_nb_signs
# 	polymake_scoring_script = scoring_script
# 	optimizer_stopping_time = signs_opti_time
# 	initial_signs_opti_time = signs_opti_time
# 	stopping_time = total_time
# 	n_iter = 1000000000

	

# 	Walking_search(degree, dim, n_iter, current_point, move_generator, move_selector, temp_files_folder, feedback_frequency,\
# 	 	save_period, save_perf_file, list_of_homologies_file, polymake_scoring_script = polymake_scoring_script,\
# 		 optimize_signs_separately=optimize_signs_separately, signs_optimizer_type=signs_optimizer_type, optimizer_stopping_time= optimizer_stopping_time,\
# 			initial_signs_opti_time =initial_signs_opti_time, stopping_obj_value =None, stopping_time = stopping_time,\
# 		 n_solutions_saved = 0, saved_solutions_file = None)

	



# # tasks = ["random_dim3_d10"]#["Harnack_5","Harnack_10","Harnack_15","random_dim3_d5","random_dim3_d10"]
# # algorithms =  ["Tabu_search.py"]#,"RL_v3.py","ES_2.py"]
# # hyperparameters = ["conf1", "conf2"]#["conf1", "conf2"]
# # # TODO complete
# # MAX_RUNNING_TIME = {"Harnack_5": 30 ,"Harnack_10": 30,"Harnack_15": 30,"random_dim3_d5": 30 ,"random_dim3_d10": 30}



# # DEGREE= {"Harnack_5":5,"Harnack_10":10,"Harnack_15":15,"random_dim3_d5":5,"random_dim3_d10":10}
# # DIMENSION= {"Harnack_5":2,"Harnack_10":2,"Harnack_15":2,"random_dim3_d5":3,"random_dim3_d10":3}

# # # (on ne s'arrete pas pour la dim 3, dans l'espoir de trouver plusieurs topologies interessantes)
# # STOPPING_OBJ_VALUE={"Harnack_5":7,"Harnack_10":27,"Harnack_15":92,"random_dim3_d5":1000,"random_dim3_d10":1000}