import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)


from utilities import turn_all_2D_or_3D_or_4D_homologies_files_in_folder_into_single_table


local_path = current_dir

exp_name = "exp_1_RL_learning_rate_triangs"

optimizer_type = "RL"

# TS 1
#parameters_sets = [["10","3000", "3", "25", "50", "False"],["30","3000", "3", "25", "50", "False"],["100","3000", "3", "25", "50", "False"]]
# TS 2
# parameters_sets = [["10","3000", "3", "25", "50", "False"],["10","3000", "3", "25", "200", "False"],["10","3000", "3", "100", "50", "False"],["10","3000", "3", "100", "200", "False"],\
#                                         ["10","3000", "6", "25", "50", "False"],["10","3000", "6", "25", "200", "False"],["10","3000", "6", "100", "50", "False"],["10","3000", "6", "100", "200", "False"]]
# TS 3
# parameters_sets = [["10","3000", "3", "100", "200", "False"], ["10","3000", "3", "100", "200", "True", "0.05", "0.05", "0.05", "0.05", "0.05", "0.05"] ]



# MCTS 1
#parameters_sets = [["3", "10"],["5","10"]]
# MCTS 2
# parameters_sets = [["5","10"], ["5","30"], ["5","100"]]

# RS
#parameters_sets = [["100"]]

# GA 1
#parameters_sets = [["10","300", "rank"],["30","300", "rank"],["100","300", "rank"]]
# GA 2
# parameters_sets = [["100","200", "rank"],["100","300", "rank"],["100","600", "rank"]]
# GA 3
# parameters_sets = [["100","300", "sss"],["100","300", "rws"],["100","300", "sus"],["100","300", "tournament"]]


# RL 1
parameters_sets = [["3", "128", "128", "32", "0.01", "500", "False", "0"], ["3", "128", "128", "32", "0.003", "500", "False", "0"], ["3", "128", "128", "32", "0.001", "500", "False", "0"], ["3", "128", "128", "32", "0.0003", "500", "False", "0"], ["3", "128", "128", "32", "0.0001", "500", "False", "0"]]
 


num_seeds = 4

# the incorrect line for the first batch of experiments
#obj_functions = [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}"), (f"homology_b0pab1_Dim.3.deg.6_2_10", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_2_10", f"{51}") ]
# TODO replace by line below
obj_functions = [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}"), (f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ]

results = dict()

for obj_function_pair in obj_functions :
	obj_function = obj_function_pair[0]
	name_components = obj_function.split("_")
	#------
	# temporary fix due to error first experiment TODO change for future experiments
	dim = int(name_components[3])
	degree = int(name_components[4])
	# if obj_function =="homology_bt_Harnack.10_2_10":
	# 	dim = 2
	# 	degree = 10
	# elif obj_function == "homology_bt_Harnack.20_2_20":
	# 	dim = 2
	# 	degree = 20
	# elif obj_function == "homology_b0pab1_Dim.3.deg.6_2_10":
	# 	dim = 3
	# 	degree = 6
	# elif obj_function == "homology_b0pab1_Dim.4.deg.4_2_10":
	# 	dim = 4
	# 	degree = 4
	# else :
	# 	print("invalid")
	#------

	for parameters_set in parameters_sets:
		sub_path = os.path.join(local_path, "Saves", exp_name, obj_function, "homologies")
		configuration_name = "_".join(["homologies",exp_name, optimizer_type, "_".join(parameters_set), obj_function])
		path_table = os.path.join(local_path, "Saves", exp_name, "total_table_"+configuration_name)
		turn_all_2D_or_3D_or_4D_homologies_files_in_folder_into_single_table(sub_path, configuration_name, path_table, degree, dim)



		