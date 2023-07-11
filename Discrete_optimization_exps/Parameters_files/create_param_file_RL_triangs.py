
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from sklearn.model_selection import ParameterGrid

my_local_path =  parent_dir


param_file = os.path.join(my_local_path, "Parameters_files","param_exp_2_RL_layers_triangs.txt")

exp_name = "exp_2_RL_layers_triangs"

saved_results_folder = exp_name

if not os.path.isdir(os.path.join(my_local_path, saved_results_folder)):
    os.mkdir(os.path.join(my_local_path, saved_results_folder))


	# 	n_layers      width_layers   learning_rate  n_sessions   	min_randomness    alpha
	# 	    0         1-n_layers       n_layers+1     n_layers+2        n_layers+3   n_layers+4


# [128, 128, 32], [256, 256, 128, 32], [256, 256, 256, 128, 32]
# 3, 4, 5 (associated to the line above)
# (0.001) 0.01, 0.003, 0.001, 0.0003, 0.0001 COMMENCER PAR LA LEARNING RATE
# (500) 200 500 1000 2000
# à voir la suite (False pour l'instant)

# [["3", "128", "128", "32", "0.001", "500", "False", "0"]]

#param_grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
# Exp 1 learning_rate
# param_grid = [{"optimizer_type" : ["RL"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}")],
#               "max_running_time": [f"{3600*10}"],
#               "optimizer_parameters": [["3", "128", "128", "32", "0.01", "500", "False", "0"], ["3", "128", "128", "32", "0.003", "500", "False", "0"], ["3", "128", "128", "32", "0.001", "500", "False", "0"], ["3", "128", "128", "32", "0.0003", "500", "False", "0"], ["3", "128", "128", "32", "0.0001", "500", "False", "0"]]
#             },
#             {"optimizer_type" : ["RL"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
#               "max_running_time": [f"{3600*2}"],
#               "optimizer_parameters": [["3", "128", "128", "32", "0.01", "500", "False", "0"], ["3", "128", "128", "32", "0.003", "500", "False", "0"], ["3", "128", "128", "32", "0.001", "500", "False", "0"], ["3", "128", "128", "32", "0.0003", "500", "False", "0"], ["3", "128", "128", "32", "0.0001", "500", "False", "0"]]
#             }
#             ]

#param_grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
# Exp 2 layers
param_grid = [{"optimizer_type" : ["RL"], 
              "num_seeds" : ["4"],
              #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["3", "128", "128", "32", "0.0003", "500", "False", "0"], ["4", "256", "256", "128", "32", "0.0003", "500", "False", "0"], ["5", "256", "256", "256", "128", "32", "0.0003", "500", "False", "0"]]
            },
            {"optimizer_type" : ["RL"], 
              "num_seeds" : ["4"],
              #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["3", "128", "128", "32", "0.0003", "500", "False", "0"], ["4", "256", "256", "128", "32", "0.0003", "500", "False", "0"], ["5", "256", "256", "256", "128", "32", "0.0003", "500", "False", "0"]]
            }
            ]



param_list = ParameterGrid(param_grid) 

with open(param_file, "w") as f:
    f.write(exp_name)
    f.write("\n")
    # TODO change depending on needs
    # batch decomposition
    f.write(str([[7, 8], [9, 10], [11, 12]]))#,[6], [7,8], [9,10] , [11,12, 13,14,15, 16, 17, 18,19,20]]))
    #f.write(str([[index +1] for index, _ in enumerate(param_list)]))
    f.write("\n")
    f.write(my_local_path + " " +saved_results_folder)
    f.write("\n")
    f.write("exp_num num_seeds dim obj_function max_running_time optimizer_type specific_parameters")
    f.write("\n")
    for index, prm_set in enumerate(param_list):
        f.write(f"{index+1} {prm_set['num_seeds']} {prm_set['obj_function'][1]} {prm_set['obj_function'][0]} {prm_set['max_running_time']} {prm_set['optimizer_type']} " + " ".join(prm_set['optimizer_parameters']))
        f.write("\n")
   