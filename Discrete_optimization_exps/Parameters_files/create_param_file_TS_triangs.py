
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from sklearn.model_selection import ParameterGrid

my_local_path =  parent_dir


param_file = os.path.join(my_local_path, "Parameters_files","param_exp_3_TS_more_exploration_triangs.txt")

exp_name = "exp_3_TS_more_exploration_triangs"

saved_results_folder = exp_name

if not os.path.isdir(os.path.join(my_local_path, saved_results_folder)):
    os.mkdir(os.path.join(my_local_path, saved_results_folder))



# regrouper pour gagner du temps
# 10 30 100
# 3000
# 3, 6
# 25, 100
# 50, 200
# False -> True
# percent a voir

# [["10","3000", "3", "25", "50", "False"]]

#     size_pop STM_length  MTM_length  SI_thresh SD_thresh more_exploration percent_more_exploration
#         0         1          2              3       4           5               6-...






#param_grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
# Exp 1 (size_pop)
# param_grid = [{"optimizer_type" : ["TS"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}") ],
#               "max_running_time": [f"{3600*10}"],
#               "optimizer_parameters": [["10","3000", "3", "25", "50", "False"],["30","3000", "3", "25", "50", "False"],["100","3000", "3", "25", "50", "False"]]
#             },
#             {"optimizer_type" : ["TS"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
#               "max_running_time": [f"{3600*2}"],
#               "optimizer_parameters": [["10","3000", "3", "25", "50", "False"],["30","3000", "3", "25", "50", "False"],["100","3000", "3", "25", "50", "False"]]
#             }

#             ]

# # Exp 2 (memory)
# 3, 6
# 25, 100
# 50, 200
# param_grid = [{"optimizer_type" : ["TS"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}") ],
#               "max_running_time": [f"{3600*10}"],
#               "optimizer_parameters": [["10","3000", "3", "25", "50", "False"],["10","3000", "3", "25", "200", "False"],["10","3000", "3", "100", "50", "False"],["10","3000", "3", "100", "200", "False"],\
#                                        ["10","3000", "6", "25", "50", "False"],["10","3000", "6", "25", "200", "False"],["10","3000", "6", "100", "50", "False"],["10","3000", "6", "100", "200", "False"]]
#             },
#             {"optimizer_type" : ["TS"], 
#               "num_seeds" : ["4"],
#               #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
#               "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
#               "max_running_time": [f"{3600*2}"],
#               "optimizer_parameters": [["10","3000", "3", "25", "50", "False"],["10","3000", "3", "25", "200", "False"],["10","3000", "3", "100", "50", "False"],["10","3000", "3", "100", "200", "False"],\
#                                        ["10","3000", "6", "25", "50", "False"],["10","3000", "6", "25", "200", "False"],["10","3000", "6", "100", "50", "False"],["10","3000", "6", "100", "200", "False"]]
#             }

#             ]


# # Exp 3 (more_exploration)
# "0.05", "0.05", "0.05", "0.05", "0.05", "0.05"
param_grid = [{"optimizer_type" : ["TS"], 
              "num_seeds" : ["4"],
              #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}") ],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["10","3000", "3", "100", "200", "False"], ["10","3000", "3", "100", "200", "True", "0.05", "0.05", "0.05", "0.05", "0.05", "0.05"] ]
            },
            {"optimizer_type" : ["TS"], 
              "num_seeds" : ["4"],
              #"obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["10","3000", "3", "100", "200", "False"], ["10","3000", "3", "100", "200", "True", "0.05", "0.05", "0.05", "0.05", "0.05", "0.05"] ]
              }

            ]





param_list = ParameterGrid(param_grid) 

with open(param_file, "w") as f:
    f.write(exp_name)
    f.write("\n")
    # TODO change depending on needs
    # batch decomposition
    f.write(str([[1,2],[3,4], [5,6, 7,8]]))
    #f.write(str([[index +1] for index, _ in enumerate(param_list)]))
    f.write("\n")
    f.write(my_local_path + " " +saved_results_folder)
    f.write("\n")
    f.write("exp_num num_seeds dim obj_function max_running_time optimizer_type specific_parameters")
    f.write("\n")
    for index, prm_set in enumerate(param_list):
        f.write(f"{index+1} {prm_set['num_seeds']} {prm_set['obj_function'][1]} {prm_set['obj_function'][0]} {prm_set['max_running_time']} {prm_set['optimizer_type']} " + " ".join(prm_set['optimizer_parameters']))
        f.write("\n")
   