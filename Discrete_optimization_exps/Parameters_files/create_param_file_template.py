
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from sklearn.model_selection import ParameterGrid

my_local_path =  parent_dir


param_file = os.path.join(my_local_path, "Parameters_files","param_exp_1_RS.txt")

exp_name = "exp_1_RS"

saved_results_folder = exp_name

if not os.path.isdir(os.path.join(my_local_path, saved_results_folder)):
    os.mkdir(os.path.join(my_local_path, saved_results_folder))



#param_grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
param_grid = {"optimizer_type" : ["RS"], 
              "num_seeds" : ["3"],
              "obj_function": [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]] ,   # the second entry of each pair is the dimension (it is determined by the objective function)
              "max_running_time": [f"{3*10}"],
              "optimizer_parameters": [["1000"], ["2000"]]
            }

# TODO make it so that the parameters written adapt to the optimizer's parameters


param_list = ParameterGrid(param_grid) 

with open(param_file, "w") as f:
    f.write(exp_name)
    f.write("\n")
    # TODO change depending on needs
    # batch decomposition
    f.write(str([[index +1] for index, _ in enumerate(param_list)]))
    f.write("\n")
    f.write(my_local_path + " " +saved_results_folder)
    f.write("\n")
    f.write("exp_num num_seeds dim obj_function max_running_time optimizer_type specific_parameters")
    f.write("\n")
    for index, prm_set in enumerate(param_list):
        f.write(f"{index+1} {prm_set['num_seeds']} {prm_set['obj_function'][1]} {prm_set['obj_function'][0]} {prm_set['max_running_time']} {prm_set['optimizer_type']} " + " ".join(prm_set['optimizer_parameters']))
        f.write("\n")
   