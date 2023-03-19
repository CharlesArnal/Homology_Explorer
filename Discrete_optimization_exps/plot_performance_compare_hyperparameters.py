
import matplotlib.pyplot as plt
from math import ceil
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import numpy as np



def ad_hoc_mean(mylist):
    max_length = max([len(my_array) for my_array in mylist])
    min_length = min(len(my_array) for my_array in mylist)
    if max_length-min_length>max_length*0.01:
        print(f"Max_length {max_length}, min length {min_length}")
    mylist = np.array([my_array[:min_length,:] for my_array in mylist])
    return np.mean(mylist,axis = 0)

local_path = current_dir

saved_results_folder = "Saves/exp_1_RS"

exp_name = "exp_1_RS"

optimizer_type = "RS"

parameters_sets = [["1000"]]

num_seeds = 1

obj_functions = [(f"graph_2.1_{N}", f"{int(N*(N-1)/2)}") for N in [9, 19, 29]] + [(f"graph_2.3_{N}", f"{int(N*(N-1)/2)}") for N in [20, 30, 40]]

results = dict()

for obj_function_pair in obj_functions :
    obj_function = obj_function_pair[0]
    results_for_obj_fun = dict()
    for parameters_set in parameters_sets:
        results_for_param_set = []
        for seed in range(num_seeds):
            # example : "exp_1_RS_RS_1000_graph_2.1_9_seed_0_scores"
            file_name = "_".join([exp_name, optimizer_type, "_".join(parameters_set), obj_function, "seed", str(seed), "scores"])
            file_path = os.path.join(local_path, saved_results_folder, obj_function, file_name)
            with open(file_path, 'r') as f:
                results_for_param_set.append(np.loadtxt(f,dtype=float))
        results_for_obj_fun["_".join(parameters_set)] = ad_hoc_mean(results_for_param_set)
    results[obj_function] = results_for_obj_fun



width_plot = ceil(len(obj_functions)/2.0)
fig, axs = plt.subplots(2, width_plot, sharey = False)

for index, obj_function_pair in enumerate(obj_functions):
    obj_function = obj_function_pair[0]
    for parameters_set in parameters_sets:
        parameters_string = "_".join(parameters_set)
        plot_1, plot_2 = int(index/width_plot), index%width_plot
        axs[plot_1,plot_2].plot(results[obj_function][parameters_string][:,0], results[obj_function][parameters_string][:,1], label = parameters_string)
        axs[plot_1,plot_2].title.set_text(obj_function)
        axs[plot_1,plot_2].legend()

plt.show()


# # colors = ["g","b","r","y"]




# for index, performance in enumerate(performances):
#     plt.plot(performance[:,0],performance[:,1],colors[index])
# #plt.plot([100,300],[10,30])
# plt.show()

