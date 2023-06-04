
import matplotlib.pyplot as plt
from math import ceil
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)


from utilities import ad_hoc_mean

import numpy as np



local_path = current_dir

saved_results_folder = "Saves/exp_1_MCTS_depth_triangs"

exp_name = "exp_1_MCTS_depth_triangs"

optimizer_type = "MCTS"

# TS 1
#parameters_sets = [["10","3000", "3", "25", "50", "False"],["30","3000", "3", "25", "50", "False"],["100","3000", "3", "25", "50", "False"]]

# GA 1
#parameters_sets = [["10","300", "rank"],["30","300", "rank"],["100","300", "rank"]]

# # MCTS 1
parameters_sets = [["3","10"], ["5","10"]]

# # RS
# parameters_sets = [["100"]]


num_seeds = 4

# TODO only for the first set of experiments, due to mistake in naming conventions
#obj_functions = [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}"), (f"homology_b0pab1_Dim.3.deg.6_2_10", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_2_10", f"{51}") ]
obj_functions = [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.20_2_20", f"{231}"), (f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ]

results = dict()
num_homologies_found = dict()

for obj_function_pair in obj_functions :
    obj_function = obj_function_pair[0]
    results_for_obj_fun = dict()
    num_homologies_found_for_obj_fun = dict()
    for parameters_set in parameters_sets:
        results_for_param_set = []
        num_homologies_found_for_param_set = []
        for seed in range(num_seeds):
            # example : "exp_1_RS_RS_1000_graph_2.1_9_seed_0_scores"
            file_name = "_".join([exp_name, optimizer_type, "_".join(parameters_set), obj_function, "seed", str(seed), "scores"])
            file_path = os.path.join(local_path, saved_results_folder, obj_function, file_name)
            with open(file_path, 'r') as f:
                sub_results = np.loadtxt(f,dtype=float,ndmin=2) # important
                results_for_param_set.append(sub_results)
            # example : "exp_1_RS_RS"
            file_name = "_".join(["homologies", exp_name, optimizer_type, "_".join(parameters_set), obj_function, "seed", str(seed)])
            file_path = os.path.join(local_path, saved_results_folder, obj_function, "homologies", file_name)
            with open(file_path, 'r') as f:
                num_homologies_found_for_param_set.append(len(f.readlines()))
        results_for_obj_fun["_".join(parameters_set)] = ad_hoc_mean(results_for_param_set)
        num_homologies_found_for_obj_fun["_".join(parameters_set)] = np.mean(num_homologies_found_for_param_set)
    results[obj_function] = results_for_obj_fun
    num_homologies_found[obj_function] = num_homologies_found_for_obj_fun



width_plot = ceil(len(obj_functions)/2.0)
fig, axs = plt.subplots(2, width_plot, sharey = False)

for index, obj_function_pair in enumerate(obj_functions):
    obj_function = obj_function_pair[0]
    for parameters_set in parameters_sets:
        parameters_string = "_".join(parameters_set)
        plot_1, plot_2 = int(index/width_plot), index%width_plot
        axs[plot_1,plot_2].plot(results[obj_function][parameters_string][:,0], results[obj_function][parameters_string][:,1], label = f"{parameters_string}, hom found : {num_homologies_found[obj_function][parameters_string]}")
        axs[plot_1,plot_2].title.set_text(obj_function)
        axs[plot_1,plot_2].legend()

plt.show()


# # colors = ["g","b","r","y"]




# for index, performance in enumerate(performances):
#     plt.plot(performance[:,0],performance[:,1],colors[index])
# #plt.plot([100,300],[10,30])
# plt.show()

