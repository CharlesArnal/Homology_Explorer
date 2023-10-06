
import matplotlib.pyplot as plt
from math import ceil
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)


from utilities import ad_hoc_mean, ad_hoc_median

import numpy as np


local_path = current_dir

saved_results_folder = "Saves/exp_all_methods_triangs"

exp_name = "exp_all_methods_triangs"

optimizer_type = "RL"

parameters_sets = [["RS", ["100"]], ["RL", ["5", "256", "256", "256", "128", "32", "0.001", "500", "True", "0.1"]], ["GA", ["100","300", "rws"]], ["TS", ["10","3000", "3", "100", "200", "False"]], ["MCTS", ["5","10"]]]


num_seeds = 8

obj_functions =  [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),
                  (f"homology_bt_Harnack.25_2_25", f"{351}"), (f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}"), (f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ]



results = dict()
results_median = dict()
num_homologies_found = dict()

colors = ["b","g","r","c","m","y"]


for obj_function_pair in obj_functions :
    obj_function = obj_function_pair[0]
    results_for_obj_fun = dict()
    results_median_for_obj_fun = dict()
    num_homologies_found_for_obj_fun = dict()
    for parameters_set in parameters_sets:
        optimizer_type = parameters_set[0]
        opt_parameters = parameters_set[1]
        print(obj_function)
        print(optimizer_type)
        print(opt_parameters)
        if optimizer_type!="MCTS" or obj_function!="homology_b0pab1_Dim.4.deg.5_4_5":
            results_for_param_set = []
            num_homologies_found_for_param_set = []
            # print("test")
            
            for seed in range(num_seeds):
                # example : "exp_1_RS_RS_1000_graph_2.1_9_seed_0_scores"
                file_name = "_".join([exp_name, optimizer_type, "_".join(opt_parameters), obj_function, "seed", str(seed), "scores"])
                file_path = os.path.join(local_path, saved_results_folder, obj_function, file_name)
                with open(file_path, 'r') as f:
                    sub_results = np.loadtxt(f,dtype=float,ndmin=2) # important
                    results_for_param_set.append(sub_results)
                # example : "exp_1_RS_RS"
                file_name = "_".join(["homologies", exp_name, optimizer_type, "_".join(opt_parameters), obj_function, "seed", str(seed)])
                file_path = os.path.join(local_path, saved_results_folder, obj_function, "homologies", file_name)
                with open(file_path, 'r') as f:
                    num_homologies_found_for_param_set.append(len(f.readlines()))
            results_for_obj_fun[optimizer_type+"_"+"_".join(opt_parameters)] = ad_hoc_mean(results_for_param_set)
            results_median_for_obj_fun[optimizer_type+"_"+"_".join(opt_parameters)] = ad_hoc_median(results_for_param_set)
            num_homologies_found_for_obj_fun[optimizer_type+"_"+"_".join(opt_parameters)] = np.mean(num_homologies_found_for_param_set)
    results[obj_function] = results_for_obj_fun
    results_median[obj_function] = results_median_for_obj_fun
    num_homologies_found[obj_function] = num_homologies_found_for_obj_fun


print("---------------")

width_plot = ceil(len(obj_functions)/2.0)
fig, axs = plt.subplots(2, width_plot, sharey = False)

for index, obj_function_pair in enumerate(obj_functions):
    obj_function = obj_function_pair[0]
    for index_2, parameters_set in enumerate(parameters_sets):
        optimizer_type = parameters_set[0]
        opt_parameters = parameters_set[1]
        if optimizer_type!="MCTS" or obj_function!="homology_b0pab1_Dim.4.deg.5_4_5":
            parameters_string = optimizer_type+"_"+"_".join(opt_parameters)
            plot_1, plot_2 = int(index/width_plot), index%width_plot
            print(obj_function)
            print(parameters_string)
            axs[plot_1,plot_2].plot(results[obj_function][parameters_string][:,0], results[obj_function][parameters_string][:,1], colors[index_2], label = f"{parameters_string}, hom found : {num_homologies_found[obj_function][parameters_string]}")
            axs[plot_1,plot_2].plot(results_median[obj_function][parameters_string][:,0], results_median[obj_function][parameters_string][:,1], colors[index_2], label = f"{parameters_string}, median", linestyle = "dashed")
            axs[plot_1,plot_2].title.set_text(obj_function)
            axs[plot_1,plot_2].legend()

plt.show()


# colors = ["g","b","r","y"]




for index, performance in enumerate(performances):
    plt.plot(performance[:,0],performance[:,1],colors[index])
#plt.plot([100,300],[10,30])
plt.show()

