
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from sklearn.model_selection import ParameterGrid

my_local_path =  parent_dir


param_file = os.path.join(my_local_path, "Parameters_files","param_exp_all_methods_triangs.txt")

exp_name = "exp_all_methods_triangs"

saved_results_folder = exp_name

if not os.path.isdir(os.path.join(my_local_path, saved_results_folder)):
    os.mkdir(os.path.join(my_local_path, saved_results_folder))


# correct parameters
param_grid = [{"optimizer_type" : ["RS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),(f"homology_bt_Harnack.25_2_25", f"{351}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["100"]] 
            },
             {"optimizer_type" : ["RL"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),(f"homology_bt_Harnack.25_2_25", f"{351}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["5", "256", "256", "256", "128", "32", "0.001", "500", "True", "0.1"]]
            },
             {"optimizer_type" : ["GA"],
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),(f"homology_bt_Harnack.25_2_25", f"{351}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["100","300", "rws"]]
            },
             {"optimizer_type" : ["TS"],
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),(f"homology_bt_Harnack.25_2_25", f"{351}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["10","3000", "3", "100", "200", "False"]]
            },
             {"optimizer_type" : ["MCTS"],
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_bt_Harnack.10_2_10", f"{66}"),(f"homology_bt_Harnack.15_2_15", f"{136}"),(f"homology_bt_Harnack.20_2_20", f"{231}"),(f"homology_bt_Harnack.25_2_25", f"{351}")],
              "max_running_time": [f"{3600*10}"],
              "optimizer_parameters": [["5","10"]]
            },#---------
            {"optimizer_type" : ["RS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["100"]]
             },
             {"optimizer_type" : ["RL"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["5", "256", "256", "256", "128", "32", "0.001", "500", "True", "0.1"]]
             },
             {"optimizer_type" : ["GA"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["100","300", "rws"]]
             },
            {"optimizer_type" : ["TS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["10","3000", "3", "100", "200", "False"]]
             },
            {"optimizer_type" : ["MCTS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.6_3_6", f"{78}"),(f"homology_b0pab1_Dim.4.deg.4_4_4", f"{51}") ],
              "max_running_time": [f"{3600*2}"],
              "optimizer_parameters": [["5","10"]]
            },#-----------            
            {"optimizer_type" : ["RS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ],
              "max_running_time": [f"{3600*4}"],
              "optimizer_parameters": [["100"]]
             },
             {"optimizer_type" : ["RL"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ],
              "max_running_time": [f"{3600*4}"],
              "optimizer_parameters": [["5", "256", "256", "256", "128", "32", "0.001", "500", "True", "0.1"]]
             },
             {"optimizer_type" : ["GA"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ],
              "max_running_time": [f"{3600*4}"],
              "optimizer_parameters": [["100","300", "rws"]]
             },
            {"optimizer_type" : ["TS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ],
              "max_running_time": [f"{3600*4}"],
              "optimizer_parameters": [["10","3000", "3", "100", "200", "False"]]
             },
            {"optimizer_type" : ["MCTS"], 
              "num_seeds" : ["8"],
              "obj_function": [(f"homology_b0pab1_Dim.3.deg.8_3_8", f"{98}"),(f"homology_b0pab1_Dim.4.deg.5_4_5", f"{126}") ],
              "max_running_time": [f"{3600*4}"],
              "optimizer_parameters": [["5","10"]]
            }
            ]


param_list = ParameterGrid(param_grid) 

with open(param_file, "w") as f:
    f.write(exp_name)
    f.write("\n")
    # TODO change depending on needs
    # batch decomposition
    # 40 in total, no more than 12 at a time (better slowly and surely)
    f.write(str([list(range(2*i+1,2*i+3)) for i in range(10)]+[list(range(21,31)),list(range(31,36)),list(range(36,41))]))
    
    f.write("\n")
    f.write(my_local_path + " " +saved_results_folder)
    f.write("\n")
    f.write("exp_num num_seeds dim obj_function max_running_time optimizer_type specific_parameters")
    f.write("\n")
    for index, prm_set in enumerate(param_list):
        f.write(f"{index+1} {prm_set['num_seeds']} {prm_set['obj_function'][1]} {prm_set['obj_function'][0]} {prm_set['max_running_time']} {prm_set['optimizer_type']} " + " ".join(prm_set['optimizer_parameters']))
        f.write("\n")
   