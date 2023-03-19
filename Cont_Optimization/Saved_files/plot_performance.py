
import matplotlib.pyplot as plt
import sys
import os 
import numpy as np

# Doesn't work in WSL



def ad_hoc_mean(mylist):
    min_length = min(len(my_array) for my_array in mylist)
    mylist = np.array([my_array[:min_length,:] for my_array in mylist])
    return np.mean(mylist,axis = 0)

local_path = r"/home/charles/Desktop/ML_RAG/Code/Optimization/Saved_files/"

functions = ["Rana","Rastrigin", "SineEnv"]
dimensions = ["5", "10", "15","25", "50"]
algorithms =  [ "TS.py", "RandomSolver.py","RL.py"]
hyperparameters = ["conf1"]

seeds = ["0","1","2"]


function = functions[2]
dimension = dimensions[2]
task = function+"_"+dimension


performances = []

for algorithm in algorithms:
    for hyperparameter_set in hyperparameters:
        sub_results = []
        for seed in seeds:
            file = f"perf_wrt_time_{task}_{algorithm}_{hyperparameter_set}_{seed}.txt"
            with open(os.path.join(local_path, file), 'r') as f:
                sub_results.append(np.loadtxt(f,dtype=float))
        performances.append(ad_hoc_mean(sub_results))


colors = ["g","b","r","y"]


# rmk : wrong labels if several hyperparameter sets
fig = plt.figure(task)
for index, performance in enumerate(performances):
    plt.plot(performance[:,0],performance[:,1],colors[index+1], label = algorithms[index])
leg = plt.legend()
plt.title(f"Function : {function}, dimension : {dimension}")
#plt.plot([100,300],[10,30])
plt.show()

"""
with open(os.path.join(local_path, f"perf_wrt_time_Rosenbrock_5_RL.py_{hyperparameter_set}_{0}.txt"), 'r') as f:
    perf = np.loadtxt(f,dtype=float)
    plt.plot(perf[:,0],perf[:,1])
    plt.show()
"""