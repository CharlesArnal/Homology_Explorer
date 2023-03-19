import matplotlib.pyplot as plt
import sys
import os 
import numpy as np

# Doesn't work in WSL
"""
# For Linux
local_path = '/mnt/c/Users/CombinatorialRL/Code/Saved_files'
perf_archive_file = sys.argv[1]
"""

local_path = r"C:\Users\CombinatorialRL\Code\Graphs\Saved_files_graphs"
file1 = "perf_wrt_time_Conj_2_3_Tabu_search_graphs.py_conf1.txt"
file2 = "perf_wrt_time_Conj_2_3_RL_graphs.py_conf1.txt"  
file3 = "perf_wrt_time_Conj_2_1_Tabu_search_graphs.py_conf1.txt"
perf_archive_files = [file3]#+[file1]+[file2]
colors = ["g","b","r"]

performances = []
for file in perf_archive_files:
    with open(os.path.join(local_path, file), 'r') as f:
        performances.append(np.loadtxt(f,dtype=float))
        


for index, performance in enumerate(performances):
    plt.plot(performance[:,0],performance[:,1],colors[index])
#plt.plot([100,300],[10,30])
plt.show()