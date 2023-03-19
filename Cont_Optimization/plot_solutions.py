import matplotlib.pyplot as plt
import sys
import os 
import numpy as np


local_path = r"C:\Users\CombinatorialRL\Code\Optimization"
saved_solutions_file = os.path.join(local_path,"TS_Rana_solutions.txt")

with open(saved_solutions_file, 'r') as f:
    solutions = np.loadtxt(f,dtype=float)
    plt.scatter(solutions[:,0],solutions[:,1],marker='o')
    plt.show()
