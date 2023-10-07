
import random
import numpy as np

from math import comb, floor
import matplotlib.pyplot as plt
import time
import os
import sys


from utilities import read_first_experiment_param_file

from Current_Point import Current_Point
from Homology_Explorer import Homology_Explorer
from homology_objective_functions import b_0, b_0_p_a_b1, b_total, compute_homology


n_confs = 22
n_parallel = 2
lower_batch_size = 	int(n_confs/n_parallel)
remainder = n_confs%n_parallel
batches_bounds = [0]
for i in range(n_parallel):
	batches_bounds += [batches_bounds[-1] + lower_batch_size + (1 if i<remainder else 0)]

print(batches_bounds)