import os


import subprocess

import sys 
# TODO adapt all files names to server context

# For Linux
LOCAL_PATH = '../'
#LOCAL_PATH = '/mnt/c/Users/CombinatorialRL/Code'
sys.path.append(LOCAL_PATH)


FUNCTIONS = ["Rastrigin"]#,"Harnack_10","Harnack_15","random_dim3_d5","random_dim3_d10"]
DIMENSIONS = [5,10,15]
algorithms =  [ "TS.py"]#,"ES_2.py"]
hyperparameters = ["conf1"]#["conf1", "conf2"]
# TODO complete
MAX_RUNNING_TIME = {5: 14400, 10 : 28800 , 15: 43600 }
#MAX_RUNNING_TIME = {5: 10, 10 : 10 , 15: 10 }



MAX_AMPLITUDE = {"Rana":500,"Rastrigin": 5.12, "Ackley":32.768, "Rosenbrock": 6}

STOPPING_OBJ_VALUE={"Rana":100000000, "Rastrigin":100000000, "Ackley":100000000, "Rosenbrock": 1000000000}

SAVE_PERIOD = 20



program_specific_arguments = dict()


# RL

learning_rate = 0.0002 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 10000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
min_randomness = False
alpha = 0.5
true_dim = 5
precision = 9
amplitude = 500

width_layers = [128,64, 4]
width_layers = [256, 128, 64, 8]
n_layers = len(width_layers)
program_specific_arguments[("RL.py","conf1")] = [str(i) for i in [n_layers, *width_layers, learning_rate, n_sessions, percentile, super_percentile, \
								min_randomness, alpha, precision] ]


width_layers = [256, 128, 64, 8]
n_layers = len(width_layers)
program_specific_arguments[("RL.py","conf2")] = [str(i) for i in [n_layers, *width_layers, learning_rate, n_sessions, percentile, super_percentile, \
								min_randomness, alpha, precision] ]

# ES

num_generations=100000
num_parents_mating=50
sol_per_pop=100

parent_selection_type = "sss"
program_specific_arguments[("ES_2.py","conf1")] = [str(i) for i in [ num_parents_mating, sol_per_pop, num_generations, parent_selection_type]]
parent_selection_type = "tournament"
program_specific_arguments[("ES_2.py","conf2")] = [str(i) for i in [num_parents_mating, sol_per_pop, num_generations, parent_selection_type]]


# TS	

n_iter = 1000000000
STM_length = 2000
MTM_length = 4
SI_thresh = 30
SD_thresh = 100
step_reduction_thresh = 100
feedback_frequency = 30
n_best_solutions_to_display = 10
initial_step = 50
n_solutions_saved = 0
saved_solutions_file = "none"

size_pop = 1000
more_exploration = True
percent_more_exploration = [0.1,0.1,0.1,0.1]
program_specific_arguments[("TS.py","conf1")] = [str(i) for i in [size_pop, n_iter, STM_length,\
							MTM_length, initial_step, step_reduction_thresh, SI_thresh, SD_thresh, feedback_frequency,\
								n_best_solutions_to_display,more_exploration,*percent_more_exploration, n_solutions_saved,saved_solutions_file]]

size_pop = 200
more_exploration = False
percent_more_exploration = []
program_specific_arguments[("TS.py","conf2")] = [str(i) for i in [size_pop, n_iter, STM_length,\
							MTM_length, initial_step, step_reduction_thresh, SI_thresh, SD_thresh,feedback_frequency,\
								n_best_solutions_to_display,more_exploration,*percent_more_exploration, n_solutions_saved,saved_solutions_file]]




# RandomSolver

batch_size = 10000
feedback_frequency = 10
n_solutions_saved = 0
saved_solutions_file = "none"

program_specific_arguments[("RandomSolver.py","conf1")] = [str(i) for i in [batch_size,feedback_frequency, n_solutions_saved,saved_solutions_file] ]



for function in FUNCTIONS:
	for dimension in DIMENSIONS:
		for algorithm in algorithms:
			for hyperparameter_set in hyperparameters:
				for seed in range(3):
					SAVE_PERF_FILE = f"Saved_files/perf_wrt_time_{function}_{dimension}_{algorithm}_{hyperparameter_set}_{seed}.txt"
					general_arguments = [function ,str(dimension), str(seed), str(MAX_AMPLITUDE[function]), str(STOPPING_OBJ_VALUE[function]),\
						str(MAX_RUNNING_TIME[dimension]), os.path.join(LOCAL_PATH,"Optimization"), SAVE_PERF_FILE, str(SAVE_PERIOD) ]
					list_files = subprocess.run(["python3", os.path.join(os.path.join(LOCAL_PATH,"Optimization"),algorithm),*program_specific_arguments[(algorithm,hyperparameter_set)],*general_arguments])




		


