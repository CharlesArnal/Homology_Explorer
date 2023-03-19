import os

import subprocess

import random
import numpy as np

from CRT import CRT


random.seed(1)
np.random.seed(1)




opti_program = "RL_v3.py"
#opti_program = "ES_2.py"
#opti_program = "Tabu_search.py"
#opti_program = "MCTS.py"


FIND_NEW_TOPOLOGIES = True
START_WITH_RANDOM_TRIANGULATION = True
VISUALIZE_SUBDIVISION = False
DEGREE = 8  # Degree of variety
DIMENSION = 3 # Ambiant dimension

NAME_EXP = f"dim_{DIMENSION}_deg_{DEGREE}_signs_only"

# for Linux
LOCAL_PATH = '/home/charles/Desktop/ML_RAG/Code'
TEMP_FILES_FOLDER = "General_test_temp_files"

OUTPUT_SCORING_FILE = TEMP_FILES_FOLDER+'/temp_score.txt'
POLYMAKE_SCORING_SCRIPT = "Scoring/score_b_total.pl"

STOPPING_OBJ_VALUE = 10000
MAX_RUNNING_TIME = 100000


LIST_OF_HOMOLOGIES_FILE = "Saved_files/homologies_"+NAME_EXP+".txt"
TEMP_HOMOLOGIES_FILE = TEMP_FILES_FOLDER+"/temp_homologies_file.txt"
SAVE_PERF_FILE =  "Saved_files/perf_wrt_time_"+NAME_EXP+".txt"
STARTING_SIGNS_DISTRIBUTIONS_FILE = "None"
OUTPUT_FILE = "Saved_files/best_solution.dat"
SAVE_PERIOD = 20


# stored as a triangulation and a list of points - the points need to be the vertices of the triangulation
TRIANGULATION_INPUT_FILE = TEMP_FILES_FOLDER+"/current_triang.dat"
POINTS_INPUT_FILE = TEMP_FILES_FOLDER+"/all_points.dat"
RELEVANT_POINTS_INDICES_INPUT_FILE = TEMP_FILES_FOLDER+"/relevant_points_indices.dat"
#N_SIGNS = comb(DEGREE+DIMENSION,DIMENSION)  #The length of the word we are generating.

general_arguments = [str(DEGREE), str(DIMENSION), str(STOPPING_OBJ_VALUE), str(MAX_RUNNING_TIME), LOCAL_PATH, TEMP_FILES_FOLDER, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
	TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE, STARTING_SIGNS_DISTRIBUTIONS_FILE, \
	TEMP_HOMOLOGIES_FILE,  str(FIND_NEW_TOPOLOGIES), LIST_OF_HOMOLOGIES_FILE, SAVE_PERF_FILE, str(SAVE_PERIOD),OUTPUT_FILE]


if opti_program == "RL_v3.py":
	width_layers = [128,64,4]
	n_layers = len(width_layers)
	# TODO I have multiplied the learning rate by 2
	learning_rate = 0.0002 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
	n_sessions =1000 #number of new sessions per iteration
	percentile = 93 #top 100-X percentile we are learning from
	super_percentile = 94 #top 100-X percentile that survives to next iteration
	min_randomness = False
	alpha = 0.5
	program_specific_arguments = [n_layers, *width_layers,  learning_rate, n_sessions, percentile, super_percentile, \
									min_randomness, alpha]
	program_specific_arguments = [str(i) for i in program_specific_arguments]



if opti_program == "ES_2.py":
	num_generations=1000
	num_parents_mating=200 # number of parents mating to create the next generation
	sol_per_pop=400		# number of elements in each generation
	parent_selection_type = "sss" # "tournament"
	program_specific_arguments = [num_parents_mating, sol_per_pop, num_generations,parent_selection_type]
	program_specific_arguments = [str(i) for i in program_specific_arguments]

	
if opti_program == "Tabu_search.py":
	size_pop = 30
	n_iter = 1000
	STM_length = 1000
	MTM_length = 3
	SI_thresh = 60
	SD_thresh = 300
	feedback_frequency = 1
	n_best_solutions_to_display = 10
	more_exploration = False
	percent_more_exploration = [0.1,0.1,0.1,0.1]
	n_solutions_saved = 10
	saved_solutions_file = "Saved_files/saved_solutions_test.txt"
	
	program_specific_arguments = [size_pop, n_iter, STM_length,	MTM_length, SI_thresh, SD_thresh,\
		feedback_frequency,n_best_solutions_to_display,more_exploration,*percent_more_exploration,\
									n_solutions_saved,saved_solutions_file]
	program_specific_arguments = [str(i) for i in program_specific_arguments]

if opti_program == "MCTS.py":
	depth = 5
	n_MCR = 10
	feedback_frequency = 1
	n_iter = 1
	n_solutions_saved = 0
	saved_solutions_file = "Saved_files/saved_solutions_test.txt"
	
	program_specific_arguments = [depth,n_MCR,feedback_frequency,n_iter,\
									n_solutions_saved,saved_solutions_file]
	program_specific_arguments = [str(i) for i in program_specific_arguments]





if START_WITH_RANDOM_TRIANGULATION:
	print(f"\nCreating random triangulation in dimension {DIMENSION} and degree {DEGREE}.")
	CRT(DIMENSION, DEGREE, LOCAL_PATH, TEMP_FILES_FOLDER, TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE)

	
# TODO : code properly (this is a quick fix)
if VISUALIZE_SUBDIVISION:
	print("Visualizing subdivision\n")
	list_files = subprocess.run(["polymake","--script", os.path.join(LOCAL_PATH,"visualize_results.pl")])


list_files = subprocess.run(["python3", os.path.join(LOCAL_PATH,opti_program),*program_specific_arguments,*general_arguments])
