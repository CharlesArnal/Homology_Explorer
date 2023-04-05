
import random
import numpy as np
import time
from math import comb, sqrt
import os
import copy

from homology_objective_functions import compute_homology, update_stored_homologies


def create_move_selector(index_selector, objective_function, must_compute_homology = True, \
			 objective_function_takes_homology_as_input = True, observed_homologies_file = None, visited_homologies_file = None):
	"""Returns a function  generic_move_manager(possible_moves, all_points_file, degree, dim) that returns selected move, current_value and move_selection_feedback_info
	 
		The index_selector can be one of the many defined in the move_selectors.py file

		The objective_function can be None ONLY if index_selector = Random_Triang_Selector (and we are not storing homology while moving randomly)

		If objective_function_takes_homology_as_input is True, objective_function takes as input a list of lists of integers
		Otherwise, it takes as input triangs_file, signs_file, points_file

		If must_compute_homology is True, updates observed_homologies_file and visited_homologies_file
	   """
	if objective_function_takes_homology_as_input :
		must_compute_homology = True
	# define the correct move selector)
	def generic_move_manager(possible_moves, all_points_file, degree, dim):
		# affine rank
		rank = dim +1 
		# number of points 
		n_points = comb(degree+dim,degree)
		# in this case, possible_moves contains nb_triangs_file_path, nb_flips_file_path, nb_signs_file_path and local_path
		possible_moves_triangs_file_path, possible_moves_flips_file_path, possible_moves_signs_file_path,\
			possible_moves_relevant_points_indices_file_path, local_path, temp_files_folder = possible_moves
		selected_triang_file_path = os.path.join(local_path,temp_files_folder,"next_triang.dat")
		selected_flip_file_path =  os.path.join(local_path,temp_files_folder,"selected_flip.dat")
		selected_signs_file_path =  os.path.join(local_path,temp_files_folder,"selected_signs.dat")
		selected_relevant_points_indices_file_path = os.path.join(local_path,temp_files_folder,"selected_points_indices.dat")
	

		triangs =[]
		flips = []
		relevant_indices = []
		signs = []
		with open(possible_moves_triangs_file_path, 'r') as f:
			for line in f:
				if len(line)>1:
					triangs.append(line)

		with open(possible_moves_relevant_points_indices_file_path, 'r') as f:
			for line in f:
				if len(line)>1:
					relevant_indices.append(line.replace("\n",""))

		with open(possible_moves_flips_file_path, 'r') as f:
			for line in f:
				if len(line)>1:
					flips.append(line)

		with open(possible_moves_signs_file_path, 'r') as f:
			for line in f:
				# they are simply strings
				signs.append(line)

		# -------------------
		# the key operation
		time1 = time.time()
		if must_compute_homology :
			temp_homologies_file = os.path.join(temp_files_folder,"temp_homologies.txt")
			homology_profiles = compute_homology(local_path, temp_files_folder, possible_moves_signs_file_path, possible_moves_triangs_file_path, all_points_file, \
	 			temp_homologies_file)
			# update the list of observed homologies
			if observed_homologies_file != None:
				update_stored_homologies(possible_moves_signs_file_path, possible_moves_triangs_file_path, homology_profiles, observed_homologies_file, verbose = True)
		if index_selector == Random_Triang_Selector:
			scores = None
			current_value = None
		elif objective_function_takes_homology_as_input:
			scores = objective_function(homology_profiles)
		else :
			scores = objective_function(possible_moves_triangs_file_path, possible_moves_signs_file_path, all_points_file, possible_moves_relevant_points_indices_file_path)

		# scores is either None (if using Random_Selector) or a numpy array (at this point)
		selected_index = index_selector(scores, triangs, signs)
		if index_selector != Random_Triang_Selector:
			current_value = scores[selected_index]
			scores = scores.tolist()
		selection_time = time.time()-time1
		# -------------------
		with open(selected_triang_file_path, 'w') as f:
			f.write(triangs[selected_index])
		with open(selected_relevant_points_indices_file_path, 'w') as f:
			f.write(relevant_indices[selected_index])
		with open(selected_flip_file_path, 'w') as f:
			# this format is needed for the C++ program to correctly read the files 
			f.write(f"[{n_points},{rank}:["+flips[selected_index][:-1]+"->0]]")
		with open(selected_signs_file_path, 'w') as f:
			f.write(signs[selected_index])

		if must_compute_homology and visited_homologies_file != None:
			# update the list of visited homologies
			update_stored_homologies(selected_signs_file_path, selected_triang_file_path, [homology_profiles[selected_index]], visited_homologies_file, verbose = False)

		move_selection_feedback_info = {"selection time": selection_time, "selected triang": triangs[selected_index].replace("\n",""), "selected signs": signs[selected_index].replace("\n",""),\
				  "selected flips": flips[selected_index][:-1], "scores": scores, "selected homology": (None if must_compute_homology == False else homology_profiles[selected_index] )}
		return (selected_triang_file_path,selected_relevant_points_indices_file_path, selected_flip_file_path,selected_signs_file_path, local_path, temp_files_folder), current_value, move_selection_feedback_info

	return generic_move_manager


# when selecting greedily, we randomly select among the best answers to minimize the risk of getting stuck in a loop
def random_best_score(scores):
	best_score = np.max(scores)
	best_scores_indices = [index for index, score in enumerate(scores) if score == best_score]
	i = random.randint(0,len(best_scores_indices)-1)
	selected_index = best_scores_indices[i]
	return selected_index


def Greedy_Selector(scores, triangs, signs):
	# scores is a numpy array
	# a randomly chosen index among those that achieve the best score
	selected_index = random_best_score(scores)
	return selected_index

def Greedy_Randomized_Selector(scores, triangs, signs):
	# scores is a numpy array
	if np.random.uniform()>0.9:
		selected_index = random.randint(0,len(scores)-1)
		print("A random choice was made")
	else:
		# a randomly chosen index among those that achieve the best score
		selected_index = random_best_score(scores)
	return selected_index

def Greedy_Expanding_Selector(scores, triangs, signs):
	# scores is a numpy array
	scores_2 = copy.deepcopy(scores)
	for index, line in enumerate(triangs):
		scores_2[index] += sqrt(len(line.split("},{")))	
	selected_index = random_best_score(scores_2)
	return selected_index

def Greedy_Randomized_Expanding_Selector(scores, triangs, signs):
	# scores is a numpy array
	scores_2 = copy.deepcopy(scores)
	for index, line in enumerate(triangs):
		scores_2[index] += sqrt(len(line.split("},{")))
	if np.random.uniform()>0.9:
		selected_index = random.randint(0,len(triangs)-1)
		print("A random choice was made")
	else:
		# a randomly chosen index among those that achieve the best score
		selected_index = random_best_score(scores_2)
	return selected_index

# An index selector
def Random_Triang_Selector(scores, triangs, signs):
	selected_index = random.randint(0,len(triangs)-1)
	return selected_index