
import random
import numpy as np
import time
from math import comb, sqrt
import os


def create_move_selector(index_selector, objective_function):
	"""Returns a function  generic_move_manager(possible_moves, all_points_file, degree, dim) that returns selected move, current_value and move_selection_feedback_info
	 
		The index_selector can be one of the many defined in the move_selectors.py file

		The objective_function can be None ONLY if index_selector = Random_Triang_Selector (and we are not storing homology while moving randomly)
	   """
	# define the correct move selector
	def generic_move_manager(possible_moves, all_points_file, degree, dim):
		# affine rank
		rank = dim +1 
		# number of points 
		n_points = comb(degree+dim,degree)
		# in this case, possible_moves contains nb_triangs_file_path, nb_flips_file_path, nb_signs_file_path and local_path
		possible_moves_triangs_file_path, possible_moves_flips_file_path, possible_moves_signs_file_path,\
			possible_moves_relevant_points_indices_file_path, local_path, temp_files_folder = possible_moves
		selected_triang_file_path = os.path.join(local_path,os.path.join(temp_files_folder,"next_triang.dat"))
		selected_flip_file_path =  os.path.join(local_path,os.path.join(temp_files_folder,"selected_flip.dat"))
		selected_signs_file_path =  os.path.join(local_path,os.path.join(temp_files_folder,"selected_signs.dat"))
		selected_relevant_points_indices_file_path = os.path.join(local_path,os.path.join(temp_files_folder,"selected_points_indices.dat"))
	

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
		# scores is either None (if using Random_Selector) or a list of floats. It is only used for feedback purposes
		selected_index, current_value, scores = index_selector(possible_moves_triangs_file_path,\
			possible_moves_signs_file_path, all_points_file, possible_moves_relevant_points_indices_file_path,  objective_function, triangs, signs)
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

		move_selection_feedback_info = (selection_time, triangs[selected_index].replace("\n",""), signs[selected_index].replace("\n",""), flips[selected_index][:-1], scores)
		return (selected_triang_file_path,selected_relevant_points_indices_file_path, selected_flip_file_path,selected_signs_file_path, local_path, temp_files_folder), current_value, move_selection_feedback_info

	return generic_move_manager


# when selecting greedily, we randomly select among the best answers to minimize the risk of getting stuck in a loop
def random_best_score(scores):
	best_score = np.max(scores)
	best_scores_indices = [index for index, score in enumerate(scores) if score == best_score]
	i = random.randint(0,len(best_scores_indices)-1)
	selected_index = best_scores_indices[i]
	return selected_index


def Greedy_Selector(triangs_file,signs_file,points_file,points_indices_file, objective_function, triangs, signs):
	# a numpy array
	scores  = objective_function(triangs_file, signs_file, points_file, points_indices_file)
	# a randomly chosen index among those that achieve the best score
	selected_index = random_best_score(scores)
	return selected_index, scores[selected_index], scores.tolist()

def Greedy_Randomized_Selector(triangs_file,signs_file,points_file,points_indices_file,objective_function, triangs, signs):
	# a numpy array
	scores  = objective_function(triangs_file,signs_file,points_file, points_indices_file)
	if np.random.uniform()>0.9:
		selected_index = random.randint(0,len(triangs)-1)
		print("A random choice was made")
	else:
		# a randomly chosen index among those that achieve the best score
		selected_index = random_best_score(scores)
	return selected_index, scores[selected_index], scores.tolist()

def Greedy_Expanding_Selector(triangs_file,signs_file,points_file,points_indices_file,objective_function, triangs, signs):
	# a numpy array
	scores  = objective_function(triangs_file,signs_file,points_file, points_indices_file)
	with open(triangs_file,"r") as f:
		for index, line in enumerate(f):
			scores[index] += sqrt(len(line.split("},{")))	
	selected_index = random_best_score(scores)
	return selected_index, scores[selected_index], scores.tolist()

def Greedy_Randomized_Expanding_Selector(triangs_file,signs_file,points_file,points_indices_file,objective_function, triangs, signs):
	# a numpy array
	scores  = objective_function(triangs_file,signs_file,points_file, points_indices_file)
	with open(triangs_file,"r") as f:
		for index, line in enumerate(f):
			scores[index] += sqrt(len(line.split("},{")))
	if np.random.uniform()>0.9:
		selected_index = random.randint(0,len(triangs)-1)
		print("A random choice was made")
	else:
		# a randomly chosen index among those that achieve the best score
		selected_index = random_best_score(scores)
	return selected_index, scores[selected_index], scores.tolist()

# An index selector
def Random_Triang_Selector(triangs_file,signs_file, points_file,points_indices_file, objective_function, triangs, signs):
	if objective_function == None:
		selected_index = random.randint(0,len(triangs)-1)
		return selected_index, 0, None
	else:
		# the scores are not used, but computing them can be needed if look_while_growing == True in some of Homology_Explorer's methods
		scores  = objective_function(triangs_file,signs_file,points_file, points_indices_file)
		selected_index = random.randint(0,len(triangs)-1)
		return selected_index, 0, None