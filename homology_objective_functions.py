import numpy as np
import time
import os

from utilities import read_known_homology_profiles

import subprocess


def remove_local_path_from_name(local_path, file_name):
  return file_name.replace(local_path+"/","")

def compute_homology(local_path, temp_files_folder, signs, triangs_input_file, points_input_file, \
	 homology_output_file, parallelization = False):
	"""
	signs can be either a numpy array of 0/1 of shape [batch_size, n_signs], OR the name of a file where such an array is stored
	
	triangs_input_file is the name of a file where there is either a single triangulation, or batch_size triangulations
	
	The paths can either contain or not contain the local_path already
	
	All signs distributions need not be the same length (as they can correspond to different triangulations)
	"""
	# to make sure that all the files are in the same format
	temp_files_folder = os.path.join(local_path, remove_local_path_from_name(local_path, temp_files_folder))
	triangs_input_file = os.path.join(local_path, remove_local_path_from_name(local_path, triangs_input_file))
	points_input_file = os.path.join(local_path, remove_local_path_from_name(local_path, points_input_file))
	homology_output_file = os.path.join(local_path, remove_local_path_from_name(local_path, homology_output_file))

	# Do something about this
	n_parallel = 4

	homology_computation_perl_script = os.path.join(local_path,"Homology_computation","compute_homology.pl")

	# Test if signs is a numpy array, or a file where a numpy array is stored
	# Store the signs in a file if signs is a numpy array
	if not isinstance(signs, str) :
		signs_temp_file = os.path.join(temp_files_folder,'temp_sign_distributions.txt')
		with open(os.path.join(local_path,signs_temp_file), 'w') as f:
			np.savetxt(f,signs,fmt='%d')
		signs = signs_temp_file
	else:
		signs = os.path.join(local_path, remove_local_path_from_name(local_path, signs))
	# Now signs is a file where a numpy array is stored


	if parallelization and n_parallel > 1:
		time_parallelization = time.time()
		# parallelizes the evaluation of the signs distributions
		# Signs
		with open(signs ,'r') as f:
			all_signs = f.readlines()
		batch_size = int(len(all_signs)/(n_parallel-1))
		signs_batches = [all_signs[i*batch_size:min((i+1)*batch_size,len(all_signs))] for i in range(n_parallel)]
		signs_temp_file_root = temp_files_folder+'/temp_sign_distributions_'


		for index, batch in enumerate(signs_batches):
			with open(signs_temp_file_root+str(index)+".txt", 'w') as f:
				f.write("".join(batch))
		# Triang
		triangs = []
		with open(triangs_input_file, 'r') as f:
			for line in f:
				if len(line)>1:
					triangs.append(line)
		triangs_temp_file_root = temp_files_folder+'/temp_triangs_'
		# either it's the same triangulation for all signs distributions, or there is one triangulation for each signs distribution
		if len(triangs) == 1:
			for index in range(len(signs_batches)):
				with open(local_path,triangs_temp_file_root+str(index)+".txt", 'w') as f:
					f.write(triangs[0])
		else:
			triangs_batches = [triangs[i*batch_size:min((i+1)*batch_size,len(triangs))] for i in range(n_parallel)]
			for index, batch in enumerate(triangs_batches):
				with open(triangs_temp_file_root+str(index)+".txt", 'w') as f:
					for index_2, triang in enumerate(batch):
						if index_2 != 0:
							f.write("\n")
						f.write(triang)
		# Writing commands for os.system
		instructions_file = temp_files_folder+"/parallel_commands.txt"
		with open(instructions_file, 'w') as f:
			for index in range(len(signs_batches)):
				if index != 0:
					f.write("\n")
				# files for the scores and homology profiles of the batch
				parallel_homology_output_file = temp_files_folder+"/output_"+str(index)+".txt"
				f.write(" ".join(["polymake","--script",  homology_computation_perl_script, 
				signs_temp_file_root+str(index)+".txt", triangs_temp_file_root+str(index)+".txt", points_input_file, parallel_homology_output_file]) )
				
		# Executing the commands (in parallel)
		os.system("parallel < "+ instructions_file)
		# Put together all the results
		homology_profiles = []
		for index in range(len(signs_batches)):
			with open(temp_files_folder+"/output_"+str(index)+".txt","r") as f:
				homology_profiles += f.readlines()
		with open(homology_output_file,"w") as f:
			f.write("\n".join([homology_profile.replace("\n","") for homology_profile in homology_profiles]))
	else:
		# if no parallelization
		# The part that currently takes very long
		# About 3 seconds to start the polymake script itself (i.e. to get to its first line, even before the "use application tropical"),
		# then the rest (about 30s for simple cases) to execute its content

		list_files = subprocess.run(["polymake","--script",  homology_computation_perl_script, \
			signs, triangs_input_file, points_input_file, homology_output_file])
		
		homology_profiles = []
		with open(homology_output_file, "r") as f:
			homology_profiles += f.readlines()

	homology_profiles_int = []
	for homology_profile in homology_profiles:
		if homology_profile !="\n":
			homology_profile = homology_profile.replace("\n","")
			profile = [int(x) for x in homology_profile.split()]
			homology_profiles_int.append(profile)
	# homology_profiles_int is always a list of lists (possibly of length 1)

	return homology_profiles_int

def update_stored_homologies(signs_input_file, triangs_input_file, homology_profiles, stored_homologies_file, verbose = True):
	"""
	Saves in stored_homology_file the homology profiles from homology_profiles that have not yet been stored, along with the associated triangs and signs

	homology_profiles is a list of lists of int

	stored_homologies_file can represent either the already observed or the already visited profiles

	signs can either be a file where the signs are stored, or a list of lists of integers

	The signs must be in the order corresponding to homology_profiles

	There can either be several triangs (in which case they must also be in that order) or a single triang in triangs_input_file
	"""
	# test if file already exists :
	if not os.path.isfile(stored_homologies_file):
		with open(stored_homologies_file, 'w') as f:
			# create the file and do nothing
			pass
		
	# Set of strings
	known_hom = set(read_known_homology_profiles(stored_homologies_file))

	signs = []
	with open(signs_input_file,'r') as f:
		for line in f:
			signs.append(line)

	triangs = []
	with open(triangs_input_file,'r') as f:
		for line in f:
			if line != "\n":
				triangs.append(line.rstrip("\n"))

	with open(stored_homologies_file, 'a') as f_archive:
		for index, profile in enumerate(homology_profiles):
			profile = " ".join([str(x) for x in profile])
			if profile not in known_hom:
				if verbose:
					print("\nNew homological profile found : "+profile+"\n")
				known_hom.add(profile)
				# case where there is a single triangulation for all the signs distributions
				if len(triangs) ==1 :
					f_archive.write(profile+"|"+signs[index].replace("\n","")+"|"+triangs[0]+"\n")
				# case where there is a triangulation for each signs distribution
				else:
					f_archive.write(profile+"|"+signs[index].replace("\n","")+"|"+triangs[index]+"\n")	
	

def b_total(homology_profiles):
	"""Takes as input a list of lists of int (representing homology profiles), returns a list of ints (representing the sum of all Betti numbers)"""
	return np.array([sum(homology_profile) for homology_profile in homology_profiles])

def b_0(homology_profiles):
	"""Takes as input a list of lists of int (representing homology profiles), returns a list of ints (representing b_0)"""
	return np.array([homology_profile[0] for homology_profile in homology_profiles])

def b_0_p_a_b1(homology_profiles):
	"""Takes as input a list of lists of int (representing homology profiles), returns a list of floats equal to b_0 + 0.1*b_1"""
	return np.array([homology_profile[0] + 0.1*homology_profile[1] for homology_profile in homology_profiles])



# Rmk : we need two types of objective functions : one that takes as input signs and a current_point and computes scores relative to a given current_point and is used by the Discrete_Optimizers
# and one that takes as inputs lists of triangs and signs (and points and points_indices) and outputs scores and is used by the move selectors


def create_objective_function_for_signs_optimization(observed_homologies_file, temp_files_folder, function_of_the_homology_profiles):
	""" Input : various information related to the experiment

		Output : obj_function(current_point, solutions),
		an objective function that takes as input a current_point and signs (a list of lists of 0s and 1s)
		and outputs scores relative to the current_point given as input (suited to being used by the signs_optimizer_for_triang_exploration function)
		obj_function also automatically stores the newly observed homology profiles in observed_homologies_file
		"""

	temp_homologies_file = os.path.join(temp_files_folder,"temp_homologies.txt")
	
	def obj_function(current_point, solutions):
		"""solutions is a list of lists of 0s and 1s representing signs"""
		# Stores the signs in solutions in a text file (needed for update_stored_homologie)
		signs_temp_file = os.path.join(current_point.local_path, temp_files_folder, 'temp_sign_distributions.txt')
		with open(signs_temp_file, 'w') as f:
			np.savetxt(f,solutions,fmt='%d')
		homology_profiles = compute_homology(current_point.local_path, temp_files_folder, signs_temp_file, current_point.triang_file, current_point.all_points_file, temp_homologies_file)
		update_stored_homologies(signs_temp_file, current_point.triang_file, homology_profiles, observed_homologies_file)
		return function_of_the_homology_profiles(homology_profiles).tolist()
	return obj_function



# def create_objective_function_for_move_selector(dim, degree, local_path, list_of_homologies_file, temp_files_folder, polymake_scoring_script):
# 	""" Input : various information related to the experiment

# 		Output : obj_function(triangs_file, signs_file, points_file, points_indices_file), 
# 		an objective function that takes as input triangs and signs files (and points and points_indices) and outputes scores,
# 		suited to being used by a move_selector
# 		"""

# 	def obj_function(triangs_file, signs_file, points_file, points_indices_file):
# 		temp_homologies_file = os.path.join(temp_files_folder,"temp_homologies.txt")
# 		find_new_topologies = True
# 		output_scoring_file = os.path.join(temp_files_folder,'temp_score.txt')	# this is in fact not used (though it must be provided, and the file IS created)

# 		return calc_score(local_path, temp_files_folder, polymake_scoring_script, signs_file,\
# 				triangs_file, points_file, points_indices_file, output_scoring_file,\
# 				degree, dim, find_new_topologies, list_of_homologies_file, temp_homologies_file)
# 	return obj_function



def triangulation_growing_objective_function(triangs_file, signs_file, points_file, points_indices_file):
	"""grows the triangulation  (to be used by a move_selector)"""
	scores =[]
	with open(triangs_file,"r") as f:
		for line in f:
			scores.append(len(line.split("},{")))
	return np.array(scores)


# def create_triangulation_growing_and_look_while_growing_objective_function(degree, dim, local_path, list_of_homologies_file, temp_files_folder):
# 	"""Creates an objective function that grows the triangulation and saves the homologies met along the way while doing so (to be used by a move_selector)"""

# 	# since the scores don't matter, any polymake script will do
# 	polymake_scoring_script = "Scoring/score_b_total.pl"
# 	homology_computing_function = create_objective_function_for_move_selector(degree, dim, local_path, list_of_homologies_file, temp_files_folder, polymake_scoring_script)
# 	def obj_function(triangs_file, signs_file, points_file, points_indices_file):
		
# 		# we compute and record the homology, but don't use it as an objective function
# 		homology_computing_function(triangs_file, signs_file, points_file, points_indices_file)
		
# 		scores =[]
# 		with open(triangs_file,"r") as f:
# 			for line in f:
# 				scores.append(len(line.split("},{")))
# 		return np.array(scores)
# 	return obj_function


#if __name__ == "__main__":
#jitted_calc_score(np.array([1,0,1,0,1,1],dtype=float))
