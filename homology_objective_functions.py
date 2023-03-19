import numpy as np
import time
import os


from Current_Point import Current_Point

import subprocess





def remove_local_path_from_name(local_path, file_name):
  return file_name.replace(local_path+"/","")

def calc_score(local_path, temp_files_folder, scoring_script, signs, triangs_input_file, points_input_file, relevant_points_indices_input_file,\
	 score_output_file, degree, dimension, find_new_topologies = False, list_of_homologies_file = "", temp_homologies_file = ""):
	"""
	signs can be either a numpy array of 0/1 of shape [batch_size, n_signs], OR the name of a file where such an array is stored
	
	triangs_input_file is the name of a file where there is either a single triangulation, or batch_size triangulations
	
	Also saves new topologies encountered along the way
	
	The paths can either contain or not contain the local_path already
	
	All signs distributions need not be the same length (as they can correspond to different triangulations)
	"""
	# to make sure that all the files are in the same format
	temp_files_folder = remove_local_path_from_name(local_path, temp_files_folder)
	triangs_input_file = remove_local_path_from_name(local_path, triangs_input_file)
	points_input_file = remove_local_path_from_name(local_path, points_input_file)
	relevant_points_indices_input_file = remove_local_path_from_name(local_path, relevant_points_indices_input_file)
	score_output_file = remove_local_path_from_name(local_path, score_output_file)
	list_of_homologies_file = remove_local_path_from_name(local_path, list_of_homologies_file)
	temp_homologies_file = remove_local_path_from_name(local_path, temp_homologies_file)

		

	# TODO do something about that
	parallelization = False
	n_parallel = 4

	# remove the local_path from the other paths if it is already inside

	# Test if signs is a numpy array, or a file where a numpy array is stored
	if not isinstance(signs, str) :
		signs_temp_file = temp_files_folder+'/temp_sign_distributions.txt'
		with open(os.path.join(local_path,signs_temp_file), 'w') as f:
			np.savetxt(f,signs,fmt='%d')
		signs = signs_temp_file
	else:
		signs = remove_local_path_from_name(local_path, signs)
	# Now signs is a file where a numpy array is stored

	if parallelization:
		time_parallelization = time.time()
		# parallelizes the evaluation of the signs distributions
		# Signs
		with open(os.path.join(local_path,signs), 'r') as f:
			all_signs = f.readlines()
		batch_size = int(len(all_signs)/n_parallel)
		signs_batches = [all_signs[i*batch_size:min((i+1)*batch_size,len(all_signs))] for i in range(n_parallel+1)]
		signs_temp_file_root = temp_files_folder+'/temp_sign_distributions_'


		for index, batch in enumerate(signs_batches):
			with open(os.path.join(local_path,signs_temp_file_root+str(index)+".txt"), 'w') as f:
				f.write("".join(batch))
		# Triang
		triangs = []
		with open(os.path.join(local_path,triangs_input_file), 'r') as f:
			for line in f:
				if len(line)>1:
					triangs.append(line)
		triangs_temp_file_root = temp_files_folder+'/temp_triangs_'
		# either it's the same triangulation for all signs distributions, or there is one triangulation for each signs distribution
		if len(triangs) == 1:
			for index in range(len(signs_batches)):
				with open(os.path.join(local_path,triangs_temp_file_root+str(index)+".txt"), 'w') as f:
					f.write(triangs[0])
		else:
			triangs_batches = [triangs[i*batch_size:min((i+1)*batch_size,len(triangs))] for i in range(n_parallel+1)]
			for index, batch in enumerate(triangs_batches):
				with open(os.path.join(local_path,triangs_temp_file_root+str(index)+".txt"), 'w') as f:
					for index_2, triang in enumerate(batch):
						if index_2 != 0:
							f.write("\n")
						f.write(triang)
		# NOTE : We don't do anything with the relevant_points_indices, are they are not currently being used by the scoring scripts
		# Writing commands for os.system
		instructions_file = temp_files_folder+"/parallel_commands.txt"
		with open(os.path.join(local_path,instructions_file), 'w') as f:
			for index in range(len(signs_batches)):
				if index != 0:
					f.write("\n")
				# files for the scores and homology profiles of the batch
				parallel_score_output_file = temp_files_folder+"/output_"+str(index)+".txt"
				parallel_homology_profiles_file = temp_files_folder+"/temp_homologies_file_"+str(index)+".txt"
				f.write(" ".join(["polymake","--script",  os.path.join(local_path,scoring_script),local_path,\
				signs_temp_file_root+str(index)+".txt", triangs_temp_file_root+str(index)+".txt", points_input_file,\
					 relevant_points_indices_input_file, parallel_score_output_file, str(find_new_topologies), parallel_homology_profiles_file]) )
		# Executing the commands (in parallel)
		os.system("parallel < "+os.path.join(local_path,instructions_file))
		# Put together all the results
		scores = []
		homology_profiles = []
		for index in range(len(signs_batches)):
			with open(os.path.join(local_path,temp_files_folder+"/output_"+str(index)+".txt"),"r") as f:
				scores += f.readlines()
			with open(os.path.join(local_path,temp_files_folder+"/temp_homologies_file_"+str(index)+".txt"),"r") as f:
				homology_profiles += f.readlines()
		with open(os.path.join(local_path,score_output_file),"w") as f:
			f.write("\n".join([score.replace("\n","") for score in scores]))
		with open(os.path.join(local_path,temp_homologies_file),"w") as f:
			f.write("\n".join([homology_profile.replace("\n","") for homology_profile in homology_profiles]))
	else:
		# if no parallelization
		# The part that currently takes very long
		# About 3 seconds to start the polymake script itself (i.e. to get to its first line, even before the "use application tropical"),
		# then the rest (about 30s for simple cases) to execute its content

		list_files = subprocess.run(["polymake","--script",  os.path.join(local_path,scoring_script), local_path,\
			signs, triangs_input_file, points_input_file, relevant_points_indices_input_file, score_output_file, str(find_new_topologies), temp_homologies_file])
		

	# Identifying and saving new homological profiles
	# the profiles are saved in the form "b0 b1 ... bn|0 1 1 0 0 ... 1|triangulation"
	# the second part being a collection of signs giving rise to the profile
	# the third part being a triangulation of the simplex
	if find_new_topologies:
		find_and_save_new_hom(os.path.join(local_path,signs), os.path.join(local_path,triangs_input_file), \
			os.path.join(local_path,temp_homologies_file),os.path.join(local_path,list_of_homologies_file))


	# The scores
	with open(os.path.join(local_path,score_output_file), 'r') as f:
		scores =  np.loadtxt(f,dtype=float)
		# make sure that an array is returned (as opposed to a single float) when scores only contains one element
		if len(np.shape(scores)) == 0 :
			scores = np.array([scores])
		f.close()
		return scores


	return None


# Rmk : we need to types of objective functions : one that takes as input signs and a current_point and computes scores relative to a given current_point and is used by the Discrete_Optimizers
# and one that takes as inputs lists of triangs and signs (and points and points_indices) and outputs scores and is used by the move selectors


def create_objective_function_for_signs_optimization(list_of_homologies_file, temp_files_folder, polymake_scoring_script):
	""" Input : various information related to the the experiment

		Output : obj_function(current_point, solutions),
		an objective function that takes as input a current_point and signs (a list of lists of 0s and 1s)
		and outputs scores relative to the current_point given as input (suited to being used by the signs_optimizer_for_triang_exploration function)"""

	output_scoring_file = temp_files_folder +'/temp_score.txt'		# the content of this file is not used in what follows (though it IS created, and a similar file is used in other situations)
	find_new_topologies = True
	list_of_homologies_file = list_of_homologies_file
	temp_homologies_file = os.path.join(temp_files_folder,"temp_homologies.txt")
	
	def obj_function(current_point : Current_Point, solutions):
		"""solutions is a list of lists of 0s and 1s"""
		solutions = np.array(solutions)
		return calc_score(current_point.local_path, temp_files_folder, polymake_scoring_script, solutions,\
				current_point.triang_file, current_point.all_points_file, current_point.current_points_indices_file, output_scoring_file,\
				current_point.degree, current_point.dim, find_new_topologies, list_of_homologies_file, temp_homologies_file).tolist()
	return obj_function


def create_objective_function_for_move_selector(dim, degree, local_path, list_of_homologies_file, temp_files_folder, polymake_scoring_script):
	""" Input : various information related to the experiment

		Output : obj_function(triangs_file, signs_file, points_file, points_indices_file), 
		an objective function that takes as input triangs and signs files (and points and points_indices) and outputes scores,
		suited to being used by a move_selector
		"""

	def obj_function(triangs_file, signs_file, points_file, points_indices_file):
		temp_homologies_file = os.path.join(temp_files_folder,"temp_homologies.txt")
		find_new_topologies = True
		output_scoring_file = os.path.join(temp_files_folder,'temp_score.txt')	# this is in fact not used (though it must be provided, and the file IS created)

		return calc_score(local_path, temp_files_folder, polymake_scoring_script, signs_file,\
				triangs_file, points_file, points_indices_file, output_scoring_file,\
				degree, dim, find_new_topologies, list_of_homologies_file, temp_homologies_file)
	return obj_function



def triangulation_growing_objective_function(triangs_file, signs_file, points_file, points_indices_file):
	"""grows the triangulation  (to be used by a move_selector)"""
	scores =[]
	with open(triangs_file,"r") as f:
		for line in f:
			scores.append(len(line.split("},{")))
	return np.array(scores)


def create_triangulation_growing_and_look_while_growing_objective_function(degree, dim, local_path, list_of_homologies_file, temp_files_folder):
	"""Creates an objective function that grows the triangulation and saves the homologies met along the way while doing so (to be used by a move_selector)"""

	# since the scores don't matter, any polymake script will do
	polymake_scoring_script = "Scoring/score_b_total.pl"
	homology_computing_function = create_objective_function_for_move_selector(degree, dim, local_path, list_of_homologies_file, temp_files_folder, polymake_scoring_script)
	def obj_function(triangs_file, signs_file, points_file, points_indices_file):
		
		# we compute and record the homology, but don't use it as an objective function
		homology_computing_function(triangs_file, signs_file, points_file, points_indices_file)
		
		scores =[]
		with open(triangs_file,"r") as f:
			for line in f:
				scores.append(len(line.split("},{")))
		return np.array(scores)
	return obj_function

def find_and_save_new_hom(signs_input_file, triangs_input_file, temp_homologies_file,list_of_homologies_file):
	# test if file already exists :
	if not os.path.isfile(list_of_homologies_file):
		with open(list_of_homologies_file, 'w') as f:
			# create the file and do nothing
			pass
		
	# Set of strings
	known_hom = set()
	with open(list_of_homologies_file, 'r') as f:
		for line in f:
			if line !="\n":
				homology = line.split("|")[0]
				known_hom.add(homology)

	signs = []
	with open(signs_input_file,'r') as f:
		for line in f:
			signs.append(line)

	triangs = []
	with open(triangs_input_file,'r') as f:
		for line in f:
			if line != "\n":
				triangs.append(line.rstrip("\n"))

	with open(list_of_homologies_file, 'a') as f_archive:
		with open(temp_homologies_file, 'r') as f_temp:
			for index, line in enumerate(f_temp):
				line = line.rstrip("\n")
				if line not in known_hom:
					print("\nNew homological profile found : "+line+"\n")
					known_hom.add(line)
					# case where there is a single triangulation for all the signs distributions
					if len(triangs) ==1 :
						f_archive.write(line+"|"+signs[index].replace("\n","")+"|"+triangs[0]+"\n")
					# case where there is a triangulation for each signs distribution
					else:
						f_archive.write(line+"|"+signs[index].replace("\n","")+"|"+triangs[index]+"\n")
				


#if __name__ == "__main__":
#jitted_calc_score(np.array([1,0,1,0,1,1],dtype=float))
