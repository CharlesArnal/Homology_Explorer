
import numpy as np
# Apparently unused
#from keras.utils import to_categorical

import os
import copy
import subprocess

from Current_Point import Current_Point

	

def generate_moves_nb_triangs(current_point : Current_Point, temp_files_folder):
	""" Generated moves are the triangulations flip-connected to the current one, all with the current signs distribution"""
	local_path = current_point.local_path
	
	nb_triangs_file_path = os.path.join(local_path, temp_files_folder,"nb_triangs.dat")
	nb_flips_file_path = os.path.join(local_path, temp_files_folder,"nb_flips.dat")
	nb_signs_file_path = os.path.join(local_path, temp_files_folder,"nb_signs.dat")
	# to start with an empty file (functions below append rather than write)
	with open(nb_signs_file_path,"w") as f:
		pass
	nb_relevant_points_indices_file_path = os.path.join(local_path, temp_files_folder,"nb_relevant_points_indices.dat")
	# compute all the neighbouring triangulations (and associated flips), stores them in nb_triangs_file_path and and nb_flips_file_path
	list_files = subprocess.run(["./nb_triangs", current_point.chiro_file, current_point.triang_file, current_point.symmetries_file, current_point.flips_file,\
		nb_triangs_file_path, nb_flips_file_path])
	# get the indices of the relevant points of the neighbouring triangulations (a list of sets)
	current_points_indices = get_current_points_indices(current_point.current_points_indices_file)
	moving_indices = get_moving_indices(nb_flips_file_path)
	store_relevant_points_indices_from_flips(current_points_indices, nb_relevant_points_indices_file_path, moving_indices, write_or_append = "w")
	# copies the correct number of times the current signs distribution in nb_signs_file_path
	# if a point becomes redundant, the corresponding sign is removed
	# if a new point is added, two copies of the signs distribution are added (one with a 0, one with a 1)
	# (oc there are also two copies of the corresponding triangulation, flips and relevant points in the relevant files)
	adapt_signs_distributions_to_new_triangs(current_points_indices, moving_indices, current_point.signs_file, nb_triangs_file_path,\
		nb_flips_file_path, nb_relevant_points_indices_file_path, nb_signs_file_path)

	return (nb_triangs_file_path, nb_flips_file_path, nb_signs_file_path, nb_relevant_points_indices_file_path, local_path, temp_files_folder)
	


def generate_moves_nb_triangs_nb_signs(current_point: Current_Point, temp_files_folder):
	""" Generated moves are the triangulations flip-connected to the current one with the current signs distribution,
		as well as the current triangulation with the neighbouring signs distributions
		current_point should be a list of file names : chiro, current_triang_and_signs, symmetries, current_flips, local_path, temp_files_folder
	"""
	local_path = current_point.local_path
	nb_triangs_file_path = os.path.join(local_path,temp_files_folder,"nb_triangs.dat")
	nb_flips_file_path = os.path.join(local_path,temp_files_folder,"nb_flips.dat")
	nb_signs_file_path = os.path.join(local_path,temp_files_folder,"nb_signs.dat")
	# to start with an empty file (functions below append rather than write)
	with open(nb_signs_file_path,"w") as f:
		pass
	nb_relevant_points_indices_file_path = os.path.join(local_path,temp_files_folder,"nb_relevant_points_indices.dat")
	# compute all the neighbouring triangulations (and associated flips), stores them in nb_triangs_file_path and nb_flips_file_path
	list_files = subprocess.run(["./nb_triangs", current_point.chiro_file, current_point.triang_file, current_point.symmetries_file, current_point.flips_file,\
		nb_triangs_file_path, nb_flips_file_path])
	# get the indices of the relevant points of the neighbouring triangulations (a list of sets)
	current_points_indices = get_current_points_indices(current_point.current_points_indices_file)
	moving_indices = get_moving_indices(nb_flips_file_path)
	store_relevant_points_indices_from_flips(current_points_indices, nb_relevant_points_indices_file_path, moving_indices, write_or_append ="w")
	# copies the correct number of times the current signs distribution in nb_signs_file_path
	# if a point becomes redundant, the corresponding sign is removed
	# if a new sign is added, two copies of the signs distribution are added (one with a 0, one with a 1)
	# (oc there are also two copies of the corresponding triangulation, flips and relevant points in the relevant files)
	adapt_signs_distributions_to_new_triangs(current_points_indices,moving_indices, current_point.signs_file, nb_triangs_file_path,\
		nb_flips_file_path, nb_relevant_points_indices_file_path, nb_signs_file_path)
	# creates and stores nb signs distributions
	# also copies and adds the correct number of times the current triangulation, flips and relevant points indices
	create_and_store_nb_signs_distributions(current_point.signs_file, current_points_indices, current_point.triang_file,\
		nb_signs_file_path,nb_triangs_file_path,nb_relevant_points_indices_file_path,nb_flips_file_path)
	return (nb_triangs_file_path, nb_flips_file_path, nb_signs_file_path, nb_relevant_points_indices_file_path, local_path, temp_files_folder)



def create_and_store_nb_signs_distributions(current_signs_file_path,current_points_indices,current_triang_file_path,\
	nb_signs_file_path,nb_triangs_file_path,nb_relevant_points_indices_file_path,nb_flips_file_path):
	with open(current_signs_file_path, "r") as f:
		current_signs = np.loadtxt(f,dtype = int)
		# adds at the end of nb_signs_file_path all the neighbouring signs distributions
		dim = np.size(current_signs)
		nb_signs = []
		for i in range(dim):
			current_signs[i] = 1- current_signs[i]
			nb_signs.append(copy.deepcopy(current_signs))
			current_signs[i] = 1- current_signs[i]
		with open(nb_signs_file_path, "a") as g:
			np.savetxt(g,np.array(nb_signs),fmt='%d')
	n_signs_added = len(nb_signs)
	# adds at the end of nb_triangs_file_path as many copies as needed of the current triangulation
	with open(current_triang_file_path, "r") as f:
		current_triang = f.readline().replace("\n","")
		with open(nb_triangs_file_path, "a") as g:
			for i in range(n_signs_added):
				g.write(current_triang + "\n")
	# adds at the end of nb_flips_file_path as many copies as needed of the trivial flip
	with open(nb_flips_file_path, "a") as g:
		for i in range(n_signs_added):
			g.write("[]\n")
	# adds at the end of nb_relevant_points_indices_file_path as many copies as needed of the current relevant points indices
	store_relevant_points_indices_from_flips(current_points_indices, nb_relevant_points_indices_file_path, [(None,None)]*n_signs_added, write_or_append="a")


def entering_and_exiting_indices_from_flip(flip : str):
	# flip of the shape [{1,2},{3,4,5}]
	entering_index = None
	exiting_index = None
	flip = flip.replace("\n","")
	flip = flip[1:-1]
	if flip != "":
		blocs = flip.split("},{")
		blocs[0] = blocs[0].replace("{","")
		blocs[1] = blocs[1].replace("}","")
		if "," not in blocs[0]:
			entering_index = int(blocs[0])
		if "," not in blocs[1]:
			exiting_index = int(blocs[1])
	return entering_index, exiting_index

def get_moving_indices(flips_file):
	moving_indices = []
	with open(flips_file,"r") as f:
		for line in f:
			selected_flip = line.replace("\n","")
			entering_index, exiting_index = entering_and_exiting_indices_from_flip(selected_flip)
			moving_indices.append((entering_index,exiting_index))
	return moving_indices


def store_relevant_points_indices_from_flips(current_points_indices : set, nb_relevant_points_indices_file: str, moving_indices, write_or_append):
	with open(nb_relevant_points_indices_file,write_or_append) as f:
		for (entering_index,exiting_index) in  moving_indices:
			new_points_indices = copy.deepcopy(current_points_indices)
			if entering_index != None:
				new_points_indices.add(entering_index)
			if exiting_index != None:
				new_points_indices.remove(exiting_index)
			f.write(str(new_points_indices)+"\n")


def get_current_points_indices(current_points_indices_file_path):
	with open(current_points_indices_file_path,"r") as h:
		current_points_indices = {int(index) for index in h.readline().replace("\n","").replace("{","").replace("}","").split(",")}
	return current_points_indices


def adapt_signs_distributions_to_new_triangs(current_points_indices,moving_indices,current_signs_file_path,\
	nb_triangs_file_path,nb_flips_file_path,nb_relevant_points_indices_file_path,nb_signs_file_path):
	ordered_current_indices = sorted(current_points_indices)
	triangulations_to_add_at_the_end = []
	flips_to_add_at_the_end = []
	relevant_points_to_add_at_the_end = []
	signs_to_add_at_the_end = []
	with open(current_signs_file_path, "r") as f:
		current_signs = np.loadtxt(f,dtype = int)
	with open(nb_signs_file_path,"a") as g:
		with open(nb_triangs_file_path, "r") as f:
			with open(nb_flips_file_path, "r") as h:
				with open(nb_relevant_points_indices_file_path, 'r') as i:
					nb_flips = h.readlines()
					nb_relevant_points = i.readlines()
					for index, line in enumerate(f):
						if moving_indices[index] == (None,None):
							np.savetxt(g,np.array([current_signs]),fmt='%d')
						if moving_indices[index][1]!= None :
							exiting_index = moving_indices[index][1]
							position = ordered_current_indices.index(exiting_index)
							adapted_signs = np.array([[i for j, i in enumerate(current_signs) if j!=position]])
							np.savetxt(g,adapted_signs,fmt='%d')
						if moving_indices[index][0] != None:
							entering_index = moving_indices[index][0]
							new_ordered_indices = sorted(ordered_current_indices+[entering_index])
							position = new_ordered_indices.index(entering_index)
							adapted_signs = np.array([current_signs.tolist()[:position]+[0]+current_signs.tolist()[position:]])
							np.savetxt(g,adapted_signs,fmt='%d')
							triangulations_to_add_at_the_end.append(line)
							flips_to_add_at_the_end.append(nb_flips[index])
							relevant_points_to_add_at_the_end.append(nb_relevant_points[index])
							signs_to_add_at_the_end.append(np.array([current_signs.tolist()[:position]+[1]+current_signs.tolist()[position:]]))
	with open(nb_triangs_file_path, "a") as f:
		for triang in triangulations_to_add_at_the_end:
			f.write(triang)
	with open(nb_signs_file_path, "a") as g:
		for signs in signs_to_add_at_the_end:
			np.savetxt(g,signs,fmt='%d')
	with open(nb_relevant_points_indices_file_path, "a") as h:
		for points in relevant_points_to_add_at_the_end:
			h.write(points)
	with open(nb_flips_file_path, "a") as i:
		for flip in flips_to_add_at_the_end:
			i.write(flip)

