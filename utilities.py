from math import floor, ceil, comb
import copy
import numpy as np
import os
import resource
import time


def waste_CPU_time(time_in_secs):
	starting_time = starting_CPU_and_wall_time()
	while(CPU_and_wall_time(starting_total_timestamp=starting_time)[0] < time_in_secs):
		np.random.normal(0,1,size=(100,100))

def get_n_points_and_dim_from_chiro(chiro_file_name):
	with open(chiro_file_name,"r") as f:
		first_line = f.readline().replace("\n","")
		first_line = first_line[:-1]
		return [int(x) for x in first_line.split(",")]



def Smith_Thom_bound(dim,degree):
	return int(((degree-1)**(dim+1)-(-1)**(dim+1))/degree + dim +(-1)**(dim+1))

def difference_parity(dim,degree,candidate):
	if (Smith_Thom_bound(dim,degree)- sum(candidate))%2 != 0:
		return False
	else:
		return True

def p_q_th_Hodge_n(dim,degree,p,q):
	if p+q>2*(dim-1) or p<0 or q<0:
		return 0
	if p+q!=dim-1 :
		if p == q :
			return 1
		else:
			return 0
	else:
		return sum([(-1)**i *comb(dim+1,i)*(comb(degree*(p+1)-(degree-1)*i-1,dim) if degree*(p+1)-(degree-1)*i-1 >= 0 else 0)    for i in range(dim+2)])+ (1 if 2*p==dim-1 else 0)


def signature_projective_hypersurface(dim,degree):
	my_sum = 0
	for i in range(dim):
		for j in range(dim):
			my_sum += (-1)**i * p_q_th_Hodge_n(dim,degree,i,j)
	return my_sum

def Euler_char(homology_profile):
	return sum([(-1)**index * b for index, b in enumerate(homology_profile)])

def Kharlamov(dim,degree,candidate):
	# the theorem doesn't apply if the ambient dim is even
	if dim%2 ==0:
		return True
	else:
		if abs(Euler_char(candidate)-1) > p_q_th_Hodge_n(dim,degree,int((dim-1)/2),int((dim-1)/2)) -1:
			return False
		else:
			return True

def Smith_Thom(dim,degree,candidate):
	if sum(candidate)> Smith_Thom_bound(dim,degree):
		return False
	else:
		return True
	
def Rokhlin(dim,degree,candidate):
	# the theorem doesn't apply if the ambient dim is even
	if dim%2 ==0:
		return True
	if sum(candidate) == Smith_Thom_bound(dim,degree):
		if (Euler_char(candidate) - signature_projective_hypersurface(dim,degree))%16 !=0:
			return False
	return True

def KGK(dim,degree,candidate):
	if dim%2 ==0:
		return True
	if sum(candidate) == Smith_Thom_bound(dim,degree)-2:
		if (Euler_char(candidate) - signature_projective_hypersurface(dim,degree))%16 not in {2,14}:
			return False
	return True

def get_homology_interdictions(dim,degree):
	interdictions = []
	upper_bound = Smith_Thom_bound(dim, degree)
	candidates = [[0]*dim]
	for d in range(ceil(dim/2)):
		new_candidates = []
		for candidate in candidates:
			#for t in range(1,upper_bound-sum(candidate)+1):
			for t in range(1,upper_bound+1):
				new_candidate = copy.copy(candidate)
				new_candidate[d] = t
				new_candidate[dim-1-d] = t
				new_candidates.append(new_candidate)
				"""
				# counts double if d != dim-1/2
				if sum(new_candidate) <= upper_bound:
					new_candidates.append(new_candidate)
				"""
		candidates += new_candidates
	# TODO update
	known_interdictions = [Smith_Thom ,difference_parity, Kharlamov, Rokhlin, KGK]

	for candidate in candidates:
		if min([known_interdiction(dim,degree,candidate) for known_interdiction in known_interdictions]) == False:
			interdictions.append(candidate)
	return interdictions

def write_homologies_in_table(homologies,table_file,dim, degree, known_interdictions):
	# works for dim = 3, 4
	with open(table_file,"w") as g:
		max_b_0 = max([b[0] for b in homologies])
		max_b_1 = max([b[1] for b in homologies])
		for i in range( int(Smith_Thom_bound(dim,degree))+1):
		#for i in range( max_b_0+1):
			if i == 0:
				g.write("b0|b1 ")
				for j in range(int(Smith_Thom_bound(dim,degree))-1):
					g.write(str(j)+" "*(4-len(str(j))))
			else:
				g.write(str(i)+" "*(4-len(str(i))+2))
				for j in range(int(Smith_Thom_bound(dim,degree))-1):
					if dim == 3:
						homology_profile = [i,j,i]
					else:
						homology_profile = [i,j,j,i]
					if homology_profile in homologies and homology_profile in known_interdictions:
						g.write(f"x   ")
					elif homology_profile in homologies:
						if (i+j+i if dim==3 else i+j+j+i) == int(Smith_Thom_bound(dim,degree)):
							g.write(f"M   ")
						else:
							g.write(f"o   ")
					elif homology_profile in known_interdictions:
						g.write("    ")
					else:
						g.write("?   ")
			g.write("\n")

# legacy
"""
def turn_3D_homologies_file_into_table(homologies_file, table_file, degree):
	dim = 3
	homologies = []
	known_interdictions = get_homology_interdictions(3,degree)

	with open(homologies_file,"r") as f:
		for line in f:
			if line !="\n":
				homology = line.split("|")[0]
				b_0 = int(homology.split()[0])
				b_1 = int(homology.split()[1])
				homologies.append([b_0,b_1,b_0])
	write_homologies_in_table(homologies,table_file,dim, degree, known_interdictions)
"""

def turn_3D_or_4D_homologies_file_into_table(homologies_file, table_file, degree, dim):
	homologies = []
	known_interdictions = get_homology_interdictions(dim,degree)

	with open(homologies_file,"r") as f:
		for line in f:
			if line !="\n":
				homology = line.split("|")[0]
				b_0 = int(homology.split()[0])
				b_1 = int(homology.split()[1])
				if dim == 3:
					homologies.append([b_0,b_1,b_0])
				else:
					homologies.append([b_0,b_1,b_1,b_0])
	write_homologies_in_table(homologies,table_file,dim, degree, known_interdictions)


# legacy  
"""
def turn_all_3D_homologies_files_in_folder_into_single_table(folder_name,first_characters_name_homologies_files,table_file, degree):
	dim = 3
	homologies = []
	known_interdictions = get_homology_interdictions(3,degree)
	homology_files = [f for f in os.listdir(folder_name) if f[0:len(first_characters_name_homologies_files)]==first_characters_name_homologies_files]
	for file in homology_files:
		with open(folder_name+file,"r") as f:
			for line in f:
				if line !="\n":
					homology = line.split("|")[0]
					b_0 = int(homology.split()[0])
					b_1 = int(homology.split()[1])
					if [b_0,b_1,b_0] not in homologies:
						homologies.append([b_0,b_1,b_0])
					if b_0*2+b_1 > Smith_Thom_bound(dim,degree):
						print("Smith-Thom bound not respected !!!")
	write_homologies_in_table(homologies,table_file,dim, degree, known_interdictions)
"""

	
def turn_all_3D_or_4D_homologies_files_in_folder_into_single_table(folder_name,first_characters_name_homologies_files,table_file, degree, dim):
	homologies = []
	known_interdictions = get_homology_interdictions(dim,degree)
	homology_files = [f for f in os.listdir(folder_name) if f[0:len(first_characters_name_homologies_files)]==first_characters_name_homologies_files]
	for file in homology_files:
		with open(folder_name+file,"r") as f:
			for line in f:
				if line !="\n":
					homology = line.split("|")[0]
					b_0 = int(homology.split()[0])
					b_1 = int(homology.split()[1])
					if dim == 3:
						homology_profile = [b_0,b_1,b_0]
					else:
						homology_profile = [b_0,b_1,b_1,b_0]

					if homology_profile not in homologies:
						homologies.append(homology_profile)
					if (b_0*2+b_1 if dim==3 else 2*(b_0+b_1)) > Smith_Thom_bound(dim,degree):
						print("Smith-Thom bound not respected !!!")
	write_homologies_in_table(homologies,table_file,dim, degree, known_interdictions)
	# print(f"Smith Thom {Smith_Thom_bound(dim,degree)}")
	# print(f"ST {Smith_Thom(dim, degree, [1,32,32,1])}")
	# print(f"DP {difference_parity(dim, degree, [1,32,32,1])}")
	# print(f"Kharlamov {Kharlamov(dim, degree, [1,32,32,1])}")
	# print(f"Rokhlin {Rokhlin(dim, degree, [1,32,32,1])}")
	# print(f"KGK {KGK(dim, degree, [1,32,32,1])}")
	# print(f"1,32,32,1 in known_interdictions {[1,32,32,1]  in known_interdictions}")
	# print(f"known interdictions {known_interdictions}")
	


def read_known_homology_profiles(known_homologies_file):
	"""Returns them as a list of strings"""
	known_homs = []
	with open(known_homologies_file, 'r') as f:
		for line in f:
			if line !="\n":
				homology = line.split("|")[0]
				known_homs.append(homology)
	return known_homs
			

# triangulation must be written as {{0,6,11},{15,16,18}...} if format =="topcom", and as {{a b c}{e f g}} if format == "polymake"
# outputs a list of lists
def read_triang(triang_file, format = "topcom"):
	triang = []
	with open(triang_file,"r") as f:
		if format == "topcom":
			triang = f.readline().replace("\n","")[2:-2]
			triang = triang.split("},{")
			triang = [[int(index) for index in simplex.split(",")] for simplex in triang]
		if format == "polymake":
			triang = f.readline().replace("\n","")[2:-2]
			triang = triang.split("}{")
			triang = [[int(index) for index in simplex.split()] for simplex in triang]
	return triang

def write_triang(triang, triang_file, format = "polymake"):
	""" input : a list of lists
		writes as {{a b c}{e f g}} if format == "polymake", and as {{a,b,c},{e,f,g}} if format == "topcom
	"""
	with open(triang_file,"w") as f:
		f.write("{")
		if format == "polymake":
			for index, simplex in enumerate(triang):
				if index !=0:
					f.write("")
				f.write("{")
				for index_2, point in enumerate(simplex):
					if index_2 !=0:
						f.write(" ")
					f.write(str(point))
				f.write("}")
		elif format == "topcom":
			for index, simplex in enumerate(triang):
				if index !=0:
					f.write(",")
				f.write("{")
				for index_2, point in enumerate(simplex):
					if index_2 !=0:
						f.write(",")
					f.write(str(point))
				f.write("}")
		f.write("}")

def purify_triang(triangulation_file, new_triangulation_file):
	triang = read_triang(triangulation_file)
	new_triang = [] 
	indices = triang_relevant_indices(triang)
	for simplex in triang:
		new_simplex = []
		for  pt_index in simplex:
			new_simplex.append(indices.index(pt_index))
		new_triang.append(new_simplex)
	write_triang(new_triang, new_triangulation_file)
			
	return 0

# Returns various starting timestamps (CPU time of the process, its children, and wall time)
def starting_CPU_and_wall_time():
	return (resource.getrusage(resource.RUSAGE_SELF).ru_utime, resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime, time.time())

# takes as input the output of starting_total_time
def CPU_and_wall_time(starting_total_timestamp):
	CPU_self_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime - starting_total_timestamp[0]
	CPU_children_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - starting_total_timestamp[1]
	wall_time = time.time() - starting_total_timestamp[2]
	return (CPU_self_time+CPU_children_time, wall_time)

def triang_relevant_indices(triang):
	indices = []
	for simplex in triang:
		for pt_index in simplex:
			if pt_index not in indices:
				indices.append(pt_index)
	indices.sort()
	return indices

def purify_points(triang_file, all_points_file, relevant_points_file):
	triang = read_triang(triang_file)
	relevant_indices = triang_relevant_indices(triang)
	all_points = None
	with open(all_points_file,"r") as f:
		all_points = np.loadtxt(f,dtype = int)
	relevant_points = all_points[relevant_indices]
	np.savetxt(relevant_points_file,relevant_points,fmt='%d')
	return 0


def points2monomials(points_file, monomials_file):
	points = None
	with open(points_file,"r") as f:
		points = f.readlines()
	with open(monomials_file,'w') as f:
		for index, point in enumerate(points):
			if index !=0:
				f.write("\n")
			point = " ".join( point.split()[1:])
			f.write(point)


def first_experiment_param_file_writer(filename, dims, degrees,  signs_opti_times, total_times, signs_opti_algs, initial_triangulation_types,scoring_scripts, look_while_growing):
	print("\nCreating experiment parameters file")
	with open(filename,"w") as f:
		i = 0
		f.write("row_num dim degree signs_opti_time total_time signs_opti_alg scoring_script rich_triangulation look_while_growing")
		for dim in dims :
			for degree in degrees :
				for signs_opti_time in signs_opti_times:
					for signs_opti_alg in signs_opti_algs:
						for total_time in total_times:
							for scoring_script in scoring_scripts:
								for initial_triangulation_type in initial_triangulation_types:
									for look_value in look_while_growing:
										if dim == 3 or degree < 7:
											f.write("\n")
											i +=1
											f.write(f"{i} {dim} {degree} {signs_opti_time} {total_time} {signs_opti_alg} {scoring_script} {initial_triangulation_type} {look_value}")


def read_first_experiment_param_file(filename,line_number):
	with open(filename,"r") as f:
		parameters = dict()
		line = f.readlines()[line_number].split()
		parameters["dim"] = int(line[1])
		parameters["degree"] = int(line[2])
		parameters["signs_opti_time"] = int(line[3])
		parameters["total_time"] = int(line[4])
		parameters["signs_opti_alg"] = line[5]
		parameters["scoring_script"] = line[6]
		parameters["initial_triangulation_type"] = (line[7] if len(line) > 7 else "")
		parameters["look_while_growing_triangulation"] = (line[8] if len(line) > 8 else "")
		parameters["look_while_growing_triangulation"] = True if parameters["look_while_growing_triangulation"] == "True" else False
		return parameters


def import_tensorflow():
	"""Filter tensorflow version warnings"""
	import os
	# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
	import warnings
	# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
	warnings.simplefilter(action='ignore', category=FutureWarning)
	warnings.simplefilter(action='ignore', category=Warning)
	import tensorflow as tf
	tf.get_logger().setLevel('INFO')
	tf.autograph.set_verbosity(0)
	import logging
	tf.get_logger().setLevel(logging.ERROR)
	return tf

if __name__ == "__main__":
	# first_experiment_param_file_writer("parameters_exps_1.txt",[3],[4,5,6,7],[30,300], [36000],["TS","MCTS"],["Trivial","Large"],\
	#    ["Scoring/score_b_total.pl","Scoring/score_b_total_w_alpha_b_0.pl","Scoring/score_b_total_w_alpha_b_1.pl"])
	format = "polymake"
	write_triang([[1,2,3],[3,4,5],[-1,2,3]],"/home/charles/Desktop/ML_RAG/Code/General_test_temp_files/test1.txt", format=format)
	triang = read_triang("/home/charles/Desktop/ML_RAG/Code/General_test_temp_files/test1.txt",format=format)
	print(triang)
