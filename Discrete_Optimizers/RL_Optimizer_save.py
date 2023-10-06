

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utilities import starting_CPU_and_wall_time, CPU_and_wall_time, waste_CPU_time, import_tensorflow

import time

import copy

tf = import_tensorflow()    
#from tensorflow import keras


from Discrete_Optimizer import Discrete_Optimizer




class RL_Optimizer(Discrete_Optimizer):
	"""
	Optimizes with Reinforcement Learning

	Everything is a numpy array, except initial_solutions, the input of obj_function and the solutions passed to end_of_iteration_routine
	"""

	def get_parameters_from_strings(parameters):
		"""
		parameters is a list of strings
		n_layers      width_layers   learning_rate  n_sessions   	min_randomness    alpha
		    0         1-n_layers       n_layers+1     n_layers+2        n_layers+3   n_layers+4
		"""
		# TODO specify percentile ?
		n_lay = int(parameters[0])
		min_randomness = True if parameters[n_lay+3]== "True" else False
		optimizer_parameters = [n_lay]+ [[int(parameters[i]) for i in range(1,n_lay+1)]] + [float(parameters[n_lay+1])] +  [int(parameters[n_lay+2])] + \
		[85, 95] + [min_randomness] + ([float(parameters[n_lay+4])] if min_randomness else [None])

		return optimizer_parameters, "RL_"+"_".join(parameters)

	

	def __init__(self, n_layers, width_layers, learning_rate, n_sessions, percentile, super_percentile, min_randomness, alpha, \
		local_path, saved_results_folder, exp_name, \
		optimizer_name = "Discrete optimizer", n_solutions_to_display = 5, feedback_period= 5, \
		saving_perf_period = 20, n_current_solutions_saved = 5, saving_solutions_period = None, n_all_time_best_solutions_saved = 5, random_seed = None):
		"""
		n_layers is an integer

		width_layers is a list of integers
		"""
		
		self.n_layers = n_layers
		self.width_layers = width_layers
		self.learning_rate = learning_rate
		self.n_sessions = n_sessions
		self.percentile = percentile
		self.super_percentile = super_percentile
		self.min_randomness = min_randomness
		self.alpha = alpha

		
		super().__init__(local_path, saved_results_folder, exp_name, \
		optimizer_name = optimizer_name, n_solutions_to_display = n_solutions_to_display, feedback_period= feedback_period, \
		saving_perf_period = saving_perf_period, n_current_solutions_saved = n_current_solutions_saved, \
		saving_solutions_period = saving_solutions_period, n_all_time_best_solutions_saved = n_all_time_best_solutions_saved, random_seed=random_seed)

	def setup(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):

		# needed for what follows
		self.dim = dim
		self.len_game = self.dim
		self.observation_space = 2*self.dim
		self.obj_function = obj_function

		if self.min_randomness:
			self.min_randomness_thresh = min(1- self.alpha**(1/float(self.dim)),0.45)
			print(f"\nMinimum randomness threshold = {self.min_randomness_thresh}\n")
		else:
			self.min_randomness_thresh = 0

		self.model = tf.keras.Sequential()
		for width in self.width_layers:
			self.model.add(tf.keras.layers.Dense(width,  activation="relu"))
		self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
		self.model.build((None, self.observation_space))
		self.model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate)) # SGD also works well, with higher learning rate

		#print(self.model.summary())

		self.super_states =  np.empty((0, self.len_game, self.observation_space), dtype = int)
		self.super_actions = np.empty((0, self.len_game), dtype = int)
		self.super_rewards = np.array([])

		if initial_solutions != None and initial_solutions != []:
		# tests if initial_solutions is a list of solutions or list of pairs (solution, score), makes it a list of pairs if it isn't the case
			if isinstance(initial_solutions[0], tuple) == False:
				scores = self.obj_function(initial_solutions)
				initial_solutions = [(initial_solutions[i], scores[i]) for i in range(len(initial_solutions))]
			# to be used by the RL algorithm
			self.super_states, self.super_actions, self.super_rewards = self.states_actions_rewards_from_solutions([sol[0] for sol in initial_solutions], scores = [sol[1] for sol in initial_solutions])

		super().setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)


	def generate_session(self):	
		"""
		Play n_session games using agent neural network.
		Terminate when games finish 
		Returns solutions (a list of pairs (solution, score)) for ease of use
		
		Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
		"""
		states =  np.zeros([self.n_sessions, self.observation_space, self.len_game], dtype=int)
		actions = np.zeros([self.n_sessions, self.len_game], dtype = int)
		state_next = np.zeros([self.n_sessions,self.observation_space], dtype = int)
		prob = np.zeros(self.n_sessions)
		states[:,self.dim,0] = 1
		step = 0
		total_score = np.zeros([self.n_sessions])
		
		while (True):
			step += 1		
			prob = self.model.predict(states[:,:,step-1], batch_size = self.n_sessions, verbose = 0)
			actions, state_next, states, total_score, terminal, CPU_scoring_time = self.play_game(actions, state_next, states, prob, step, total_score)
			
			if terminal:
				solutions = [(state_next[i,:self.dim].tolist(),total_score[i]) for i in range(self.n_sessions)]
				break
		return (states, actions, total_score), CPU_scoring_time, solutions


	# Updates by one step the vectors [0,1,1,0...,  0,0,...,0,1,0,...] for n_sessions parallel sessions
	def play_game(self, actions, state_next, states, prob, step, total_score):
		terminal = step == self.dim
		for i in range(self.n_sessions):
			
			if np.random.rand() < min(max(prob[i], self.min_randomness_thresh),1-self.min_randomness_thresh):
				action = 1
			else:
				action = 0

			actions[i][step-1] = action
			state_next[i] = states[i,:,step-1]

			if (action > 0):
				state_next[i][step-1] = action
			state_next[i][self.dim + step-1] = 0
			if (step < self.dim):
				state_next[i][self.dim + step] = 1		

			# record sessions 
			if not terminal:
				states[i,:,step] = state_next[i]	
		# calculate final score (modified compared to base version)
		# Now calc_score takes a [n_sessions, dim] numpy array as an input
		if terminal:
			scoring_starting_time = starting_CPU_and_wall_time()
			total_score = self.obj_function(state_next[:,:self.dim].tolist())
			CPU_scoring_time = CPU_and_wall_time(scoring_starting_time)[0]
		else :
			total_score = None
			CPU_scoring_time = 0
		return actions, state_next, states, total_score, terminal, CPU_scoring_time

	def select_elites(self, states_batch, actions_batch, rewards_batch):
		"""
		Select states and actions from games that have rewards >= percentile
		:param states_batch: list of lists of states, states_batch[session_i][t]
		:param actions_batch: list of lists of actions, actions_batch[session_i][t]
		:param rewards_batch: list of rewards, rewards_batch[session_i]
		:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
		
		This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
		If this function is the bottleneck, it can easily be sped up using numba
		"""
		counter = self.n_sessions * (100.0 - self.percentile) / 100.0
		reward_threshold = np.percentile(rewards_batch, self.percentile)

		elite_states = []
		elite_actions = []
		elite_rewards = []
		for i in range(len(states_batch)):
			if rewards_batch[i] >= reward_threshold-0.0000001:		
		# a bit weird ?
				if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
					for item in states_batch[i]:
						elite_states.append(item.tolist())
					for item in actions_batch[i]:
						elite_actions.append(item)			
				counter -= 1
		elite_states = np.array(elite_states, dtype = int)	
		elite_actions = np.array(elite_actions, dtype = int)	
		return elite_states, elite_actions

		
	def select_super_sessions(self, states_batch, actions_batch, rewards_batch):
		"""
		Select all the sessions that will survive to the next generation
		Similar to select_elites function
		If this function is the bottleneck, it can easily be sped up using numba
		"""
		
		counter = self.n_sessions * (100.0 - self.super_percentile) / 100.0
		reward_threshold = np.percentile(rewards_batch, self.super_percentile)

	# Not the same shape as in select_elites
		super_states = []
		super_actions = []
		super_rewards = []
		for i in range(len(states_batch)):
			if rewards_batch[i] >= reward_threshold-0.0000001:
				if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
					super_states.append(states_batch[i])
					super_actions.append(actions_batch[i])
					super_rewards.append(rewards_batch[i])
					counter -= 1
		super_states = np.array(super_states, dtype = int)
		super_actions = np.array(super_actions, dtype = int)
		super_rewards = np.array(super_rewards)
		return super_states, super_actions, super_rewards

	def states_actions_rewards_from_solutions(self, solutions, scores = None):
		"""turns solutions (a list of lists of 0s and 1s) into states, actions and rewards to initialize the RL algorithm
		Does not recompute scores if they have already been computed"""
		actions = solutions
		if scores == None:
			total_score =  np.array(self.obj_function(solutions))
		else:
			total_score = scores
		states = []
		for i in range(np.shape(solutions)[0]):
			states_seq = [[0]*self.dim+[1]+[0]*(self.dim-1)]
			for j in range(self.dim-1):
				current_state = copy.deepcopy(states_seq[-1])
				current_state[j] = solutions[i][j]
				current_state[self.dim+j] = 0
				current_state[self.dim+j+1] = 1
				states_seq.append(current_state)
			states.append(states_seq)
		return (np.array(states), np.array(actions), np.array(total_score))



	
	def optimize(self, n_iter, dim, obj_function, initial_solutions = None, stopping_condition = None, max_running_time = None, clear_log = True):
		""" The main optimization function - optimizes with respect to obj_function

		stopping_condition is either None or a function that takes current solutions (a list of pairs (solution, score)) as input
		and outputs True if some stopping condition has been reached (and stops the optimization should it be the case)

		initial_solutions must be either a list of solutions, or a list of pairs (solution, score), or None"""

		self.setup(n_iter, dim, obj_function, initial_solutions, stopping_condition, max_running_time, clear_log)
		starting_time = starting_CPU_and_wall_time()
		iteration = 0
		stop = False
		while(stop == False):
			iteration_starting_time = starting_CPU_and_wall_time()


			
			# solutions is only of use to the routines of the Discrete_Optimizer class, not to the rest of the RL optimization algorithm
			sessions, scoring_time, solutions = self.generate_session()

			states_batch = np.array(sessions[0], dtype = int)
			actions_batch = np.array(sessions[1], dtype = int)
			rewards_batch = np.array(sessions[2])
			states_batch = np.transpose(states_batch,axes=[0,2,1])

			states_batch = np.append(states_batch, self.super_states, axis=0)
			# if iteration>0: # modified compared to original code
			actions_batch = np.append(actions_batch, self.super_actions, axis=0)
			rewards_batch = np.append(rewards_batch, self.super_rewards)



			elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch) #pick the sessions to learn from
	
			super_sessions = self.select_super_sessions(states_batch, actions_batch, rewards_batch) #pick the sessions to survive

			super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
			super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)


			# NOTE currently a single epoch of training (default behavior of model.fit)
			self.model.fit(elite_states, elite_actions, verbose = 0) # learn from the elite sessions
			
			
			self.super_states =  [super_sessions[i][0] for i in range(len(super_sessions))]
			self.super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
			self.super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]

			stop = super().end_of_iteration_routine(iteration, solutions, iteration_time = CPU_and_wall_time(iteration_starting_time)[0],\
				 scoring_time = scoring_time, current_running_time = CPU_and_wall_time(starting_time)[0])
			

			iteration += 1
			if stop:
				super().end_of_run_routine(starting_time)
				return super().get_all_time_best_solution_and_score()






if __name__ == "__main__":
	my_local_path = "/home/charles/Desktop/ML_RAG/Code/Discrete_Optimizers"
	saved_files_folder = "saved_files"

	
	n_layers = 2
	width_layers = [64, 64]
	learning_rate = 0.1
	n_sessions = 100
	percentile = 0.9
	super_percentile = 0.9
	min_randomness = True
	alpha = 0.90

	opti = RL_Optimizer(n_layers, width_layers, learning_rate, n_sessions, percentile, super_percentile, min_randomness, alpha,\
		my_local_path, saved_files_folder, "test_RL",\
		optimizer_name = "RL optimizer", n_solutions_to_display=6, n_current_solutions_saved=3, n_all_time_best_solutions_saved=5,\
			feedback_period=1, saving_perf_period= 3, saving_solutions_period=6 )

	n_iter = 6
	dim = 5
	obj_function = lambda my_liste : [sum(x) for x in my_liste]
	initial_solutions = [([1,1,1,1,1], 5), ([0,0,0,0,0],0)]
	opti.optimize(n_iter, dim, obj_function, initial_solutions = initial_solutions, stopping_condition = None, max_running_time = 100, clear_log = True)










# print("\n\nUsing reinforcement learning to optimize signs distribution.\n")

# N_LAYERS = int(sys.argv[1])
# WIDTH_LAYERS = [int(width) for width in sys.argv[2:2+N_LAYERS]]
# LEARNING_RATE, n_sessions, percentile, super_percentile, MIN_RANDOMNESS, ALPHA, \
# 	DEGREE, DIMENSION, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME, LOCAL_PATH, TEMP_FILES_FOLDER, OUTPUT_SCORING_FILE, POLYMAKE_SCORING_SCRIPT,\
# 	TRIANGULATION_INPUT_FILE, POINTS_INPUT_FILE, RELEVANT_POINTS_INDICES_INPUT_FILE,STARTING_SIGNS_DISTRIBUTIONS_FILE, \
# 	TEMP_HOMOLOGIES_FILE,  FIND_NEW_TOPOLOGIES, LIST_OF_HOMOLOGIES_FILE , SAVE_PERF_FILE, SAVE_PERIOD, OUTPUT_FILE = sys.argv[2+N_LAYERS:]

# DEGREE = int(DEGREE)
# DIMENSION = int(DIMENSION)
# STOPPING_OBJ_VALUE = int(STOPPING_OBJ_VALUE)
# MAX_RUNNING_TIME = int(MAX_RUNNING_TIME)
# LEARNING_RATE = float(LEARNING_RATE)
# n_sessions = int(n_sessions) #!= number of generations
# percentile = float(percentile)
# super_percentile = float(super_percentile)
# FIND_NEW_TOPOLOGIES = True if FIND_NEW_TOPOLOGIES == "True" else False
# MIN_RANDOMNESS = True if MIN_RANDOMNESS == "True" else False
# ALPHA = float(ALPHA)
# SAVE_PERIOD = int(SAVE_PERIOD)


# # Can njit make the input/output part of the score function faster ?
# # Apparently no, according to the internet
# # For now, NJIT might not work

# USE_NJIT = False

# ARCHIVE_FOLDER_PATH = "archive_files_RL"


# n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
# 			  #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
			  
# #observation_space = 2*N_SIGNS #Leave this at 2*N_SIGNS. The input vector will have size 2*N_SIGNS, where the first N_SIGNS letters encode our partial word (with zeros on
# 						  #the positions we haven't considered yet), and the next N_SIGNS bits one-hot encode which letter we are considering now.
# 						  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
# 						  #Is there a better way to format the input to make it easier for the neural network to understand things?



# # Get the number of signs
# with open(os.path.join(LOCAL_PATH, RELEVANT_POINTS_INDICES_INPUT_FILE), 'r') as f:
# 	N_SIGNS =  len(f.readline().split(","))
# 	observation_space = 2*N_SIGNS
# 	print(f"\nNumber of signs to generate : {N_SIGNS}\n")


	
# if MIN_RANDOMNESS:
# 	MIN_RANDOM_THRESH = min(1- ALPHA**(1/float(N_SIGNS)),0.45)
# 	print(f"\nMinimum randomness threshold = {MIN_RANDOM_THRESH}\n")
# else:
# 	MIN_RANDOM_THRESH = 0


# len_game = N_SIGNS 
# state_dim = (observation_space,)

# INF = 1000000


# #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
# #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
# #It is important that the loss is binary cross-entropy if alphabet size is 2.

# model = keras.Sequential()
# for width in WIDTH_LAYERS:
# 	model.add(keras.layers.Dense(width,  activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))
# model.build((None, observation_space))
# model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = LEARNING_RATE)) #Adam optimizer also works well, with lower learning rate

# print(model.summary())


# """# Running the experiment"""


# super_states =  np.empty((0,len_game,observation_space), dtype = int)
# super_actions = np.array([], dtype = int)
# super_rewards = np.array([])
# sessgen_time = 0
# fit_time = 0
# score_time = 0



# myRand = random.randint(0,1000) #used in the filename

# STARTING_TIME = time.time()

# for i in range(10000000): #1000000 generations should be plenty
# 	print(f"\nStart session {i}\n")
# 	generation_starting_time = time.time()
# 	#generate new sessions
# 	#performance can be improved with joblib
# 	tic = time.time()
# 	sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
# 	sessgen_time = time.time()-tic
# 	tic = time.time()
	
# 	states_batch = np.array(sessions[0], dtype = int)
# 	actions_batch = np.array(sessions[1], dtype = int)
# 	rewards_batch = np.array(sessions[2])
# 	states_batch = np.transpose(states_batch,axes=[0,2,1])

	
# 	states_batch = np.append(states_batch,super_states,axis=0)

# 	if i>0:
# 		actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
# 	rewards_batch = np.append(rewards_batch,super_rewards)
		
# 	randomcomp_time = time.time()-tic 
# 	tic = time.time()

# 	elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
# 	select1_time = time.time()-tic

# 	tic = time.time()
# 	super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
# 	select2_time = time.time()-tic
	
# 	tic = time.time()
# 	super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
# 	super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
# 	select3_time = time.time()-tic
	
# 	tic = time.time()
# 	model.fit(elite_states, elite_actions) #learn from the elite sessions
# 	fit_time = time.time()-tic
	
# 	tic = time.time()
	
# 	super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
# 	super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
# 	super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
	
# 	rewards_batch.sort()
# 	mean_all_reward = np.mean(rewards_batch[-100:])	
# 	mean_best_reward = np.mean(super_rewards)	

# 	score_time = time.time()-tic
	
# 	print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
	
# 	#uncomment below line to print out how much time each step in this loop takes. 
# 	# Rmk : score_time is not what you would (reasonably) expect it to be
# 	#print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
	
# 	generation_time = time.time()-generation_starting_time
# 	print(	"Mean reward: " + str(mean_all_reward) + "\nGeneration total time: "+str(generation_time) ) 
	
# 	save_performance(max(super_rewards), time.time() - STARTING_TIME,SAVE_PERIOD,os.path.join(LOCAL_PATH,SAVE_PERF_FILE))

# 	if STOPPING_OBJ_VALUE in super_rewards :
# 		print("Objective reached")
# 		break
# 	if time.time() - STARTING_TIME > MAX_RUNNING_TIME:
# 		print("Time limit exceeded")
# 		break

# 	# Old way of saving results
# 	"""
# 	if (i%20 == 1): #Write all important info to files every 20 iterations
# 		with open(ARCHIVE_FOLDER_PATH+'/best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
# 			pickle.dump(super_actions, fp)
# 		with open(ARCHIVE_FOLDER_PATH+'/best_species_txt_'+str(myRand)+'.txt', 'w') as f:
# 			for item in super_actions:
# 				f.write(str(item))
# 				f.write("\n")
# 		with open(ARCHIVE_FOLDER_PATH+'/best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
# 			for item in super_rewards:
# 				f.write(str(item))
# 				f.write("\n")
# 		with open(ARCHIVE_FOLDER_PATH+'/best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
# 			f.write(str(mean_all_reward)+"\n")
# 		with open(ARCHIVE_FOLDER_PATH+'/best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
# 			f.write(str(mean_best_reward)+"\n")
# 	if (i%200==2): # To create a timeline, like in Figure 3
# 		with open(ARCHIVE_FOLDER_PATH+'/best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
# 			f.write(str(super_actions[0]))
# 			f.write("\n")
# 	"""

	


