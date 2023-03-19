
#import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc. Does not work with numba (yet)
import random
import numpy as np
# Apparently unused
#from keras.utils import to_categorical

import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import SGD, Adam
#from keras.models import load_model

import time
import os
import sys


from RanaFun import RanaFun
from RosenbrockFun import RosenbrockFun
from AckleyFun import AckleyFun
from RastriginFun import RastriginFun	
from SineEnvFun import SineEnvFun
from RanaFunConstraint import RanaFunConstraint
from Binary2continuous import Binary2continuous

from save_performance import save_performance


# Updates by one step the vectors [0,1,1,0...,  0,0,...,0,1,0,...] for n_sessions parallel sessions
def play_game(objective_function, dim, true_dim, precision, amplitude, n_sessions, actions,state_next,states,prob, step, total_score, min_random_thresh):
	terminal = step == dim
	for i in range(n_sessions):
		
		if np.random.rand() < min(max(prob[i], min_random_thresh),1-min_random_thresh):
			action = 1
		else:
			action = 0

		actions[i][step-1] = action
		state_next[i] = states[i,:,step-1]

		if (action > 0):
			state_next[i][step-1] = action
		state_next[i][dim + step-1] = 0
		if (step < dim):
			state_next[i][dim + step] = 1		

		# record sessions 
		if not terminal:
			states[i,:,step] = state_next[i]	
	# calculate final score (modified compared to base version)
	if terminal:
		starting_scoring_time = time.time()
		total_score = [objective_function(Binary2continuous(state_next[i,:dim], precision, true_dim,amplitude)) for i in range(np.shape(state_next)[0])]
		scoring_time = time.time()-starting_scoring_time
		print(f"Scoring time : {scoring_time}")	
	return actions, state_next, states, total_score, terminal


def generate_session(dim, true_dim, precision, amplitude, objective_function, agent, n_sessions, min_random_thresh, verbose = 1):	
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	observation_space = 2*dim
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
	actions = np.zeros([n_sessions, len_game], dtype = int)
	state_next = np.zeros([n_sessions,observation_space], dtype = int)
	prob = np.zeros(n_sessions)
	states[:,dim,0] = 1
	step = 0
	total_score = np.zeros([n_sessions])
	pred_time = 0
	play_time = 0
	
	while (True):
		step += 1		
		tic = time.time()
		prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
		pred_time += time.time()-tic
		tic = time.time()
		actions, state_next,states, total_score, terminal = play_game(objective_function, dim, true_dim, precision, amplitude,\
			n_sessions, actions,state_next,states,prob, step, total_score, min_random_thresh)	
		play_time += time.time()-tic
		
		if terminal:
			break
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time))
	return states, actions, total_score
	

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]
	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

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
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

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




if __name__ == "__main__":
	

	N_LAYERS = int(sys.argv[1])
	WIDTH_LAYERS = [int(width) for width in sys.argv[2:2+N_LAYERS]]
	LEARNING_RATE, N_SESSIONS, PERCENTILE, SUPER_PERCENTILE, MIN_RANDOMNESS, ALPHA, PRECISION, \
	FUNCTION, DIMENSION, SEED, MAX_AMPLITUDE, STOPPING_OBJ_VALUE, MAX_RUNNING_TIME,LOCAL_PATH,\
	SAVE_PERF_FILE, SAVE_PERIOD = sys.argv[2+N_LAYERS:]

	print("\n\nReinforcement Learning.\n")

	print(f"\nOptimizing {FUNCTION} function in dimension {DIMENSION}\n")

	# Rmk : the algorithm maximises the target function


	if FUNCTION == "Rana":
		objective_function = RanaFun
	elif FUNCTION == "Rosenbrock":
		def objective_function(x):
			return -1*RosenbrockFun(x)
	elif FUNCTION == "Rastrigin":
		def objective_function(x):
			return -1*RastriginFun(x)
	elif FUNCTION == "Ackley":
		def objective_function(x):
			return -1*AckleyFun(x)
	elif FUNCTION == "SineEnv":
		objective_function = SineEnvFun

	
	random.seed(int(SEED))
	np.random.seed(int(SEED))
	
	width_layers = WIDTH_LAYERS
	learning_rate = float(LEARNING_RATE) #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
	n_sessions = int(N_SESSIONS) #number of new sessions per iteration
	percentile = int(PERCENTILE) #top 100-X percentile we are learning from
	super_percentile = int(SUPER_PERCENTILE) #top 100-X percentile that survives to next iteration
	min_randomness = True if MIN_RANDOMNESS == "True" else False
	alpha = float(ALPHA)
	true_dim = int(DIMENSION)
	precision = int(PRECISION)
	amplitude = float(MAX_AMPLITUDE)
	dim = true_dim*precision
	observation_space = dim*2



	stopping_obj_value = float(STOPPING_OBJ_VALUE)
	max_running_time = float(MAX_RUNNING_TIME)
	num_generations = 10000000000

	save_period = int(SAVE_PERIOD)
	save_perf_file = os.path.join(LOCAL_PATH,SAVE_PERF_FILE)
	
	archive_folder_path = "archive_files_RL"

	n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
				#such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
				



	if min_randomness:
		min_random_thresh = min(1- alpha**(1/float(dim)),0.45)
		print(f"\nMinimum randomness threshold = {min_random_thresh}\n")
	else:
		min_random_thresh = 0


	len_game = dim
	state_dim = (observation_space,)

	INF = 1000000


	#Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
	#I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
	#It is important that the loss is binary cross-entropy if alphabet size is 2.

	model = keras.Sequential()
	for width in width_layers:
		model.add(keras.layers.Dense(width,  activation="relu"))
	model.add(keras.layers.Dense(1, activation="sigmoid"))
	model.build((None, observation_space))
	model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate)) #Adam optimizer also works well, with lower learning rate

	print(model.summary())


	"""# Running the experiment"""


	super_states =  np.empty((0,len_game,observation_space), dtype = int)
	super_actions = np.array([], dtype = int)
	super_rewards = np.array([])
	sessgen_time = 0
	fit_time = 0
	score_time = 0



	myRand = random.randint(0,1000) #used in the filename

	starting_time = time.time()

	for i in range(num_generations): #1000000 generations should be plenty
		print(f"\nStart session {i}\n")
		generation_starting_time = time.time()
		#generate new sessions
		#performance can be improved with joblib
		tic = time.time()

		sessions = generate_session(dim, true_dim, precision, amplitude,\
			 objective_function, model,n_sessions,min_random_thresh,0) #change 0 to 1 to print out how much time each step in generate_session takes 
		sessgen_time = time.time()-tic
		tic = time.time()
		
		states_batch = np.array(sessions[0], dtype = int)
		actions_batch = np.array(sessions[1], dtype = int)
		rewards_batch = np.array(sessions[2])
		states_batch = np.transpose(states_batch,axes=[0,2,1])

		
		states_batch = np.append(states_batch,super_states,axis=0)

		if i>0:
			actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
		rewards_batch = np.append(rewards_batch,super_rewards)
			
		randomcomp_time = time.time()-tic 
		tic = time.time()

		elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
		select1_time = time.time()-tic

		tic = time.time()
		super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
		select2_time = time.time()-tic
		
		tic = time.time()
		super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
		super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
		select3_time = time.time()-tic
		
		tic = time.time()
		model.fit(elite_states, elite_actions) #learn from the elite sessions
		fit_time = time.time()-tic
		
		tic = time.time()
		
		super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
		super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
		super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
		
		rewards_batch.sort()
		mean_all_reward = np.mean(rewards_batch[-100:])	
		mean_best_reward = np.mean(super_rewards)	

		score_time = time.time()-tic
		
		print("\n" + str(i) +  ". Best scores: " + str(np.flip(np.sort(super_rewards))[:10]))
		
		#uncomment below line to print out how much time each step in this loop takes. 
		# Rmk : score_time is not what you would (reasonably) expect it to be
		#print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
		
		generation_time = time.time()-generation_starting_time
		print(	"Mean of the best rewards: " + str(mean_best_reward) + "\nGeneration total time: "+str(generation_time) ) 
		
		save_performance(max(super_rewards), time.time() - starting_time,save_period,save_perf_file)

		if stopping_obj_value < max(super_rewards) :
			print("Objective reached")
			break
		if time.time() - starting_time > max_running_time:
			print("Time limit exceeded")
			break

		# Old way of saving results
		"""
		if (i%20 == 1): #Write all important info to files every 20 iterations
			with open(ARCHIVE_FOLDER_PATH+'/best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
				pickle.dump(super_actions, fp)
			with open(ARCHIVE_FOLDER_PATH+'/best_species_txt_'+str(myRand)+'.txt', 'w') as f:
				for item in super_actions:
					f.write(str(item))
					f.write("\n")
			with open(ARCHIVE_FOLDER_PATH+'/best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
				for item in super_rewards:
					f.write(str(item))
					f.write("\n")
			with open(ARCHIVE_FOLDER_PATH+'/best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(mean_all_reward)+"\n")
			with open(ARCHIVE_FOLDER_PATH+'/best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(mean_best_reward)+"\n")
		if (i%200==2): # To create a timeline, like in Figure 3
			with open(ARCHIVE_FOLDER_PATH+'/best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(super_actions[0]))
				f.write("\n")
		"""

		


