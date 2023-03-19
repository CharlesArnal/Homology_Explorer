
from cmath import sqrt
import numpy as np
import networkx as nx
import time

from numba import njit


import math



def obj_fun_2_1(state,N):
	"""
	Calculates the reward for a given word. 
	This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
	:param state: the first MYN letters of this param are the word that the neural network has constructed.
	:returns: the reward (a real number). Higher is better, the network will try to maximize this.
	
	The function must be maximised, the conjecture is disproven at 0
	"""	
	
	#Example reward function, for Conjecture 2.1
	#Given a graph, it minimizes lambda_1 + mu.
	#Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
	#There is a lot of run-to-run variance.
	#Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.
	
	#Construct the graph 
	#N = int(math.floor((1+sqrt(1+8*len(state)))/2+0.1))
	INF = 1000000
	G= nx.Graph()
	G.add_nodes_from(list(range(N)))
	count = 0
	for i in range(N):
		for j in range(i+1,N):
			if state[count] == 1:
				G.add_edge(i,j)
			count += 1
	
	#G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
	if not (nx.is_connected(G)):
		return -INF
		
	#Calculate the eigenvalues of G
	evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
	evalsRealAbs = np.zeros_like(evals)
	for i in range(len(evals)):
		evalsRealAbs[i] = abs(evals[i])
	lambda1 = max(evalsRealAbs)
	
	#Calculate the matching number of G
	maxMatch = nx.max_weight_matching(G)
	mu = len(maxMatch)
		
	#Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
	#We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
	myScore = math.sqrt(N-1) + 1 - lambda1 - mu
		
	return myScore




def bfs(Gdeg,edgeListG,N):
	#simple breadth first search algorithm, from each vertex
	
	distMat1 = np.zeros((N,N))
	conn = True
	for s in range(N):
		visited = np.zeros(N,dtype=np.int8)
	 
		# Create a queue for BFS. Queues are not suported with njit yet so do it manually
		myQueue = np.zeros(N,dtype=np.int8)
		dist = np.zeros(N,dtype=np.int8)
		startInd = 0
		endInd = 0

		# Mark the source node as visited and enqueue it 
		myQueue[endInd] = s
		endInd += 1
		visited[s] = 1

		while endInd > startInd:
			pivot = myQueue[startInd]
			startInd += 1
			
			for i in range(Gdeg[pivot]):
				if visited[edgeListG[pivot][i]] == 0:
					myQueue[endInd] = edgeListG[pivot][i]
					dist[edgeListG[pivot][i]] = dist[pivot] + 1
					endInd += 1
					visited[edgeListG[pivot][i]] = 1
		if endInd < N:
			conn = False #not connected
		
		for i in range(N):
			distMat1[s][i] = dist[i]
		
	return distMat1, conn

jitted_bfs = njit()(bfs)


# Takes as input a vector of length int(N*(N-1)/2) 
# TODO check that we are not sending a longer vector
def obj_fun_2_3_pre_jitted(state, N):
	"""
	Reward function for Conjecture 2.3, using numba
	With n=30 it took a day to converge to the graph in figure 5, I don't think it will ever find the best graph
	(which I believe is when the neigbourhood of that almost middle vertex is one big clique).
	(This is not the best graph for all n, but seems to be for n=30)

	Conjecture disproved if >=0
	"""
	INF = 1000000
	#N = int(math.floor((1+sqrt(1+8*len(state)))/2+0.1))
	#construct the graph G
	adjMatG = np.zeros((N,N),dtype=np.int8) #adjacency matrix determined by the state
	edgeListG = np.zeros((N,N),dtype=np.int8) #neighbor list
	Gdeg = np.zeros(N,dtype=np.int8) #degree sequence
	count = 0
	for i in range(N):
		for j in range(i+1,N):
			if state[count] == 1:
				adjMatG[i][j] = 1
				adjMatG[j][i] = 1
				edgeListG[i][Gdeg[i]] = j
				edgeListG[j][Gdeg[j]] = i
				Gdeg[i] += 1
				Gdeg[j] += 1
			count += 1
	
	distMat, conn = jitted_bfs(Gdeg,edgeListG,N)
	#G has to be connected
	if not conn:
		return -INF
		
	diam = np.amax(distMat)
	sumLengths = np.zeros(N,dtype=np.int8)
	sumLengths = np.sum(distMat,axis=0)		
	evals =  np.linalg.eigvalsh(distMat)
	evals = -np.sort(-evals)
	proximity = np.amin(sumLengths)/(N-1.0)

	ans = -(proximity + evals[math.floor(2*diam/3) - 1])

	return ans

obj_fun_2_3 =   njit()(obj_fun_2_3_pre_jitted)



if __name__ == "__main__":

	N = 30
	M = 10000
	dim = int(N*(N-1)/2) 
	print("start")
	print(obj_fun_2_3(np.random.randint(0,2, size = dim), N))
	print("start 2")
	t1_0 = time.time()
	for index, x in enumerate(np.random.randint(0,2, size = [M, dim])):
		obj_fun_2_3(x, N)
	print(f" time 1 = {time.time()- t1_0}")
	

	
	t2_0 = time.time()
	for index, x in enumerate(np.random.randint(0,2, size = [M, dim])):
		obj_fun_2_3(x, N)
	print(f"time 2 = {time.time()- t2_0}")