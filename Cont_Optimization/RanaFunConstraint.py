
import math
import numpy as np

# takes a numpy array as an input
def RanaFunConstraint(x):
    if np.max(np.abs(x)) > 500:
        return -1
    else:
        return 1
        
#print(RanaFunConstraint(np.array([120,-550])))