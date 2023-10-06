import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
 
import tensorflow as tf
from utilities import ad_hoc_median

# NOTE
from memory_profiler import profile

import numpy as np

from utilities import Smith_Thom_bound

print(Smith_Thom_bound(2, 15))
print(Smith_Thom_bound(2, 20))
print(Smith_Thom_bound(2, 25))

#3 8
a = len("0 1 1 1 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 0 0 0 1 0 1 1 0 0 0 0 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 1 0 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 0 0 0 0 1 1 0 0 1 0 1 1 0".split())

#4 5
b = len("0 0 0 1 0 0 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 1 1 1 0 0 1 0 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 1 0 0".split())

# Harnack 15 (test)

c = len("0 1 0 1 0 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 0 1 1 1 0 1 1 1 1 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 1 1 0 0 1 0 1 0 1 0 0 1 1 0 0 1 1 0 0 0 1 1 1 1 1 0 1 0 1 0 0 0 0 0 0 1 1 0 1 1 1 1 0 0 0 0 0 1 1 0 0 1 1 0 1 0 0 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 1 0 0 1 0 0".split())

# Harnack 25

d = len("1 1 0 1 1 0 0 0 1 1 1 0 0 0 1 1 1 1 0 1 1 1 0 0 1 0 0 0 1 0 1 0 1 1 0 0 0 1 1 0 1 0 1 0 1 1 0 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 1 0 1 0 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 1 1 0 0 0 1 0 0 1 1 0 0 0 1 1 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 0 0 1 0 0 1 0 1 0 0 1 1 1 1 0 1 0 0 1 0 1 0 0 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 0".split())

print(a)
print(b)
print(c)
print(d)