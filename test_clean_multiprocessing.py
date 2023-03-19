import subprocess
import math
import numpy as np
import os
import copy
import time

from multiprocessing import Process

def f(t):
    time.sleep(t)
    print("done")

T = 10

p_1 = Process(target=f, args=[T])
p_1.run()
p_2 = Process(target=f, args=[T])
p_2.run()
p_3 = Process(target=f, args=[T])
p_3.run()

