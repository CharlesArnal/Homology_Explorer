
import math
import numpy as np

import matplotlib.pyplot as plt



# takes a list as an input,  adapts to the dimension automatically
# in fact minus the SineEnvFun (that must usually be minimised)
# the max should be fairly close to 0.02*n in x = 0 (semble faux)
# amplitude = 100
def SineEnvFun(x):
    x = np.array(x,dtype=float)
    n = len(x)
    x_i = x[:-1]
    x_ip1 = x[1:]

    return np.sum(np.sin(np.sqrt(x_ip1**2 + x_i**2)-0.5 ) **2 / (0.001*(x_ip1**2+x_i**2)+1)**2 + 0.5)
    

    


if __name__ == "__main__":
    amplitude = 20
    x = np.linspace(-amplitude,amplitude,100)
    y = np.linspace(-amplitude,amplitude,100)
    X,Y = np.meshgrid(x,y)

    Z = np.array([[SineEnvFun(np.array([xi,yi])) for xi,yi in zip(xj,yj)] for xj,yj in zip(X,Y) ]) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    valeurs = []

    plt.show()
    
