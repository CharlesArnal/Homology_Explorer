
import math
import numpy as np

import matplotlib.pyplot as plt



# takes aa list as an input,  adapts to the dimension automatically
def RanaFun(x):
    x = np.array(x)
    n = len(x)
    x_i = x[:-1]
    x_ip1 = x[1:]
    v1 = np.sqrt(np.absolute(x_ip1+x_i+1))
    v2 = np.sqrt(np.absolute(x_ip1-x_i+1))
    return np.sum( x_i*np.cos(v1)*np.sin(v2) + (1+x_ip1)*np.cos(v2)*np.sin(v1) )
    


if __name__ == "__main__":
    x = np.linspace(-500,500,30)
    y = np.linspace(-500,500,30)
    X,Y = np.meshgrid(x,y)

    Z = np.array([[RanaFun(np.array([xi,yi])) for xi,yi in zip(xj,yj)] for xj,yj in zip(X,Y) ]) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

