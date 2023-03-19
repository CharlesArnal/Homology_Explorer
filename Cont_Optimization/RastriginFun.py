
import math
import numpy as np
import matplotlib.pyplot as plt


#hypercube [-5.12, 5.12]
# minimize (min = 0)

# takes a list as an input,  adapts to the dimension automatically
def RastriginFun(x):
    x = np.array(x,dtype=float)
    n = len(x)
    A = 10
    return A*n +np.sum(x*x - A*np.cos(2*math.pi*x))

if __name__ == "__main__":
    amplitude = 10
    x = np.linspace(-amplitude,amplitude,100)
    y = np.linspace(-amplitude,amplitude,100)
    X,Y = np.meshgrid(x,y)

    Z = np.array([[RastriginFun(np.array([xi,yi])) for xi,yi in zip(xj,yj)] for xj,yj in zip(X,Y) ]) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    print(f"\n{RastriginFun([1,2,-5,1])}")