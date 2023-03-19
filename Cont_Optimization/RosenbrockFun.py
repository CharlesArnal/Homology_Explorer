
import math
import numpy as np
import matplotlib.pyplot as plt


# not constrained, mostly interesting from -6 to 6
# minimize (min = 0)

# takes a list as an input,  adapts to the dimension automatically
def RosenbrockFun(x):
    x = np.array(x,dtype=float)
    n = len(x)
    x_i = x[:-1]
    x_ip1 = x[1:]
    return np.sum(100*(x_ip1-x_i**2)**2 + (1-x_i)**2)
    

if __name__ == "__main__":
    x = np.linspace(-50,50,30)
    y = np.linspace(-50,50,30)
    X,Y = np.meshgrid(x,y)

    Z = np.array([[RosenbrockFun(np.array([xi,yi])) for xi,yi in zip(xj,yj)] for xj,yj in zip(X,Y) ]) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, np.log(Z), 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    print(f"\n{RosenbrockFun([1,2,10,1])}")