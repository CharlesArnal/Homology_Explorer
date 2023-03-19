
import math
import numpy as np

import matplotlib.pyplot as plt



# hypercube [-32.768, 32.768] 
# minimize (min = 0)
# takes a list as an input,  adapts to the dimension automatically
def AckleyFun(x):

    a = 20
    b = 0.2
    c = 2
    x = np.array(x)
    n = len(x)  
    return -a*np.exp(-b*np.sqrt(np.sum(x*x)/n)) - np.exp(np.sum(np.cos(c*x))/n)+a+np.exp(1)
    


if __name__ == "__main__":
    x = np.linspace(-30,30,300)
    y = np.linspace(-30,30,300)
    X,Y = np.meshgrid(x,y)

    Z = np.array([[AckleyFun(np.array([xi,yi])) for xi,yi in zip(xj,yj)] for xj,yj in zip(X,Y) ]) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    print(f"\n{AckleyFun([1,2,10,1])}")

