from mip import Model, xsum, minimize
import mip
import numpy as np

def is_inside(polytope, point, delta=0):
    """
    polytope: numpy array of shape (n,m) where n is the number of dimensions of the space and m is the number of vertices in the polytope
    point: numpy array of shape (n,) where n is the number of dimensions of the space
    Returns True if the point is in the interior of the polytope, False otherwise
    """

    model = Model()
    model.verbose = 0
    x = [model.add_var(name='x'+str(i)) for i in range(polytope.shape[1])]
    for i in range(polytope.shape[0]):
        model += xsum(polytope[i,j]*x[j] for j in range(polytope.shape[1])) == point[i]
    model += xsum(x) >= 1
    model += xsum(x) <= 1+delta
    
    for i in range(len(x)):
        model += x[i] >= 0 
    model.objective = minimize(xsum(x))
    model.optimize()
    # print(model.status)
    return model.status == mip.OptimizationStatus.OPTIMAL

