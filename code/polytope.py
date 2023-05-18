from mip import Model, xsum, minimize
import mip
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import numpy as np

def is_inside_lp(archetypes, point, delta=0):
    """
    polytope: numpy array of shape (n,m) where n is the number of dimensions of the space and m is the number of vertices in the polytope
    point: numpy array of shape (n,) where n is the number of dimensions of the space
    Returns True if the point is in the interior of the polytope, False otherwise
    """

    model = Model()
    model.verbose = 0
    x = [model.add_var(name='x'+str(i)) for i in range(archetypes.shape[1])]
    for i in range(archetypes.shape[0]):
        model += xsum(archetypes[i,j]*x[j] for j in range(archetypes.shape[1])) == point[i]
    model += xsum(x) >= 1
    model += xsum(x) <= 1+delta
    
    for i in range(len(x)):
        model += x[i] >= 0 
    model.objective = minimize(xsum(x))
    model.optimize()
    # print(model.status)
    return model.status == mip.OptimizationStatus.OPTIMAL


def is_inside(polytope, point):
    # Compute the convex hull of the principal components
    hull = ConvexHull(polytope)

    # Check if the projected point is inside the convex hull of the principal components
    for eq in hull.equations:
        if np.dot(eq[:-1], point) + eq[-1] > 0:
            return False
    return True