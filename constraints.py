from casadi import dot, fmax, sumsqr, sum1, sum2, sqrt, fmin, mtimes
from numpy.linalg import norm
import numpy as np

def enforce_convex_hull(normals, points, opti, X, min_station_distance):
    """enforce the convex hull obstacle given face noramls and centroids on the opti stack

    Args:
        normals (np.ndarray(num_normals, 3)): list of 3d vectors with convex hull face normals
        points (np.ndarray(num_centroids, 3)): list of 3d vectors containing triangular mesh face centroids
        opti (Opti): opti stack object for ocp
        X (MX(num_timesteps, 6)): state vector 
        min_station_distance (float): minimum distance of state vector configurations from the convex hull being enforced
    """
    # get dimensions
    num_timesteps = X.shape[0]
    num_normals = normals.shape[0]

    # for each state timestep we apply the convex hull keepout constraint
    for j in range(num_timesteps):

        # create a convex hull keepout constraint for each time step:
        dot_max = -1 # we can instantiate the max dot product as -1 because dot products less than zero do not satisfy the constraint (we take maximum)
        for i in range(num_normals):

            # first retrieve parameters for each face instance
            n = normals[[i],:] # face normal
            n = n/norm(n) # normalize normal
            p = points[[i],:] # centroid corresponding to face normal
            x = X[j,:3] # state at timestep j (just position)
            r = x-p # vector from face centroid to state position

            # only one dot product must be greater than zero so we take the maximum value
            # of all of them to use as the constraint (for each timestep)
            dot_max = fmax(dot_max, dot(n,r)) # Given convexity, pull out the closest face to x (state)
        
        # if max dot product value is above zero, then constraint is met (only one needs to be greater)
        opti.subject_to(dot_max > min_station_distance)


def enforce_convex_hull_value(normals, points, opti, X, min_station_distance):
    """
    this doesn't work on account of failed assertion_solved() case in OptiNode
    create constraint formulation for opti stack for a convex hull given face normals and centroids
    normals - list of 3d vectors (dir) np.ndarray(num_normals, 3)
    centroids - list of 3d vectors (position) np.ndarray(num_centroids, 3)
    opti - opti stack variable
    X - state variable MX.shape(num_timesteps, 6)
    """
    # normalize face normals
    N = normals/norm(normals, axis=1)

    # compute state-centroid matrix:
    X_cur = np.array(opti.value(X))
    R = np.subtract.outer(X_cur[:,:3], points)[:,range(3),range(3),:].transpose(0,2,1)

    # manual dot product from each normal--state-centroid combination
    # >> create new axis along timesteps, elementwise multiply, then add along spatial dim axis (last one) to take manual dot product
    dot_outer = np.sum(X_cur[np.newaxis,:,:]*R, axis=-1) # (num_timesteps, num_normals)

    # find maximum for each state along second axis (normals axis)
    max_dot_idx = np.argmax(dot_outer, axis=1)

    # now compute the dot product for each timestep with max_dot_idx since we now know which normal-centroid (face) corresponds to each state
    dot = np.sum(X[:,:3]*N[max_dot_idx,:], axis=1) # size(num_timesteps)

    # enforce dot product being larger than a minimum distance value
    opti.subject_to(dot > min_station_distance)

def integrate_runge_kutta(X, U, dt, f, opti):
    """
    integrate forward dynamics - f - using runge kutta 4 integration for each timestep dt for state X and actions U
    
    Inputs:
        X - symbolic matrix size(n_timesteps+1, n_states)
        U - symbolic matrix size(n_timesteps, n_inputs)
        dt - float matrix size(n_timesteps)
        f - function f(X, U)
        opti - casadi optimization problem
    """

    # ensure shapes match
    assert U.shape[0] == X.shape[0]-1
    assert U.shape[0] == len(dt)

    n_timesteps = U.shape[0] # number of timesteps to integrate over

    for k in range(n_timesteps):
        # Runge-Kutta 4 integration
        k1 = f(X[k,:],              U[k,:])
        k2 = f(X[k,:]+dt[k]/2*k1.T, U[k,:])
        k3 = f(X[k,:]+dt[k]/2*k2.T, U[k,:])
        k4 = f(X[k,:]+dt[k]*k3.T,   U[k,:])
        x_next = X[k,:] + dt[k]/6*(k1.T+2*k2.T+2*k3.T+k4.T)
        opti.subject_to(X[k+1,:]==x_next); # close the gaps

    # for one step integration
    # opti.subject_to(X[k+1,:].T == X[k,:].T + dt[k] * f(X[k,:], U[k,:]))

    return

def extract_knot_idx(X, opti, knots, knot_idx):
    """
    extract knot_idx corresponding to closest state vector configurations from casadi symbolic MX

    Inputs:
        X (MX (n_timesteps_1, n_states)): symbolic state vector
        opti (casadi object): optimization problem - opti stack variable
        knots (np.ndarray(n_knots, n_states)): knot points
        knot_idx (np.ndarray(n_knots)): list of initial knot point correspondances 

    Return:
        list of indices corresponding with the closest points along the state vector array to knot points.
    """

    # extract numpy array of current state vector values from the optistack problem
    X_cur = np.array(opti.value(X))

    # search for the state vector configurations closest to the knot points within a given range of original knot_idx's
    start_idx = 0
    close_knot_idx = []
    for ki in range(len(knot_idx)-1):
        # look for closest state vector configuration halfway behind and ahead of current state vector point
        end_idx = (knot_idx[ki] + knot_idx[ki+1])//2+1

        # use the square distance (less computation and still monotonic)
        sq_dist = np.sum((X_cur[start_idx:end_idx,:3] - knots[[ki], :3])**2)
        closest_idx = np.argmin(sq_dist)
        close_knot_idx.append(closest_idx + start_idx)
        start_idx = end_idx

    return close_knot_idx

def compute_knot_cost(X, knots, knot_idx, closest=False):
    """
    compute distance between knot points and path X (enforces position - first three states, but not velocity)

    Inputs:
        X - state matrix size(n_timesteps+1, n_states)
        knots - knot points np.ndarray(n_knots, n_states)
        knot_idx - index of X corresponding with each knot point

    Returns: 
        knot_cost - cumulative distance between knot points and path X
    """

    # knot_cost = 0
    # for i, k in enumerate(knot_idx):
        # knot_cost += sumsqr(X[k,:3].T - knots[i,:3])

    if closest:
        last_idx = 0
        knot_cost = 0
        for ki in range(len(knot_idx)-1):
            closest_dist = np.Inf
            next_idx = (knot_idx[ki] + knot_idx[ki+1])//2+1
            for idx in range(last_idx, next_idx):
                dist = sumsqr(knots[[ki], :3] - X[[idx], :3]) # compare state, look at reshape
                closest_dist = fmin(closest_dist, dist)
            knot_cost += closest_dist
            last_idx = next_idx
    else: knot_cost = sumsqr(X[knot_idx, :3] - knots[:, :3])

    return knot_cost

def compute_path_cost(X):
    """
    compute length of path X

    Inputs:
        X - state matrix size(n_timesteps+1, n_states)

    Returns:
        path_cost - path length of X
    """
    # path_cost as path length seems to return bad gradients:
    # path_cost = sum2(sqrt(sum1((X[1:, :] - X[:-1,:])**2)))

    # instead just use the sum of squares of each path segment:
    path_cost = sumsqr(X[1:, :] - X[:-1, :])

    return path_cost

def compute_fuel_cost(U, dt, g0=9.81, Isp=80):
    """
    compute fuel cost for actions U

    Inputs:
        U - action sequence

    Return:
        fuel_cost - float
    """
    assert U.shape[0] == len(dt) # make sure vectors match dimensions

    n_timesteps = U.shape[0]

    total_impulse = 0
    for k in range(n_timesteps):
        total_impulse += sumsqr(U[k,:]) * dt[k]**2
    fuel_cost = total_impulse/g0**2/Isp**2 # squared fuel cost

    return fuel_cost
