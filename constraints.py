from casadi import dot, fmax, sumsqr, sum1, sum2, sqrt
from numpy.linalg import norm

def enforce_convex_hull(normals, points, opti, X, min_station_distance):
    """
    create constraint formulation for opti stack for a convex hull given face normals and centroids
    normals - list of 3d vectors (dir) np.ndarray(num_normals, 3)
    centroids - list of 3d vectors (position) np.ndarray(num_centroids, 3)
    opti - opti stack variable
    X - state variable MX.shape(num_timesteps, 6)
    """
    num_normals = normals.shape[0]
    num_timesteps = X.shape[0]

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

def compute_knot_cost(X, knots, knot_idx):
    """
    compute distance between knot points and path X (enforces position - first three states, but not velocity)

    Inputs:
        X - state matrix size(n_timesteps+1, n_states)
        knots - knot points np.ndarray(n_knots, n_states
        knot_idx - index of X corresponding with each knot point

    Returns: 
        knot_cost - cumulative distance between knot points and path X
    """

    # knot_cost = 0
    # for i, k in enumerate(knot_idx):
        # knot_cost += sumsqr(X[k,:3].T - knots[i,:3])

    knot_cost = sumsqr(X[knot_idx, :3] - knots[:, :3])

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