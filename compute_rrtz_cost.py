import numpy as np
from numpy.linalg import norm
from os import listdir, getcwd
from os.path import join
from utils import filter_path_na

def import_knot_file(view_distance, local):
    """import knot file with corresponding view_distance and locality

    Args:
        view_distance (str): viewpoint generation distance
        local (bool): locality of path generation
    """
    for file in listdir(join(getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=join(getcwd(), 'ccp_paths', file)
                break
    print('Importing Knot File: ', knotfile)

    path = np.loadtxt(knotfile, delimiter=',') # (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans

    return knots

def compute_line_costCW(p1, p2, dt, N=100):
    """generate cost to traverse line given CW disturbances

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        dt (float): time step
        N (int): number of discretization points
    """
    m_I = 5.75 # inspector mass (kg)
    mu = 3.986e14 # standard gravitational parameter
    a = 6.6e6 # international space station orbit radius
    n = np.sqrt(mu/a**3) # orbital rate of target craft
    g0 = 9.81 # acceleration due to gravity
    Isp = 80 # specific impulse of rocket engine
    m = 5.75 # mass of agent
    # list of points
    x = np.zeros((N,3))
    for i in range(3):
        x[:,i] = np.linspace(p1[i], p2[i], N)

    xdot = (p2-p1)/dt

    xddot = np.zeros((N, 3))
    xddot[:,0] = 3*n**1*x[:,0] + 2*n*xdot[1]
    xddot[:,1] = -2*n*xdot[0]
    xddot[:,2] = -n**2 * x[:,2]

    fuel_cost = np.sum((np.sqrt(np.sum(xddot**2, axis=1))/g0/Isp)*dt/N) * 1000 # convert kg to grams

    return fuel_cost

def compute_fuel_cost(knots, velocity, g0=9.81, Isp=60, m=5.75):
    """compute fuel cost from knots

    Args:
        knots (np.ndarray): size(N,6) knot points
        velocity (float): velocity of path following vehicle
        g0 (float): acceleration due to gravity
        Isp (float): specific impulse of rocket engine
        m (float): mass of agent
    """

    N = knots.shape[0]
    fuel_cost = 0
    dt = norm(knots[1,:3] - knots[0,:3])/velocity
    fuel_cost +=compute_line_costCW(knots[0,:3], knots[1,:3], dt)
    for i in range(N-2):
        # get velocity vectors
        v1 = knots[i+1,:3] - knots[i,:3]
        v1 = v1/norm(v1) * velocity
        v2 = knots[i+2,:3] - knots[i+1,:3]
        v2 = v2/norm(v2) * velocity

        # delta-v needed
        dv = norm(v2-v1)

        # thrust needed
        thrust = dv * m

        # compute fuel use per delta v
        fuel_cost += thrust / Isp / g0 * 1000
        # fuel_cost += (np.exp(dv/Isp/g0)-1)*m * 1000

        # compute cw cost
        dt = norm(v2)/velocity
        fuel_cost += compute_line_costCW(knots[i+1,:3], knots[i+2,:3], dt)

    return fuel_cost

def compute_path_time(knots, velocity):
    """compute time to traverse path of knots with velocity with impulsive maneuvers

    Args:
        knots (np.ndarray(N, 6)): knot points (first three) and view direction (last three)
        velocity (): agent velocity
    """
    # get vectors between states
    dk = np.diff(knots[:,:3], axis=0)

    # get distance between states
    dk_norm = norm(dk, axis=1)

    # find time along each subtrajectory
    dt = dk_norm/velocity

    # sum times
    return np.sum(dt)

def evaluate_costs():
    """evaluate costs of all knot files
    """
    velocity=0.1

    fuel_costs = ['FUEL COSTS and path time Not Local | Local']
    fuel_costs.append('\ndistance | fuel cost, path time | fuel cost, path time')
    knot_file_names = [str(x)+'m' for x in np.arange(1.5, 5, 0.5)]
    for knot_file in knot_file_names:
        fuel_costs.append('\n'+knot_file+' ')
        knotpoints = import_knot_file(view_distance=knot_file, local=False)
        fuel_costs.append(compute_fuel_cost(knotpoints, velocity))
        fuel_costs.append(', ')
        fuel_costs.append(compute_path_time(knotpoints, velocity))
        fuel_costs.append(' | ')
        knotpoints = import_knot_file(view_distance=knot_file, local=True)
        fuel_costs.append(compute_fuel_cost(knotpoints, velocity))
        fuel_costs.append(', ')
        fuel_costs.append(compute_path_time(knotpoints, velocity))
        fuel_costs.append(' | ')

    print(*fuel_costs)

if __name__ == "__main__":
    evaluate_costs()