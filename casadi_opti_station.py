from casadi import *
from ode import ode_funCW
from utils import plot_solution3_convex_hull, filter_path_na, compute_time_intervals, linear_initial_path
from os import getcwd, mkdir, listdir
from os.path import join, exists
from sys import argv
from initial_conditions import convex_hull_station
from constraints import *
import numpy as np

def ocp_station_knot(meshdir=join(getcwd(), 'model', 'convex_detailed_station'),
                     # knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                     # save_dir='thrust_test_k_1_p_1_f_1',
                     save_folder=join(getcwd(), 'ocp_paths', 'final'),
                     save_path=None,
                     view_distance='1.5m',
                     local=False,
                     # save_file='1.5m',
                     show=False,
                     thrust_limit=0.2,
                     min_station_distance=1.0,
                     k_weight=1,
                     p_weight=1,
                     f_weight=1
                     ):
    """
    ocp_station with knot points
    """
    print('Importing Initial Conditions...', flush=True)

    for file in listdir(join(getcwd(), 'ccp_paths')):
        if str(view_distance) == file[:4]:
            if ((file[5] == 'l') and local) or ((file[5] != 'l') and not local):
                knotfile=join(getcwd(), 'ccp_paths', file)
                break
    print('Importing Knot File: ', knotfile)

    path = np.loadtxt(knotfile, delimiter=',') # (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans

    velocity = 0.2
    n_timesteps = 400
    dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    print('Path Duration: ', np.sum(dt), flush=True)
    n_timesteps = len(dt)
    goal_config_weight = 1
    knot_cost_weight = k_weight
    path_cost_weight = p_weight
    fuel_cost_weight = f_weight
    initial_path = linear_initial_path(knots[:,:3], knot_idx, dt)

    obs, n_states, n_inputs, g0, Isp = convex_hull_station()

    ## define ode
    f = ode_funCW(n_states, n_inputs)

    ## instantiate opti stack
    print('Setting up Optimization Problem...', flush=True)
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    ## constrain dynamics
    print('Constraining Dynamics...', flush=True)
    integrate_runge_kutta(X, U, dt, f, opti)

    ## constrain thrust limits
    print('Imposing Thrust Limits...', flush=True)
    opti.subject_to(sum1(U**2) <= thrust_limit**2)

    ## cost function
    print('Initializaing Cost Function...', flush=True)
    fuel_cost = compute_fuel_cost(U, dt)

    ## knot cost function
    knot_cost = compute_knot_cost(X, knots, knot_idx)

    ## Path length cost
    path_cost = compute_path_cost(X)

    cost = fuel_cost_weight * fuel_cost + knot_cost_weight * knot_cost + path_cost_weight * path_cost# + goal_config_weight * goal_cost

    # add cost to optimization problem
    opti.minimize(cost)

    # convex hull obstacle
    print('Enforcing Convex Hull Obstacle...', flush=True)
    # enforce convex hull constraint for each obstacle in obs list
    for o in obs:
        normals, points = o
        enforce_convex_hull(normals, points, opti, X, min_station_distance)

    # warm start problem with linear interpolation
    print('Setting up Warm Start...', flush=True)
    opti.set_initial(X[:,:3], initial_path[:,:3])
    # opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    print('Setting up Solver...', flush=True)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
    opti.solver('ipopt', opts)

    # solve problem
    print('Solving OCP...', flush=True)
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    # save path and actions
    print('Saving Solution...', flush=True)
    if not exists(save_folder): mkdir(save_folder)

    if save_path is None:
        if local: save_path = join(save_folder, view_distance + '_local')
        else: save_path = join(save_folder, view_distance)
    # thrust_str = str(thrust_limit//1)[0] + '_' + str(((thrust_limit*10)%10)//1)[0] + str(((thrust_limit*100)%10)//1)[0]
    np.savetxt(save_path + '_X.csv', x_opt)
    np.savetxt(save_path + '_U.csv', u_opt)
    np.savetxt(save_path + '_t.csv', np.insert(np.cumsum(dt),0,0))

    if show: 
        meshfiles = []
        for i in range(15):
            meshfiles.append(join(meshdir, str(i) + '.stl'))
        # meshfile = join(filename, 'mercury_convex.stl')
        ## knot cost function
        knot_cost = 0
        for i, k in enumerate(knot_idx):
            knot_cost += sumsqr(x_opt[k,:3].T - knots[i,:3])
        fuel_cost = np.sum(u_opt**2)/g0/Isp
        path_cost = np.sum((x_opt[1:,:] - x_opt[:-1,:])**2)
        print('Knot Cost = ', knot_cost, flush=True)
        print('Fuel Cost = ', fuel_cost, flush=True)
        print('Path Cost = ', path_cost, flush=True)
        print('Plotting Solution')
        plot_solution3_convex_hull(x_opt, u_opt, meshfiles, dt, save_fig_file=join('path_figures', 'ocp'), station=True)


if __name__ == "__main__":
    # example: python casadi_opti.py 0.2 1 1 1
    # read in thrust limit
    if len(argv) > 1: thrust_limit_input = float(argv[1])
    else: thrust_limit_input = 0.2

    # read in cost weights
    if len(argv) > 2: knot_cost_weight = float(argv[2])
    else: knot_cost_weight = 1

    # parse weight for string (assume weights only go to first decimal place
    if str(knot_cost_weight)[-1] == '0': k_str = str(knot_cost_weight)[:-2]
    else: k_str = str(knot_cost_weight)[:-2] + '_' + str(knot_cost_weight)[-1]

    if len(argv) > 3: path_cost_weight = float(argv[3])
    else: path_cost_weight = 1
    if str(path_cost_weight)[-1] == '0': p_str = str(path_cost_weight)[:-2]
    else: p_str = str(path_cost_weight)[:-2] + '_' + str(path_cost_weight)[-1]

    if len(argv) > 4: fuel_cost_weight = float(argv[4])
    else: fuel_cost_weight = 1
    if str(fuel_cost_weight)[-1] == '0': f_str = str(fuel_cost_weight)[:-2]
    else: f_str = str(fuel_cost_weight)[:-2] + '_' + str(fuel_cost_weight)[-1]

    if len(argv) > 5: view_distance = argv[5] + 'm'
    else: view_distance='1.5m'
    # print('View distance: ', view_distance)

    if len(argv) > 6 and argv[6] == '-l': local_input=True
    else: local_input=False
    # print('Local? ', local_input)

    # save_dir_input = 'thrust_test_k_' + k_str + '_p_' + p_str + '_f_' + f_str

    ocp_station_knot(view_distance=view_distance, local=local_input, thrust_limit=thrust_limit_input, k_weight=knot_cost_weight, p_weight=path_cost_weight, f_weight=fuel_cost_weight)
