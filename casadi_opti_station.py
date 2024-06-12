from casadi import *
from ode import ode_funCW
from utils import plot_solution3_convex_hull, filter_path_na, compute_time_intervals, linear_initial_path, num2str
from os import getcwd, mkdir, listdir
from os.path import join, exists
from sys import argv
from initial_conditions import convex_hull_station
from constraints import *
import numpy as np

def ocp_parallel(args):
    """
    parallel wrpper for ocp_station_knot()

    arg: list of args for ocp_station_knot
    """
    save_dir, filestr, thrust_limit, kw, pw, fw, closest_knot_bool, view_distance_str, input_local = args

    ocp_station_knot(save_folder=save_dir,
                     save_path=filestr,
                     thrust_limit=thrust_limit,
                     k_weight=kw,
                     p_weight=pw,
                     f_weight=fw,
                     closest_knot=closest_knot_bool,
                     view_distance=view_distance_str,
                     local=input_local
                     )

def ocp_station_knot(meshdir=join(getcwd(), 'model', 'convex_detailed_station'),
                     save_folder=join(getcwd(), 'ocp_paths', 'default'),
                     save_path=None,
                     view_distance='1.5m',
                     local=False,
                     show=False,
                     thrust_limit=0.2,
                     min_station_distance=1.0,
                     k_weight=1,
                     p_weight=1,
                     f_weight=1,
                     closest_knot=False,
                     knot_cost_normalization=100,
                     fuel_cost_normalization=1e-5
                     ):
    """casadi ocp program for space station proximity operations with enforced knotpoints

    Args:
        meshdir (path, optional): directory to find mesh pieces. Defaults to join(getcwd(), 'model', 'convex_detailed_station').
        save_folder (path, optional): directory to save paths. Defaults to join(getcwd(), 'ocp_paths', 'final').
        save_path (path, optional): save file name. Defaults to None.
        view_distance (str, optional): view distance to retrieve knot points from. Defaults to '1.5m'.
        local (bool, optional): using locally generated viewpoints. Defaults to False.
        show (bool, optional): True to show final solution. Defaults to False.
        thrust_limit (float, optional): thrust limit (Newtons) to impose on actions. Defaults to 0.2.
        min_station_distance (float, optional): constrain keepout distance for convex station hull. Defaults to 1.0.
        k_weight (int, optional): knot weight in cost function. Defaults to 1.
        p_weight (int, optional): path length weight in cost function. Defaults to 1.
        f_weight (int, optional): fuel cost weight in cost function. Defaults to 1.
        closest_knot (bool, optional): If true, match knot point to closest in range of knot points. Defaults to False.
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
    # start = knots[0,:3] # get initial position
    start = np.array([2,3,4.2])

    knots[0,:3] = start

    velocity = 0.2
    n_timesteps = 400
    dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    print('Path Duration: ', np.sum(dt), flush=True)
    n_timesteps = len(dt)
    # goal_config_weight = 1
    knot_cost_weight = k_weight
    path_cost_weight = p_weight
    fuel_cost_weight = f_weight
    initial_path = linear_initial_path(knots[:,:3], knot_idx, dt)

    obs, n_states, n_inputs, g0, Isp = convex_hull_station()

    ## define ode
    f = ode_funCW(n_states, n_inputs)

    ## instantiate opti stack
    # print('Setting up Optimization Problem...', flush=True)
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    ## constrain dynamics
    # print('Constraining Dynamics...', flush=True)
    integrate_runge_kutta(X, U, dt, f, opti)

    ## constrain start pose
    min_start_distance = 0.1
    opti.subject_to(sumsqr(X[0,:3].T - knots[0,:3]) <= min_start_distance**2)

    ## constrain thrust limits
    # print('Imposing Thrust Limits...', flush=True)
    opti.subject_to(sum1(U**2) <= thrust_limit**2)

    ## cost function
    # print('Initializaing Cost Function...', flush=True)
    fuel_cost = compute_fuel_cost(U, dt)

    ## knot cost function
    # close_knot_idx = extract_knot_idx(X, opti, knots, knot_idx)
    knot_cost = compute_knot_cost(X, knots, knot_idx, closest=closest_knot)

    ## Path length cost
    path_cost = compute_path_cost(X)

    cost = fuel_cost_weight * fuel_cost / fuel_cost_normalization + knot_cost_weight * knot_cost / knot_cost_normalization + path_cost_weight * path_cost# + goal_config_weight * goal_cost

    # add cost to optimization problem
    opti.minimize(cost)

    # convex hull obstacle
    print('Enforcing Convex Hull Obstacle...', flush=True)
    # enforce convex hull constraint for each obstacle in obs list
    for o in obs:
        normals, points = o
        enforce_convex_hull(normals, points, opti, X, min_station_distance)

    # warm start problem with linear interpolation
    # print('Setting up Warm Start...', flush=True)
    opti.set_initial(X[:,:3], initial_path[:,:3])
    # opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    # print('Setting up Solver...', flush=True)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
    opti.solver('ipopt', opts)

    # solve problem
    print('Solving OCP...', flush=True)
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    # save path and actions
    if save_path is not None: print('Saving Solution: ', join(save_folder, save_path), flush=True)
    else: print('Saving Solution: ', join(save_folder, view_distance), flush=True)

    if not exists(save_folder): mkdir(save_folder)

    if save_path is None:
        if local: save_path = join(save_folder, view_distance + '_local')
        else: save_path = join(save_folder, view_distance)
    # thrust_str = str(thrust_limit//1)[0] + '_' + str(((thrust_limit*10)%10)//1)[0] + str(((thrust_limit*100)%10)//1)[0]
    np.savetxt(join(save_folder, save_path + '_X.csv'), x_opt)
    np.savetxt(join(save_folder, save_path + '_U.csv'), u_opt)
    np.savetxt(join(save_folder, save_path + '_t.csv'), np.insert(np.cumsum(dt),0,0))


if __name__ == "__main__":
    # example: python casadi_opti_station.py 0.2 1 1 1
    # read in thrust limit

    if len(argv) > 1: thrust_limit_input = float(argv[1])
    else: thrust_limit_input = 0.2

    ## READ IN KNOT WEIGHT
    if len(argv) > 2: knot_cost_weight = float(argv[2])
    else: knot_cost_weight = 1
    k_str = num2str(knot_cost_weight) # parse

    ## READ IN PATH WEIGHT
    if len(argv) > 3: path_cost_weight = float(argv[3])
    else: path_cost_weight = 1
    p_str = num2str(path_cost_weight) # parse

    ## READ IN FUEL WEIGHT
    if len(argv) > 4: fuel_cost_weight = float(argv[4])
    else: fuel_cost_weight = 1
    f_str = num2str(fuel_cost_weight)

    if len(argv) > 5: view_distance = argv[5] + 'm'
    else: view_distance='1.5m'

    if len(argv) > 6 and argv[6] == '-l': local_input=True
    else: local_input=False

    if len(argv) > 7: save_dir_input = join(getcwd(), 'ocp_paths', argv[7])
    else: save_dir_input=join(getcwd(), 'ocp_paths', 'default')

    # save_dir_input = 'thrust_test_k_' + k_str + '_p_' + p_str + '_f_' + f_str

    ocp_station_knot(save_folder=save_dir_input, view_distance=view_distance, local=local_input, thrust_limit=thrust_limit_input, k_weight=knot_cost_weight, p_weight=path_cost_weight, f_weight=fuel_cost_weight)

    # old parsing code
    # if str(knot_cost_weight)[-1] == '0': k_str = str(knot_cost_weight)[:-2]
    # else: k_str = str(knot_cost_weight)[:-2] + '_' + str(knot_cost_weight)[-1]
    # if str(path_cost_weight)[-1] == '0': p_str = str(path_cost_weight)[:-2]
    # else: p_str = str(path_cost_weight)[:-2] + '_' + str(path_cost_weight)[-1]
    # if str(fuel_cost_weight)[-1] == '0': f_str = str(fuel_cost_weight)[:-2]
    # else: f_str = str(fuel_cost_weight)[:-2] + '_' + str(fuel_cost_weight)[-1]
