from casadi import *
import casadi as c
from ode import ode_fun2, ode_fun3, ode_funCW
from matplotlib import pyplot as plt
from utils import plot_solution2, plot_solution3, plot_solution3_convex_hull, filter_path_na, compute_time_intervals, linear_initial_path
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from initial_conditions import *
from tqdm import tqdm
from os import getcwd, mkdir
from os.path import join, exists
from sys import argv
from constraints import *

def ocp_mockup_knot(meshdir=join(getcwd(), 'model', 'convex_detailed_station'), knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'), save_file='1.5m', show=True):
    """
    ocp_station with knot points
    """
    print('Importing Initial Conditions...')
    # three knot points
    knots = np.array([[1.5, -1.0, 0.5],
                      [1.5, 1.0, 0.5],
                      [0.0, 1.0, 0.5],
                      [0.0, -1.0, 0.5]])

    velocity = 0.02
    n_timesteps = 101
    dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    print('Path Duration: ', np.sum(dt))
    n_timesteps = len(dt)
    min_station_distance = 0.5
    goal_config_weight = 1
    knot_cost_weight = 1
    path_cost_weight = 1
    fuel_cost_weight = 1
    thrust_limit = 0.2
    initial_path = linear_initial_path(knots, knot_idx, dt)

    obs, n_states, n_inputs, g0, Isp = convex_hull_mockup()

    ## define ode
    f = ode_funCW(n_states, n_inputs)

    ## instantiate opti stack
    print('Setting up Optimization Problem...')
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    # use constraint to ensure initial and final conditions
    # opti.subject_to(X[0,:].T == x0)
    # opti.subject_to(X[-1,:].T == xf)

    ## constrain dynamics
    print('Constraining Dynamics...')
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

    # for i, k in enumerate(knot_idx):
    #     opti.subject_to(X[k,:].T == knots[i,:])

    ## constrain thrust limits
    print('Imposing Thrust Limits...')
    opti.subject_to(sum1(U**2) <= thrust_limit**2)

    ## cost function
    print('Initializaing Cost Function...')
    fuel_cost = sumsqr(U)/g0/Isp

    ## knot cost function
    knot_cost = 0
    for i, k in enumerate(knot_idx):
        knot_cost += sumsqr(X[k,:3].T - knots[i,:3])

    ## Path length cost
    path_cost = sumsqr(X[1:,:] - X[:-1,:]) # squared path length

    cost = fuel_cost_weight * fuel_cost + knot_cost_weight * knot_cost + path_cost_weight * path_cost# + goal_config_weight * goal_cost

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    # convex hull obstacle
    print('Enforcing Convex Hull Obstacle...')
    for o in obs:
        normals, points = o
        enforce_convex_hull(normals, points, opti, X, min_station_distance)

    # warm start problem with linear interpolation
    print('Setting up Warm Start...')
    opti.set_initial(X[:,:3], initial_path)

    # look at solution at each iteration
    # if visualize:
    #     save_file = os.path.join(os.getcwd(), 'optimization_steps', 'one_obstacle', 'offsetx_' + str(offsetx) + '_offsety_' + str(offsety))
    #     if not os.path.exists(save_file):
    #         os.mkdir(save_file)
    #     opti.callback(lambda i: plot_solution3(opti.debug.value(X), opti.debug.value(U), [obs], T, save_fig_file=os.path.join(save_file, 'iteration_' + str(i))))

    # set initial conditions
    # opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    print('Setting up Solver...')
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
    opti.solver('ipopt', opts)

    # solve problem
    print('Solving OCP...')
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    # save path and actions
    print('Saving Solution...')
    save_path = join(getcwd(), 'ocp_paths', save_file)
    np.savetxt(save_path + '_X.csv', x_opt)
    np.savetxt(save_path + '_U.csv', u_opt)
    np.savetxt(save_path + '_t.csv', np.insert(np.cumsum(dt),0,0))

    # if save: plot_solution3_convex_hull(x_opt, u_opt, meshfile, T, save_fig_file='gemini_convex_below_above')
    meshfiles = []
    for i in range(15):
        meshfiles.append(join(meshdir, str(i) + '.stl'))

    if show: 
        # meshfile = join(filename, 'mercury_convex.stl')
        ## knot cost function
        knot_cost = 0
        for i, k in enumerate(knot_idx):
            knot_cost += sumsqr(x_opt[k,:3].T - knots[i,:3])
        print('Knot Cost = ', knot_cost)
        print('Plotting Solution')
        plot_solution3_convex_hull(x_opt, u_opt, meshfiles, dt, station=False)


def ocp_station_knot(meshdir=join(getcwd(), 'model', 'convex_detailed_station'),
                     knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                     save_dir='thrust_test_k_1_p_1_f_1',
                     save_file='1.5m',
                     show=False,
                     thrust_limit=0.2,
                     k_weight=1,
                     p_weight=1,
                     f_weight=1
                     ):
    """
    ocp_station with knot points
    """
    print('Save Directory: ', save_dir)
    if not exists(join('ocp_paths', save_dir)): mkdir(join('ocp_paths', save_dir))
    print('Importing Initial Conditions...', flush=True)
    path = np.loadtxt(knotfile, delimiter=',') # (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans

    velocity = 0.2
    n_timesteps = 400
    dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    print('Path Duration: ', np.sum(dt), flush=True)
    n_timesteps = len(dt)
    min_station_distance = 0.2
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

    # for k in range(n_timesteps):
        # # Runge-Kutta 4 integration
        # k1 = f(X[k,:],              U[k,:])
        # k2 = f(X[k,:]+dt[k]/2*k1.T, U[k,:])
        # k3 = f(X[k,:]+dt[k]/2*k2.T, U[k,:])
        # k4 = f(X[k,:]+dt[k]*k3.T,   U[k,:])
        # x_next = X[k,:] + dt[k]/6*(k1.T+2*k2.T+2*k3.T+k4.T)
        # opti.subject_to(X[k+1,:]==x_next); # close the gaps

    ## constrain thrust limits
    print('Imposing Thrust Limits...', flush=True)
    opti.subject_to(sum1(U**2) <= thrust_limit**2)

    ## cost function
    print('Initializaing Cost Function...', flush=True)
    fuel_cost = compute_fuel_cost(U, dt)
    # total_impulse = 0
    # for k in range(n_timesteps):
        # total_impulse += sumsqr(U[k,:]) * dt[k]**2
    # fuel_cost = total_impulse/g0**2/Isp**2 # squared fuel cost

    ## knot cost function
    knot_cost = compute_knot_cost(X, knots, knot_idx)
    # knot_cost = 0
    # for i, k in enumerate(knot_idx):
        # knot_cost += sumsqr(X[k,:3].T - knots[i,:3])

    ## Path length cost
    path_cost = compute_path_cost(X)
    # path_cost = sumsqr(X[1:,:] - X[:-1,:]) # squared path length

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

    # look at solution at each iteration
    # if visualize:
    #     save_file = os.path.join(os.getcwd(), 'optimization_steps', 'one_obstacle', 'offsetx_' + str(offsetx) + '_offsety_' + str(offsety))
    #     if not os.path.exists(save_file):
    #         os.mkdir(save_file)
    #     opti.callback(lambda i: plot_solution3(opti.debug.value(X), opti.debug.value(U), [obs], T, save_fig_file=os.path.join(save_file, 'iteration_' + str(i))))

    # set initial conditions
    # opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    print('Setting up Solver...', flush=True)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-7}
    opti.solver('ipopt', opts)

    # solve problem
    print('Solving OCP...', flush=True)
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    # save path and actions
    print('Saving Solution...', flush=True)
    save_path = join(getcwd(), 'ocp_paths', save_dir, save_file)
    thrust_str = str(thrust_limit//1)[0] + '_' + str(((thrust_limit*10)%10)//1)[0] + str(((thrust_limit*100)%10)//1)[0]
    np.savetxt(save_path + '_X_' + thrust_str + '.csv', x_opt)
    np.savetxt(save_path + '_U_' + thrust_str + '.csv', u_opt)
    np.savetxt(save_path + '_t_' + thrust_str + '.csv', np.insert(np.cumsum(dt),0,0))

    # if save: plot_solution3_convex_hull(x_opt, u_opt, meshfile, T, save_fig_file='gemini_convex_below_above')

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


def ocp_station(filename=join(getcwd(), 'model', 'mockup'), visualize=False, show=False):
    """
    detailed convex station obstacle (multiple convex parts) for planning point to point paths around
    """
    ## problem size
    n_timesteps = 100
    T = 10.0
    dt = T/n_timesteps
    goal_config_weight = 1

    # x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = convex_hull_station()
    x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = convex_hull_mercury()

    ## define ode
    f = ode_funCW(n_states, n_inputs)

    ## instantiate opti stack
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    # use constraint to ensure initial and final conditions
    opti.subject_to(X[0,:].T == x0)
    # opti.subject_to(X[-1,:].T == xf)

    ## constrain dynamics
    for k in range(n_timesteps):
        # Runge-Kutta 4 integration
        k1 = f(X[k,:],           U[k,:])
        k2 = f(X[k,:]+dt/2*k1.T, U[k,:])
        k3 = f(X[k,:]+dt/2*k2.T, U[k,:])
        k4 = f(X[k,:]+dt*k3.T,   U[k,:])
        x_next = X[k,:] + dt/6*(k1.T+2*k2.T+2*k3.T+k4.T)
        opti.subject_to(X[k+1,:]==x_next); # close the gaps

        # one step integration
        # opti.subject_to(X[k+1,:].T == X[k,:].T + dt * f(X[k,:], U[k,:]))

    ## constrain thrust limits
    opti.subject_to(sum1(U**2) <= thrust_limit)

    ## cost function
    fuel_cost = sumsqr(U)/g0/Isp
    goal_cost = sumsqr(X[-1,:].T - xf)
    cost = fuel_cost_weight * fuel_cost + goal_config_weight * goal_cost

    # convex hull obstacle
    for o in obs:
        normals, points = o
        enforce_convex_hull(normals, points, opti, X)

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    # look at solution at each iteration
    # if visualize:
    #     save_file = os.path.join(os.getcwd(), 'optimization_steps', 'one_obstacle', 'offsetx_' + str(offsetx) + '_offsety_' + str(offsety))
    #     if not os.path.exists(save_file):
    #         os.mkdir(save_file)
    #     opti.callback(lambda i: plot_solution3(opti.debug.value(X), opti.debug.value(U), [obs], T, save_fig_file=os.path.join(save_file, 'iteration_' + str(i))))

    # set initial conditions
    x_initial = DM.zeros(n_timesteps+1, n_states)
    for i in range(6):
        x_initial[:,i] = c.linspace(x0[i], xf[i], n_timesteps+1)
    # opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(X, x_initial)
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-7}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    # if save: plot_solution3_convex_hull(x_opt, u_opt, meshfile, T, save_fig_file='gemini_convex_below_above')
    meshfiles = []
    for i in range(15):
        meshfiles.append(join(filename, str(i) + '.stl'))

    if show: 
        meshfile = join(filename, 'mercury_convex.stl')
        t = linspace(0,T,n_timesteps)
        plot_solution3_convex_hull(x_opt, u_opt, [meshfile], t)


def many_obstacles():
    """
    two spheres
    """
    n_timesteps = 50
    T = 20.0
    dt = T/n_timesteps
    goal_config_weight = 1

    # x0, xf, obstacles, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = two_spheres()
    x0, xf, obstacles, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = concatenated_spheres4()
    f = ode_funCW(n_states, n_inputs)

    # setup optimizer
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    # use constraint to ensure initial and final conditions
    opti.subject_to(X[0,:].T == x0)
    opti.subject_to(X[-1,:].T == xf)

    ## constrain dynamics
    for k in range(n_timesteps):
        opti.subject_to(X[k+1,:].T == X[k,:].T + dt * f(X[k,:], U[k,:]))

    ## constrain thrust limits
    opti.subject_to(sum1(U**2) <= thrust_limit)

    ## constrain collisions (distance away from centerpoint of circle)
    for k in range(n_timesteps):
        for obs in obstacles:
            opti.subject_to(((X[k,0] - obs[0])**2 + (X[k,1] - obs[1])**2 + (X[k,2] - obs[2])**2) >= obs[3]**2)

    ## cost function
    fuel_cost = sumsqr(U)/g0/Isp
    goal_cost = sumsqr(X[-1,:].T - xf)
    cost = fuel_cost_weight * fuel_cost + goal_config_weight * goal_cost

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    # set initial conditions
    opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-5}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    plot_solution3(x_opt, u_opt, obstacles, T)

# def one_obstacle(offsetx, offsety):
def one_obstacle(offsetx=0.6, offsety=0.5, visualize=False):
    ## problem size
    threeD = True
    n_timesteps = 100
    T = 10.0
    dt = T/n_timesteps
    goal_config_weight = 1

    x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = one_obs(threespace=threeD, ox=offsetx, oy=offsety)

    ## define ode
    # f = ode_fun3(n_states, n_inputs)
    f = ode_funCW(n_states, n_inputs)

    ## instantiate opti stack
    opti = Opti()

    ## optimization variables
    X = opti.variable(n_timesteps+1, n_states)
    U = opti.variable(n_timesteps, n_inputs)

    ### CONSTRAINTS ###

    # use constraint to ensure initial and final conditions
    opti.subject_to(X[0,:].T == x0)
    opti.subject_to(X[-1,:].T == xf)

    ## constrain dynamics
    for k in range(n_timesteps):
        opti.subject_to(X[k+1,:].T == X[k,:].T + dt * f(X[k,:], U[k,:]))

    ## constrain thrust limits
    opti.subject_to(sum1(U**2) <= thrust_limit)

    ## constrain collisions (distance away from centerpoint of circle)
    for k in range(n_timesteps):
        if threeD:
            opti.subject_to(((X[k,0] - obs[0])**2 + (X[k,1] - obs[1])**2 + (X[k,2] - obs[2])**2) >= obs[3]**2)
        else:
            opti.subject_to(((X[k,0] - obs[0])**2 + (X[k,1] - obs[1])**2) >= obs[2]**2)

    ## cost function
    fuel_cost = sumsqr(U)/g0/Isp
    goal_cost = sumsqr(X[-1,:].T - xf)
    cost = fuel_cost_weight * fuel_cost + goal_config_weight * goal_cost

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    # look at solution at each iteration
    if visualize:
        save_file = os.path.join(os.getcwd(), 'optimization_steps', 'one_obstacle', 'offsetx_' + str(offsetx) + '_offsety_' + str(offsety))
        if not os.path.exists(save_file):
            os.mkdir(save_file)
        opti.callback(lambda i: plot_solution3(opti.debug.value(X), opti.debug.value(U), [obs], T, save_fig_file=os.path.join(save_file, 'iteration_' + str(i))))

    # set initial conditions
    opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-5}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)


    # if threeD:
    #     plot_solution3(x_opt, u_opt, [obs], T)
    # else:
    #     plot_solution2(x_opt, u_opt, obs, T)


def grid_test():
    # many_obstacles()
    numx = 21
    numy = 21
    xoffsets = np.linspace(0, 1, numx)
    yoffsets = np.linspace(0, 1, numy)
    solution = []
    nosolution = []
    for ox in tqdm(xoffsets):
        for oy in yoffsets:
            try: 
                one_obstacle(offsetx=ox, offsety=oy, visualize=True)
                solution.append([ox, oy])
            except:
                nosolution.append([ox, oy])

    fig, ax = plt.subplots()

    solution = np.array(solution).T
    nosolution = np.array(nosolution).T
    print(solution.shape)
    print(nosolution.shape)
    print(solution)
    print(nosolution)
    ax.plot(solution[0,:], solution[1,:], 'gx', lw=2, label='Converged')
    ax.plot(nosolution[0,:], nosolution[1,:], 'rx', lw=2, label='Unable to Converge')
    ax.set_title("Trajectory Optimization Success for Varying Planar Sphere Offsets")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.legend()
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]*1.2])
    fig.savefig('grid_test_convergence_sphere.png', dpi=300)
    plt.show()

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

    # read in save directory
    # if len(argv) > 2: save_dir_input = argv[2]
    # else: save_dir_input='thrust_test_k_1_p_1_f_1',
    save_dir_input = 'thrust_test_k_' + k_str + '_p_' + p_str + '_f_' + f_str

    ocp_station_knot(thrust_limit=thrust_limit_input, save_dir=save_dir_input, k_weight=knot_cost_weight, p_weight=path_cost_weight, f_weight=fuel_cost_weight)
