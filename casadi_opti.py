from casadi import *
from ode import ode_fun2, ode_fun3, ode_funCW
from matplotlib import pyplot as plt
from utils import plot_solution2, plot_solution3
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from initial_conditions import one_obs, two_spheres, concatenated_spheres2, concatenated_spheres3

def two_obstacles():
    """
    two spheres
    """
    n_timesteps = 100
    T = 20.0
    dt = T/n_timesteps

    # x0, xf, obstacles, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = two_spheres()
    x0, xf, obstacles, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = concatenated_spheres3(obstacle_offset=0.7)
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
    cost = fuel_cost_weight * sumsqr(U)/g0/Isp

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    # set initial conditions
    opti.set_initial(X, DM.zeros(n_timesteps+1, n_states))
    opti.set_initial(U, DM.zeros(n_timesteps, n_inputs))

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-9}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    plot_solution3(x_opt, u_opt, obstacles, T)

def one_obstacle():
    ## problem size
    threeD = True
    n_timesteps = 400
    T = 10.0
    dt = T/n_timesteps

    x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp = one_obs(threespace=threeD)

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
    cost = fuel_cost_weight * sumsqr(U)/g0/Isp

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-5}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    if threeD:
        plot_solution3(x_opt, u_opt, [obs], T)
    else:
        plot_solution2(x_opt, u_opt, obs, T)


if __name__ == "__main__":
    two_obstacles()