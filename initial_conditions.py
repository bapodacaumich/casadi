from casadi import vertcat
from numpy import linspace

def concatenated_spheres4(obstacle_coords=[(0.8,1.0),(-0.8,3.0),(0.8,5.0),(0.0,8.0)], n_obs=17, goal_separation=9.1):
    """
    overlapping spheres
    fourth 'obstacle' by Ella's suggestion
    """
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    ## state definition and obstacle specification
    n_states = 6
    n_inputs = 3

    ## initial and final conditions
    x0 = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xf = vertcat(0.0, 0.0, goal_separation, 0.0, 0.0, 0.0)

    ## obstacle specifications
    obs = []
    for offset in linspace(-2,2,n_obs):
        for u, v in obstacle_coords:
            obs.append(ic_sphere(x0, xf, x0_offset=u, x1_offset=offset, x2_offset=v))
        # obs.append(ic_sphere(x0, xf, x0_offset=-obstacle_offset, x1_offset=offset, x2_offset=obs_x2[1]))
        # obs.append(ic_sphere(x0, xf, x0_offset=obstacle_offset, x1_offset=offset, x2_offset=obs_x2[2]))
        # obs.append(ic_sphere(x0, xf, x0_offset=final_obstacle_offset, x1_offset=offset, x2_offset=obs_x2[3]))
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp

def concatenated_spheres3(obstacle_offset=0.8, obs_x2=[1.0,3.0,5.0], n_obs=17, goal_separation=6.0):
    """
    overlapping spheres
    """
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    ## state definition and obstacle specification
    n_states = 6
    n_inputs = 3

    ## initial and final conditions
    x0 = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xf = vertcat(0.0, 0.0, goal_separation, 0.0, 0.0, 0.0)

    ## obstacle specifications
    obs = []
    for offset in linspace(-2,2,n_obs):
        obs.append(ic_sphere(x0, xf, x0_offset=obstacle_offset, x1_offset=offset, x2_offset=obs_x2[0]))
        obs.append(ic_sphere(x0, xf, x0_offset=-obstacle_offset, x1_offset=offset, x2_offset=obs_x2[1]))
        obs.append(ic_sphere(x0, xf, x0_offset=obstacle_offset, x1_offset=offset, x2_offset=obs_x2[2]))
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp

def concatenated_spheres2(obstacle_offset=0.8, sep_factor=0.25, n_obs=17):
    """
    overlapping spheres
    """
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    ## state definition and obstacle specification
    n_states = 6
    n_inputs = 3

    ## initial and final conditions
    x0 = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xf = vertcat(0.0, 0.0, 4.0, 0.0, 0.0, 0.0)

    ## obstacle specifications
    obs_x2 = []
    obs_x2.append((xf[2].__float__() - x0[2].__float__())*sep_factor)
    obs_x2.append((xf[2].__float__() - x0[2].__float__())*(1-sep_factor))

    obs = []
    for offset in linspace(-2,2,n_obs):
        obs.append(ic_sphere(x0, xf, x0_offset=obstacle_offset, x1_offset=offset, x2_offset=obs_x2[0]))
        obs.append(ic_sphere(x0, xf, x0_offset=-obstacle_offset, x1_offset=offset, x2_offset=obs_x2[1]))
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp

def two_spheres(obstacle_offset=0.7, sep_factor=0.25):
    """
    two spheres offset from 
    """
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    ## state definition and obstacle specification
    n_states = 6
    n_inputs = 3

    ## initial and final conditions
    x0 = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xf = vertcat(0.0, 0.0, 4.0, 0.0, 0.0, 0.0)

    ## obstacle specifications
    obs_x2 = []
    obs_x2.append((xf[2].__float__()- x0[2].__float__())*sep_factor)
    obs_x2.append((xf[2].__float__()- x0[2].__float__())*(1-sep_factor))

    obs = [ic_sphere(x0, xf, x0_offset=obstacle_offset, x1_offset=0.0, x2_offset=obs_x2[0]),
           ic_sphere(x0, xf, x0_offset=-obstacle_offset, x1_offset=0.0, x2_offset=obs_x2[1])]
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp

def one_obs(threespace=True, ox=0.6, oy=0.5):
    """
    initial conditions and obstacle for zero gravity environment
    threespace - bool indicating 3D (false for 2D)
    """
    ## Constraints and Parameters
    thrust_limit = 100.0
    fuel_cost_weight = 1.0
    g0 = 9.81
    Isp = 80

    ## state definition and obstacle specification
    if threespace:
        n_states = 6
        n_inputs = 3
    else:
        n_states = 4
        n_inputs = 2

    ## initial and final conditions
    if threespace:
        x0 = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        xf = vertcat(0.0, 0.0, 4.0, 0.0, 0.0, 0.0)
    else:
        x0 = vertcat(0.0, 0.0, 0.0, 0.0)
        xf = vertcat(0.0, 4.0, 0.0, 0.0)

    ## obstacle specifications
    if threespace: # three dimensions
        obs = ic_sphere(x0, xf, x0_offset=ox, x1_offset=oy)
    else: # sphere
        obs = ic_circle(x0, xf)
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp

def ic_sphere(x0, xf, s_r = 1.0, x0_offset=0.6, x1_offset=0.5, x2_factor=0.5):
    """
    generate sphere between x0 and xf with small amount of offset
    """
    s_x0 = x0[0].__float__() + x0_offset
    s_x1 = x0[1].__float__() + x1_offset
    s_x2 = (xf[2].__float__() - x0[2].__float__())*x2_factor
    # s_x2 = x0[2].__float__() + x2_offset
    return [s_x0, s_x1, s_x2, s_r]

def ic_circle(x0, xf, s_r=1.0, x1_factor=0.5):
    """
    generate circle obstacle btween x0 and xf
    """
    s_x0 = x0[0].__float__()
    s_x1 = (xf[1].__float__() - x0[1].__float__())*x1_factor
    return [s_x0, s_x1, s_r]