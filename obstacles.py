from casadi import vertcat

def ic_sphere(threespace=True):
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
        s_x0 = x0[0].__float__() + 0.6
        s_x1 = x0[1].__float__() + 0.5
        s_x2 = (xf[2].__float__() - x0[2].__float__())/2
        s_r = 1.0
        obs = [s_x0, s_x1, s_x2, s_r]
    else: # sphere
        s_x0 = x0[0].__float__()
        s_x1 = (xf[1].__float__() - x0[1].__float__())/2
        s_r = 1.0
        obs = [s_x0, s_x1, s_r]
    return x0, xf, obs, n_states, n_inputs, thrust_limit, fuel_cost_weight, g0, Isp
