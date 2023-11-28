from numpy import sqrt
from casadi import MX, vertcat, Function

def ode_funCW(n_states, n_inputs):
    """
    ode function for free floating inspector in space using CW equations
    """
    m_I = 1 # inspector mass
    mu = 3.986e14 # standard gravitational parameter
    a = 6.6e6 # international space station orbit radius
    n = sqrt(mu/a**3) # orbital rate of target craft

    x = MX.sym('x', n_states)
    u = MX.sym('u', n_inputs)

    xdot = vertcat(x[3],
                   x[4],
                   x[5],
                   3*n**2* x[0] + 2*n*x[4] + u[0]/m_I,
                   -2*n*x[3]    + u[1]/m_I,
                   -n**2*x[2]   + u[2]/m_I)

    return Function('ode_fun', [x, u], [xdot])

def ode_fun3(n_states, n_inputs):
    """
    ode function for free floating inspector in 3D
    """
    m_I = 1

    x = MX.sym('x', n_states)
    u = MX.sym('u', n_inputs)

    xdot = vertcat(x[3],
                   x[4],
                   x[5],
                   u[0]/m_I,
                   u[1]/m_I,
                   u[2]/m_I)

    return Function('ode_fun', [x, u], [xdot])


def ode_fun2(n_states, n_inputs):
    """
    ode function for free floating inspector (2D problem)
    """
    m_I = 1

    x = MX.sym('x', n_states)
    u = MX.sym('u', n_inputs)

    xdot = vertcat(x[2],
                   x[3],
                   u[0]/m_I,
                   u[1]/m_I)

    return Function('ode_fun', [x, u], [xdot])
