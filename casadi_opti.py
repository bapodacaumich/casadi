from casadi import *
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def ode_funCW(n_states, n_inputs):
    """
    ode function for free floating inspector in space using CW equations
    """
    m_I = 1 # inspector mass
    mu = 3.986e14 # standard gravitational parameter
    a = 6.6e6 # international space station orbit radius
    n = np.sqrt(mu/a**3) # orbital rate of target craft

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

def plot_solution3(x, u, s, T):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    t = np.linspace(0,T,N)
    qs = 20 # quiver spacing: spacing of visual control actions along path

    # draw sphere
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = s[0] + s[3]*np.cos(u_sphere)*np.sin(v_sphere)
    y_sphere = s[1] + s[3]*np.sin(u_sphere)*np.sin(v_sphere)
    z_sphere = s[2] + s[3]*np.cos(v_sphere)

    fig = plt.figure()
    ax0 = fig.add_subplot(projection='3d')

    ax0.plot_surface(x_sphere, y_sphere, z_sphere,
                     color='k',
                     alpha=0.2)
    ax0.plot_wireframe(x_sphere, y_sphere, z_sphere,
                       color='k',
                       linewidth=0.2,
                       label='Obstacle')
    ax0.plot(x[:,0],x[:,1],x[:,2],
             color='tab:blue',
             label='Inspector')
    ax0.quiver(x[:-1:qs,0],x[:-1:qs,1],x[:-1:qs,2],
               u[::qs,0],  u[::qs,1],  u[::qs,2],
               color='tab:red',
               label='Thrust',
               length=0.4,
               normalize=True)
    ax0.legend()
    ulim = max(ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1])
    llim = min(ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0])
    z_scale = 1.3
    ax0.set_zlim([llim, ulim])
    ulim = (ulim-llim)*z_scale + llim
    ax0.set_xlim([llim, ulim])
    ax0.set_ylim([llim, ulim])

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(t, x[:,0], label='x0')
    ax1.plot(t, x[:,1], label='x1')
    ax1.plot(t, x[:,2], label='x2')
    ax1.plot(t, x[:,3], label='x0_dot')
    ax1.plot(t, x[:,4], label='x1_dot')
    ax1.plot(t, x[:,5], label='x2_dot')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('State')
    ax1.set_title('Optimal States')
    ax1.legend()

    ax2.plot(t[1:], u[:,0], label='u0')
    ax2.plot(t[1:], u[:,1], label='u1')
    ax2.plot(t[1:], u[:,2], label='u2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Thrust')
    ax2.set_title('Optimal Control Inputs')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_solution2(x, u, c, T):
    """
    2D solution space
    x - states shape(N,4)
    u - actions shape(N,2)
    s - circle obstacle [x1, x2, radius]
    """
    N = x.shape[0]
    t = np.linspace(0,T,N)
    qs = 20 # quiver spacing: spacing of visual control actions along path

    # obstacle parameterization
    theta_circle = np.linspace(0,2*np.pi,100)
    x_circle = c[2]*np.cos(theta_circle) + c[0]
    y_circle = c[2]*np.sin(theta_circle) + c[1]

    _, ax0 = plt.subplots()
    _, (ax1, ax2) = plt.subplots(2,1)

    ax0.plot(x[:,0], x[:,1],
             color='tab:blue',
             label='Inspector')
    ax0.plot(x_circle, y_circle, 
             color='k',
             label='Obstacle')
    ax0.quiver(x[1::qs,0], x[1::qs,1], u[::qs,0], u[::qs,1],
               color='tab:red',
               scale=1, 
               width=0.005,
               headwidth=2,
               label='Thrust')
    ax0.set_xlabel('x_0')
    ax0.set_ylabel('y_0')
    ax0.set_title('Inspector Path')
    ax0.legend()
    ax0.set_aspect('equal','box')

    ax1.plot(t,x[:,0], label='x0')
    ax1.plot(t,x[:,1], label='x1')
    ax1.plot(t,x[:,2], label='x0_dot')
    ax1.plot(t,x[:,3], label='x1_dot')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('State')
    ax1.set_title('State Vector')
    ax1.legend()

    ax2.plot(t[:-1],u[:,0], label='u_0')
    ax2.plot(t[:-1],u[:,1], label='u_1')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Action')
    ax2.set_title('Control Inputs')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    ## problem size
    threespace = True
    n_timesteps = 400
    T = 10.0
    dt = T/n_timesteps

    ## Constraints and Parameters
    thrust_limit = 10.0
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
        x0 = vertcat(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        xf = vertcat(1.0, 1.0, 4.0, 0.0, 0.0, 0.0)
    else:
        x0 = vertcat(0.0, 0.0, 0.0, 0.0)
        xf = vertcat(0.0, 4.0, 0.0, 0.0)


    ## obstacle specifications
    if threespace: # three dimensions
        s_x0 = x0[0].__float__()
        s_x1 = x0[1].__float__()
        s_x2 = (xf[2].__float__() - x0[2].__float__())/2
        s_r = 1.0
        sphere = [s_x0, s_x1, s_x2, s_r]
    else: # sphere
        s_x0 = x0[0].__float__()
        s_x1 = (xf[1].__float__() - x0[1].__float__())/2
        s_r = 1.0
        circle = [s_x0, s_x1, s_r]

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
        if threespace:
            opti.subject_to(((X[k,0] - sphere[0])**2 + (X[k,1] - sphere[1])**2 + (X[k,2] - sphere[2])**2) >= s_r**2)
        else:
            opti.subject_to(((X[k,0] - circle[0])**2 + (X[k,1] - circle[1])**2) >= s_r**2)

    ## cost function
    cost = fuel_cost_weight * sumsqr(U)/g0/Isp

    # add obstacle to cost fn -- Dont need this
    # cost += obstacle_cost_weight * sumsqr(exp(-1*((X[:,:2].T - vertcat(s_x1, s_x2))**2)))

    # add cost to optimization problem
    opti.minimize(cost)

    ## solver
    # create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1}
    opti.solver('ipopt', opts)

    # solve problem
    sol = opti.solve()

    # optimal states
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    if threespace:
        plot_solution3(x_opt, u_opt, sphere, T)
    else:
        plot_solution2(x_opt, u_opt, circle, T)


if __name__ == "__main__":
    main()