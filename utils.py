from numpy import linspace, mgrid, pi, sin, cos, mean
from matplotlib import pyplot as plt

def plot_solution3(x, u, obs, T):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    t = linspace(0,T,N)
    qs = 5 # quiver spacing: spacing of visual control actions along path

    fig = plt.figure()
    ax0 = fig.add_subplot(projection='3d')

    for s in obs:
        # draw sphere
        u_sphere, v_sphere = mgrid[0:2*pi:20j, 0:pi:10j]
        x_sphere = s[0] + s[3]*cos(u_sphere)*sin(v_sphere)
        y_sphere = s[1] + s[3]*sin(u_sphere)*sin(v_sphere)
        z_sphere = s[2] + s[3]*cos(v_sphere)

        ax0.plot_surface(x_sphere, y_sphere, z_sphere,
                        color='k',
                        alpha=0.2)
        ax0.plot_wireframe(x_sphere, y_sphere, z_sphere,
                        color='k',
                        linewidth=0.2,
                        label='')
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
    lim_range = ulim - llim
    avg_x2 = mean(x[:,2])
    llim = avg_x2 - lim_range/2
    ulim = avg_x2 + lim_range/2
    ax0.set_zlim([llim, ulim])
    lim_range = (ulim - llim)*z_scale
    llim = -lim_range/2
    ulim = lim_range/2
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
    t = linspace(0,T,N)
    qs = 20 # quiver spacing: spacing of visual control actions along path

    # obstacle parameterization
    theta_circle = linspace(0,2*pi,100)
    x_circle = c[2]*cos(theta_circle) + c[0]
    y_circle = c[2]*sin(theta_circle) + c[1]

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
