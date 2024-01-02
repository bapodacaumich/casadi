from numpy import linspace, mgrid, pi, sin, cos, mean, zeros
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
from casadi import dot, fmax

def enforce_convex_hull(normals, points, opti, X):
    """
    create constraint formulation for opti stack for a convex hull given face normals and centroids
    normals - list of 3d vectors (dir) np.ndarray(num_normals, 3)
    centroids - list of 3d vectors (position) np.ndarray(num_centroids, 3)
    opti - opti stack variable
    X - state variable MX.shape(num_timesteps, 6)
    """
    num_normals = normals.shape[0]
    num_timesteps = X.shape[0]

    constraint = []
    for i in range(num_normals):
        for j in range(num_timesteps):
            n = normals[[i],:] # face normal
            p = points[[i],:] # centroid corresponding to face normal
            x = X[j,:3] # state at timestep j (just position)
            r = x-p # vector from face centroid to state position
            # constraint += fmax(dot(n,r), 0) # slack variable that is positive for points in front of face
            constraint.append(dot(n,r) > 0)
            # constraint = constraint or (dot(n,r) > 0)

    print(constraint)
    opti.subject_to(any(constraint)) # all constraints must be true

def plot_solution3_convex_hull(x, u, meshfile, T, plot_state=False, plot_actions=True, save_fig_file=None):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    t = linspace(0,T,N)
    qs = 5 # quiver spacing: spacing of visual control actions along path


    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(meshfile)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    # axes.auto_scale_xyz(scale, scale, scale)

    axes.plot(x[:,0],x[:,1],x[:,2],
             color='tab:blue',
             label='Inspector')
    if plot_actions:
        axes.quiver(x[:-1:qs,0],x[:-1:qs,1],x[:-1:qs,2],
                u[::qs,0],  u[::qs,1],  u[::qs,2],
                color='tab:red',
                label='Thrust',
                length=0.4,
                normalize=True)
    axes.legend()
    ulim = max(axes.get_xlim()[1], axes.get_ylim()[1], axes.get_zlim()[1])
    llim = min(axes.get_xlim()[0], axes.get_ylim()[0], axes.get_zlim()[0])
    z_scale = 1.3
    lim_range = ulim - llim
    avg_x2 = mean(x[:,2])
    llim = avg_x2 - lim_range/2
    ulim = avg_x2 + lim_range/2
    axes.set_zlim([llim, ulim])
    lim_range = (ulim - llim)*z_scale
    llim = -lim_range/2
    ulim = lim_range/2
    axes.set_xlim([llim, ulim])
    axes.set_ylim([llim, ulim])

    if plot_state:
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
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close("all")
    if save_fig_file is not None:
        figure.savefig(save_fig_file + '_path.png', dpi=300)
        plt.close(figure)
        if plot_state:
            fig.savefig(save_fig_file + '_states.png', dpi=300)
            plt.close(fig)
    else: plt.show()


def plot_solution3(x, u, obs, T, ax0=None, plot_state=False, plot_actions=True, save_fig_file=None):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    t = linspace(0,T,N)
    qs = 5 # quiver spacing: spacing of visual control actions along path

    if ax0 is None:
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
    if plot_actions:
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

    if plot_state:
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
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close("all")
    if save_fig_file is not None:
        fig.savefig(save_fig_file, dpi=300)
        plt.close(fig)
    else: plt.show()


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
