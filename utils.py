from numpy import linspace, mgrid, pi, sin, cos, mean, isnan, diff, sum, floor, cumsum, array, insert, append, loadtxt
from numpy.linalg import norm
from numpy.matlib import repmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
from casadi import dot, fmax

def linear_initial_path(knots, knot_idx, dt):
    """
    linearly interpolate between knot points 
    """
    lin_path = knots[[0],:]
    for i in range(knots.shape[0]-1):
        # start and end of linear interpolation
        prev_pose = knots[i,:]
        cur_pose = knots[i+1,:]

        # indices of time vector for linear interpolation
        prev_idx = knot_idx[i]
        cur_idx = knot_idx[i+1]

        # cumulative sum of the time steps to compute total time elapsed until each interpolated pose from 'prev_pose'
        dt_cumsum = cumsum(dt[prev_idx:cur_idx])

        # linear interpolation period
        T = dt_cumsum[-1]

        # constant velocity for interpolation period
        v = (cur_pose - prev_pose)/T
        v = repmat(v.reshape((1,-1)), dt_cumsum.shape[0], 1)
        dt_cumsum = repmat(dt_cumsum.reshape((-1,1)), 1, 6)

        # compute poses from cumsum vector
        poses = cur_pose + v*dt_cumsum

        # append linearly interpolated poses to path
        lin_path = append(lin_path, poses, axis=0)

    return lin_path

def filter_path_na(path):
    """
    remove waypoints with nan from path
    """
    knot_bool = ~isnan(path)[:,-1]
    return path[knot_bool,:]

def compute_time_intervals(knots, velocity, num_timesteps):
    """
    compute time intervals for optimal control path given a velocity and path knot points
    knot_idx - index of state variable for knot point enforcement
    """
    # get distance between each knotpoint
    dknots = diff(knots[:,:3], axis=0)
    ds = norm(dknots, axis=1)

    total_time = sum(ds)/velocity # total path time

    dt = total_time/num_timesteps # regular timestep length
    dt_knot = ds/velocity # time between each knotpoint
    num_dt_per_interval = (dt_knot//dt).astype(int) # number of regular timesteps between each knotpoint
    extra_dt = dt_knot%dt # timestep remainder from regular timesteps (irregular)

    # construct dt's
    dts = []
    knot_idx = [0]
    for i, num_dt in enumerate(num_dt_per_interval):
        dts.extend([dt]*num_dt)
        dts.append(extra_dt[i])
        knot_idx.append(knot_idx[-1] + num_dt + 1)

    return dts, knot_idx


def enforce_convex_hull(normals, points, opti, X, min_station_distance):
    """
    create constraint formulation for opti stack for a convex hull given face normals and centroids
    normals - list of 3d vectors (dir) np.ndarray(num_normals, 3)
    centroids - list of 3d vectors (position) np.ndarray(num_centroids, 3)
    opti - opti stack variable
    X - state variable MX.shape(num_timesteps, 6)
    """
    num_normals = normals.shape[0]
    num_timesteps = X.shape[0]

    # for each state timestep we apply the convex hull keepout constraint
    for j in range(num_timesteps):

        # create a convex hull keepout constraint for each time step:
        dot_max = -1 # we can instantiate the max dot product as -1 because dot products less than zero do not satisfy the constraint (we take maximum)
        for i in range(num_normals):

            # first retrieve parameters for each face instance
            n = normals[[i],:] # face normal
            p = points[[i],:] # centroid corresponding to face normal
            x = X[j,:3] # state at timestep j (just position)
            r = x-p # vector from face centroid to state position

            # only one dot product must be greater than zero so we take the maximum value
            # of all of them to use as the constraint (for each timestep)
            dot_max = fmax(dot_max, dot(n,r)) 
        
        # if max dot product value is above zero, then constraint is met (only one needs to be greater)
        opti.subject_to(dot_max > min_station_distance)


def plot_solution3_convex_hull(x, u, meshfiles, t, plot_state=False, plot_actions=True, save_fig_file=None):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    # t = linspace(0,T,N)
    qs = 5 # quiver spacing: spacing of visual control actions along path


    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # get station offset
    translation = loadtxt('translate_station.txt', delimiter=',').reshape(1,1,3)

    # Load the STL files and add the vectors to the plot
    for meshfile in meshfiles:
        your_mesh = mesh.Mesh.from_file(meshfile)
        vectors = your_mesh.vectors + translation
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors))

    # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    # axes.auto_scale_xyz(scale, scale, scale)

    axes.plot(x[:,0],x[:,1],x[:,2],
             color='k',
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
