from numpy import linspace, mgrid, pi, sin, cos, mean, isnan, diff, sum, floor, cumsum, array, insert, append, loadtxt
import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
from casadi import dot, fmax
from camera import Camera
from os import getcwd, listdir
from os.path import join
from tqdm import tqdm

def set_aspect_equal_3d(axes):
    # get aspect ratios
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    zlim = axes.get_zlim()
    axes.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])

    return axes

def load_path_data(sol_dir=join(getcwd(), 'ocp_paths', 'thrust_test_k_100_p_0_1_f_1'),
                   knot_file=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')):
    """
    load file states and time vector along with knotpoints
    """
    path = np.loadtxt(knot_file, delimiter=',') # (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans
    x = np.loadtxt(join(sol_dir, '1.5m_X_1_70.csv'))
    t = np.loadtxt(join(sol_dir, '1.5m_t_1_70.csv'))
    return knots, x, t

def get_knot_idx(knots, x):
    """get the knot idx that correspond to x states

    Args:
        knots (_type_): (num_knots, 6) [x y z xdir ydir zdir]
        x (_type_): (N, 6) [x y z xdot ydot zdot]
    """
    num_knots = knots.shape[0]
    num_states = x.shape[0]
    knot_idx = [0]
    for i in range(1, num_knots-1):
        knot_closeness = np.sum((x[:,:3] - knots[i, :3])**2, axis=1)
        closest_knot_idx = np.argmin(knot_closeness)
        knot_idx.append(closest_knot_idx)
    knot_idx.append(num_states-1)
    return knot_idx

def process_data(knots, x, t):
    """process data for packaging

    Args:
        knots (np.array): knot points
        x (np.array): ocp state vector
        t (np.array): time vector
    """
    # velocity = 0.2
    # n_timesteps = 400
    # dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    # print(knot_idx)

    # t_load = np.insert(np.cumsum(dt),0,0)
    # print(t_load.shape)
    # assert (t_load == t).all()
    t = t.reshape((-1,1))
    knot_idx = np.array(get_knot_idx(knots, x))

    norepeats = True
    while norepeats:
        knot_sort_idx = np.argsort(knot_idx)
        knots = knots[knot_sort_idx,:]
        knot_idx = knot_idx[knot_sort_idx]

        norepeats = False
        for i in range(len(knot_idx)-1):
            if knot_idx[i] == knot_idx[i+1]:
                norepeats = True
                knot_idx[i+1] += 1

    x[:,3:] = 0
    x[0, 3:] = knots[0, 3:]
    for i in range(len(knot_idx)-1):
        prev_knot = knots[i]
        cur_knot = knots[i+1]

        prev_idx = knot_idx[i]
        cur_idx = knot_idx[i+1]
        n_interval = knot_idx[i+1] - knot_idx[i] + 1 # inclusive

        orientations = np.linspace(prev_knot[3:], cur_knot[3:], n_interval) # linearly interpolate orientation (last three)
        orientations = orientations / np.linalg.norm(orientations, axis=1).reshape((-1,1))

        x[prev_idx+1:cur_idx+1, 3:] = orientations[1:,:]

    return np.concatenate((x,t), axis=1)


def compute_path_coverage(knots, x, t,
                          meshdir='remeshed'):
    """compute coverage of path X over meshfiles

    Args:
        X (np.ndarray): path vector [x, y, z, xl, yl, zl] - first three are position, second three are look direction (xl, yl, zl) = unit look direction vector
        meshfile (list): list of meshfile locations
    """
    # load path stuff
    # knots, x, t = load_path_data(soln_dir, knot_file)
    X = process_data(knots, x, t)

    # load meshes
    meshes = []
    face_counts = []
    meshdir = join(getcwd(), 'model', meshdir)
    for file in listdir(meshdir):
        cur_mesh = mesh.Mesh.from_file(join(meshdir, file))
        face_counts.append(len(cur_mesh.normals))
        meshes.append(cur_mesh)

    # setup camera
    cam = Camera(fov=(pi/4, pi/4), d=5)

    # iterate through poses and determine coverage for every mesh face
    coverage_count = 0
    # for face_count, cur_mesh in tqdm(zip(face_counts, meshes), total=len(face_counts), position=2, leave=False):
    for face_count, cur_mesh in zip(face_counts, meshes):
    # for face_count, cur_mesh in zip(face_counts, meshes):
        cur_coverage_map = np.zeros(face_count)

        for pose_idx in range(X.shape[0]):
            pose = np.zeros(5)
            pose[:3] = X[pose_idx,:3]
            pose[3] = np.arctan2(X[pose_idx,4], X[pose_idx,3])
            pose[4] = np.arctan2(np.sqrt(X[pose_idx,3]**2 + X[pose_idx,4]**2), -X[pose_idx,5])

            for i in range(face_count):
                normal = cur_mesh.normals[i]
                if (cam.coverage(pose, cur_mesh.v0[i], normal)):
                    cur_coverage_map[i] = 1
                # if (cam.coverage(pose, cur_mesh.v0[i], normal) and
                #     cam.coverage(pose, cur_mesh.v1[i], normal) and 
                #     cam.coverage(pose, cur_mesh.v2[i], normal)):
                #     cur_coverage_map[i] = 1

        coverage_count += np.sum(cur_coverage_map)

    coverage_ratio = coverage_count/np.sum(np.array(face_counts))

    return coverage_ratio

def num2str(num):
    """ parse num into hundredth palce string 123.45678900000 --> 123_45. works for numbers under 1000 and equal to or above 0.01

    Args:
        num (float): float to parse into string
    """
    string = ''
    if num >= 100:
        string += str(num/100)[0]
        num = num % 100
        if num >= 10: string += str(num/10)[0]
        else: string += '0'
        num = num % 10
    if num >= 10:
        string += str(num/10)[0]
        num = num % 10
    string += str(num)[0] + '_'
    num = np.round(num % 1, decimals=3)
    string += str(num)[2:5]
    return string

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
        dt_cumsum = repmat(dt_cumsum.reshape((-1,1)), 1, 3)

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

def compute_time_intervals(knots, velocity, num_timesteps, start_pose=None):
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


def plot_solution3_convex_hull(x, u, meshfiles, t, plot_state=False, plot_actions=True, save_fig_file=None, station=True):
    """
    2D solution space
    x - states shape(N,6)
    u - actions shape(N,3)
    s - sphere obstacle [x1, x2, x3, radius]
    """
    N = x.shape[0]
    # t = linspace(0,T,N)
    qs = 1 # quiver spacing: spacing of visual control actions along path


    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    if station:
        # get station offset
        translation = loadtxt('translate_station.txt', delimiter=',').reshape(1,1,3)

        # Load the STL files and add the vectors to the plot
        for meshfile in meshfiles:
            your_mesh = mesh.Mesh.from_file(meshfile)
            vectors = your_mesh.vectors + translation
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors))
    else:
        files = ['mercury_convex.stl','gemini_convex.stl']
        for f in files:
            meshfile = join(getcwd(), 'model', 'mockup', f)
            your_mesh = mesh.Mesh.from_file(meshfile)
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    # axes.auto_scale_xyz(scale, scale, scale)

    axes.plot(x[:,0],x[:,1],x[:,2],
             color='k',
             label='Inspector')
    if plot_actions:
        s = 10 # scale factor
        axes.quiver(x[:-1:qs,0],x[:-1:qs,1],x[:-1:qs,2],
                    s*u[::qs,0],  s*u[::qs,1],  s*u[::qs,2],
                    color='tab:red',
                    label='Thrust')
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
        fig0 = plt.figure()
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
    fig0 = plt.gcf()
    if save_fig_file is not None:
        fig0.savefig(save_fig_file, dpi=300)
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
