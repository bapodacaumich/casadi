import os 
from os import getcwd, listdir
from os.path import join
from stl import mesh
from mpl_toolkits import mplot3d
from utils import filter_path_na
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import numpy as np
from sys import argv


def plot_solution(station=False, mockup=False, soln_dir='thrust_test_k_1_p_1_f_1', soln_file=None, thrust_limit=0.2, local=False, distance='1.5m'):
    if station:
        for file in listdir(join(getcwd(), 'ccp_paths')):
            if (distance == file[:4]):
                if not ((file[5] == 'l') ^ local):
                    knotfile=join(getcwd(), 'ccp_paths', file)
        # knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')
        # load knot points
        path = np.loadtxt(knotfile, delimiter=',') # (N, 6)
        knots = filter_path_na(path) # get rid of configurations with nans
    if mockup: 
        knots = np.array([[1.0, -1.0, 0.5],
                            [1.0, 1.0, 0.5],
                            [-1.0, 1.0, 0.5],
                            [-1.0, -1.0, 0.5]])

    # load station offset
    translation = np.loadtxt('translate_station.txt', delimiter=',').reshape(1,1,3)

    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')
    # axes = mplot3d.Axes3D(figure)

    scale = np.array([0])
    if station:
        for i in range(15):
            meshfile = join(getcwd(), 'model', 'convex_detailed_station', str(i) + '.stl')

            # Load the STL files and add the vectors to the plot
            your_mesh = mesh.Mesh.from_file(meshfile)
            vectors = your_mesh.vectors + translation
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors))
            wf = vectors.reshape(-1, 3)
            axes.plot(wf[:,0], wf[:,1], wf[:,2], 'k')

            # Auto scale to the mesh size
            scale = np.concatenate((scale, your_mesh.points.flatten()))
    elif mockup:
        files = ['apollo_convex.stl', 'gemini_convex.stl', 'mercury_convex.stl', 'solar_convex.stl']
        for f in files:
            meshfile = join(getcwd(), 'model', 'mockup', f)

            # Load the STL files and add the vectors to the plot
            your_mesh = mesh.Mesh.from_file(meshfile)
            vectors = your_mesh.vectors
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors))
            wf = vectors.reshape(-1, 3)
            axes.plot(wf[:,0], wf[:,1], wf[:,2], 'k')

            # Auto scale to the mesh size
            scale = np.concatenate((scale, your_mesh.points.flatten()))

    # axes.auto_scale_xyz(scale, scale, scale)

    # plot knot points
    axes.plot(knots[:,0], knots[:,1], knots[:,2],'k--')
    axes.scatter(knots[:,0], knots[:,1], knots[:,2])

    # # load path
    if soln_file is None:
        thrust_str = str(thrust_limit)[0] + '_' + str((thrust_limit%1)*10)[0] + str(((thrust_limit*10)%1)*10)[0]
        X = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_X_' + thrust_str + '.csv'), delimiter=' ')
        U = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_U_' + thrust_str + '.csv'), delimiter=' ')
        t = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_t_' + thrust_str + '.csv'), delimiter=' ')
        soln_file = '1.5m_' + thrust_str
    else:
        X = np.loadtxt(soln_file + '_X.csv', delimiter=' ')
        U = np.loadtxt(soln_file + '_U.csv', delimiter=' ')
        t = np.loadtxt(soln_file + '_t.csv', delimiter=' ')

    # plot path
    axes.plot(X[:,0], X[:,1], X[:,2], 'k-')
    qs = 1
    s = 5 # scale factor
    axes.quiver(X[:-1:qs,0],X[:-1:qs,1],X[:-1:qs,2],
                s*U[::qs,0],s*U[::qs,1],s*U[::qs,2],
                color='tab:red',
                label='Thrust')

    # axis labels
    axes.set_xlabel('X Axis')
    axes.set_ylabel('Y Axis')
    axes.set_zlabel('Z Axis')

    fig, ax = plt.subplots()
    ax.plot(t[:-1], np.sqrt(np.sum(U**2, axis=1)))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust Magnitude')
    ax.set_title('Thrust Magnitude, Duration = ' + str(t[-1])[:6] + 's')

    # # plot fig
    # plt.show()

    # savefig
    savefile = os.path.basename(os.path.normpath(soln_file))
    save_dpi = 300
    # print('saving: ', savepath)
    # figure.savefig(savepath, dpi=1000)

    view_num = 0
    ax.view_init(elev=30, azim=30)
    plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)

    # view from underneath
    ax.view_init(elev=-30, azim=30)
    view_num += 1
    plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)

    # rotate around
    ax.view_init(elev=30, azim=120)
    view_num += 1
    plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)

if __name__ == '__main__':
    # python plot_solution.py 0.2 1 1 1
    if argv[1] == '-h':
        print('Example Args:\npython plot_solution.py 0.2 1 1 1')
        print('python plot_solution.py -f pareto_extremes k_1_0_f_1_0 1.5m')
    elif argv[1] == '-d':
        soln_file_input = join(getcwd(), 'ocp_paths', argv[2])
        if argv[2][-1] == 'l': local_input = True
        else: local_input = False
        plot_solution(station=True, thrust_limit=1.7, soln_file=soln_file_input, distance=argv[2][:4], local=local_input)
    elif argv[1] == '-f':
        soln_file_input = join(getcwd(), 'ocp_paths', argv[2], argv[3])
        if len(argv) > 4: distance_input = argv[4]
        else: distance_input = '1.5m'
        plot_solution(station=True, thrust_limit=1.0, soln_file=soln_file_input, distance=distance_input, local=False)
    else:
        if len(argv) > 1: thrust_input = float(argv[1])
        else: thrust_input = 0.2 # float

        if len(argv) > 2: k_weight = argv[2] # string
        else: k_weight = '1'
        if len(argv) > 3: p_weight = argv[3] # string
        else: p_weight = '1'
        if len(argv) > 4: f_weight = argv[4] # string
        else: f_weight = '1'

        soln_dir = 'thrust_test_k_' + k_weight + '_p_' + p_weight + '_f_' + f_weight
        plot_solution(station=True, thrust_limit=thrust_input, soln_dir=soln_dir)
