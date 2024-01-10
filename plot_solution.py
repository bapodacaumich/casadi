from os import getcwd
from os.path import join
from stl import mesh
from mpl_toolkits import mplot3d
from utils import filter_path_na
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np


def plot_solution(station=False, mockup=False):
    if station:
        knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')
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
    axes = mplot3d.Axes3D(figure)

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

    axes.auto_scale_xyz(scale, scale, scale)

    # plot knot points
    axes.plot(knots[:,0], knots[:,1], knots[:,2],'rx')

    # # load path
    # X = np.loadtxt(join(getcwd(), 'ocp_paths', '1.5m_X.csv'), delimiter=' ')
    # U = np.loadtxt(join(getcwd(), 'ocp_paths', '1.5m_U.csv'), delimiter=' ')
    # t = np.loadtxt(join(getcwd(), 'ocp_paths', '1.5m_t.csv'), delimiter=' ')
    X = np.loadtxt(join(getcwd(), 'ocp_paths', '6340sec', '1.5m_X.csv'), delimiter=' ')
    U = np.loadtxt(join(getcwd(), 'ocp_paths', '6340sec', '1.5m_U.csv'), delimiter=' ')
    t = np.loadtxt(join(getcwd(), 'ocp_paths', '6340sec', '1.5m_t.csv'), delimiter=' ')

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

    # plot fig
    plt.show()

    # savefig
    # savepath = join(getcwd(), 'gcf.png')
    # print('saving: ', savepath)
    # figure.savefig(savepath, dpi=300)

if __name__ == '__main__':
    plot_solution(station=True)