from os import getcwd
from os.path import join
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np

def station():
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    scale = np.array([0])
    for i in range(15):
        meshfile = join(getcwd(), 'model', 'convex_detailed_station', str(i) + '.stl')

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(meshfile)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
        normalized_normals = your_mesh.normals/norm(your_mesh.normals, axis=1, keepdims=True)
        axes.quiver(your_mesh.v0[:,0], your_mesh.v0[:,1], your_mesh.v0[:,2],
                    normalized_normals[:,0], normalized_normals[:,1], normalized_normals[:,2],
                    color='tab:red',
                    length=0.3
                    )

        # Auto scale to the mesh size
        scale = np.concatenate((scale, your_mesh.points.flatten()))

    axes.auto_scale_xyz(scale, scale, scale)

    # axis labels
    axes.set_xlabel('X Axis')
    axes.set_ylabel('Y Axis')
    axes.set_zlabel('Z Axis')

    # Show the plot to the screen
    plt.show()


def mockup():
    files = ['mercury_convex.stl', 'gemini_convex.stl', 'solar_convex.stl', 'apollo_convex.stl']

    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    scale = np.array([0])
    for f in files:
        meshfile = join(getcwd(), 'model', 'mockup', f)

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(meshfile)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
        normalized_normals = your_mesh.normals/norm(your_mesh.normals, axis=1, keepdims=True)
        axes.quiver(your_mesh.v0[:,0], your_mesh.v0[:,1], your_mesh.v0[:,2],
                    normalized_normals[:,0], normalized_normals[:,1], normalized_normals[:,2],
                    color='tab:red',
                    length=0.3
                    )

        # Auto scale to the mesh size
        scale = np.concatenate((scale, your_mesh.points.flatten()))

    axes.auto_scale_xyz(scale, scale, scale)

    # axis labels
    axes.set_xlabel('X Axis')
    axes.set_ylabel('Y Axis')
    axes.set_zlabel('Z Axis')

    # Show the plot to the screen
    plt.show()

def mercury():
    meshfile = join(getcwd(), 'model', 'mockup', 'mercury_convex.stl')
    # str_mesh = mesh.Mesh.from_file(meshfile)
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(meshfile)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    # normalized_normals = your_mesh.normals/norm(your_mesh.normals, axis=1, keepdims=True)
    # axes.quiver(your_mesh.v0[:,0], your_mesh.v0[:,1], your_mesh.v0[:,2],
    #             normalized_normals[:,0], normalized_normals[:,1], normalized_normals[:,2],
    #             color='tab:red',
    #             length=0.3
    #             )

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # axis labels
    axes.set_xlabel('X Axis')
    axes.set_ylabel('Y Axis')
    axes.set_zlabel('Z Axis')

    # Show the plot to the screen
    plt.show()

if __name__ == "__main__":
    station()