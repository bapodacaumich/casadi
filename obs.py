from stl import mesh
from os import getcwd
from os.path import join
from numpy import arange
from numpy.linalg import norm

def ic_sphere(x0, xf, s_r = 1.0, x0_offset=0.6, x1_offset=0.5, x2_factor=0.5):
    """
    generate sphere between x0 and xf with small amount of offset
    """
    s_x0 = x0[0].__float__() + x0_offset
    s_x1 = x0[1].__float__() + x1_offset
    s_x2 = (xf[2].__float__() - x0[2].__float__())*x2_factor
    # s_x2 = x0[2].__float__() + x2_offset
    return [s_x0, s_x1, s_x2, s_r]

def ic_circle(x0, xf, s_r=1.0, x1_factor=0.5):
    """
    generate circle obstacle btween x0 and xf
    """
    s_x0 = x0[0].__float__()
    s_x1 = (xf[1].__float__() - x0[1].__float__())*x1_factor
    return [s_x0, s_x1, s_r]

def load_mesh(mesh_file, show=False):
    """
    import mesh .stl file
    returns mesh face normals and single point on face to establish plane position
    """

    if show: print('Importing mesh: ', mesh_file)
    mesh_file = join(getcwd(), mesh_file)
    str_mesh = mesh.Mesh.from_file(mesh_file)

    return str_mesh.normals/norm(str_mesh.normals), str_mesh.v0