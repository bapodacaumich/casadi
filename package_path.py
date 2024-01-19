import numpy as np
import sys
from utils import compute_time_intervals, filter_path_na
from os.path import join
from os import getcwd
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt


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

def package_data(save_path_file=join(getcwd(), 'final_paths', '1.5m_path.csv')):
    """
    integrate orientation data with state data and concatenate with time vector
    """
    knots, x, t = load_path_data()

    velocity = 0.2
    n_timesteps = 400
    dt, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)

    t_load = np.insert(np.cumsum(dt),0,0)
    assert (t_load == t).all()
    t = t.reshape((-1,1))

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

    data = np.concatenate((x,t), axis=1)
    np.savetxt(save_path_file, data, delimiter=',')

if __name__ == "__main__":
    package_data()