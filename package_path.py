import numpy as np
import sys
from utils import compute_time_intervals, filter_path_na, process_data
from os.path import join, exists
from os import getcwd, listdir, mkdir
from utils import load_path_data, process_data


def package_data(soln_dir=join(getcwd(), 'ocp_paths', 'final'),
                 knot_dir=join(getcwd(), 'ccp_paths'),
                 save_dir=join(getcwd(), 'final')):
    """
    integrate orientation data with state data and concatenate with time vector
    """
    if not exists(save_dir): mkdir(save_dir)

    for file in listdir(soln_dir):
        if file[-5] != 'X': continue
        d_str = file[:3]
        local = file[5] == 'l'
        X = np.loadtxt(join(soln_dir, file), delimiter=' ')
        U = np.loadtxt(join(soln_dir, file[:-5] + 'U.csv'), delimiter=' ')
        t = np.loadtxt(join(soln_dir, file[:-5] + 't.csv'), delimiter=' ')
        for kfile in listdir(knot_dir):
            if (d_str == kfile[:3]) and not (local ^ (kfile[5] == 'l')): # distance string matches
                path = np.loadtxt(join(knot_dir, kfile), delimiter=',')
                knots = filter_path_na(path) # get rid of configurations with nans
                print('Loading: ', join(soln_dir, file), ' and ', join(knot_dir, kfile))

        data = process_data(knots, X, t)
        if local: filestr = d_str + '_local.csv'
        else: filestr = d_str + '.csv'
        np.savetxt(join(save_dir, filestr), data, delimiter=',')

if __name__ == "__main__":
    package_data()