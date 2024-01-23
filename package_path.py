import numpy as np
import sys
from utils import compute_time_intervals, filter_path_na, process_data
from os.path import join
from os import getcwd
from utils import load_path_data, process_data


def package_data(save_path_file=join(getcwd(), 'final_paths', '1.5m_path.csv')):
    """
    integrate orientation data with state data and concatenate with time vector
    """
    knots, x, t = load_path_data()

    data = process_data(knots, x, t)

    np.savetxt(save_path_file, data, delimiter=',')

if __name__ == "__main__":
    package_data()