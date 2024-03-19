from casadi_opti_station import ocp_station_knot, ocp_parallel
import numpy as np
from os.path import join
from os import getcwd
from sys import argv
from utils import num2str
from tqdm import tqdm
from multiprocessing import Pool

def generate_pareto_grid_parallel(knot_range=(0.001, 100, 12),
                                  fuel_range=(0.001, 100, 12),
                                  thrust_limit=0.5,
                                  save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions'),
                                  view_distance_str='1.5m'
                                  ):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """
    knot_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    fuel_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])

    print('BUILDING PARALLEL TASK')
    print('Knot Weights: ', knot_weights)
    print('Fuel Weights: ', fuel_weights)

    arg_list = []
    for kw in knot_weights:
        for fw in fuel_weights:
            if kw < 0.2: continue
            if kw < 1 and fw < 0.02: continue
            # parse kw and fw into a filename string
            k_str = num2str(kw)
            f_str = num2str(fw)
            filestr = 'k_' + k_str + '_f_' + f_str
            print('Current Weight Config: ', filestr)
            print('Save path: ', join(save_dir, filestr + '_X.csv'))

            args = (save_dir,
                   filestr,
                   thrust_limit,
                   kw,
                   0.1,
                   fw,
                   True,
                   view_distance_str,
                   False
                   )

            arg_list.append(args)

    with Pool(2) as p:
        r = list(tqdm(p.imap(ocp_parallel, arg_list), total=len(arg_list)))

def generate_pareto_grid(knot_range=(0.001, 100, 12),
                         fuel_range=(0.001, 100, 12),
                         thrust_limit=0.5,
                         save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions'),
                         view_distance_str='1.5m'
                         ):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """
    knot_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    fuel_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])

    print('Knot Weights: ', knot_weights)
    print('Fuel Weights: ', fuel_weights)

    for kw in knot_weights:
        for fw in fuel_weights:
            if kw < 0.01: continue
            if kw < 0.01 and fw < 50: continue
            # parse kw and fw into a filename string
            k_str = num2str(kw)
            f_str = num2str(fw)
            filestr = 'k_' + k_str + '_f_' + f_str
            print('Current Weight Config: ', filestr)
            print('Save path: ', join(save_dir, filestr + '_X.csv'))

            ocp_station_knot(save_folder=save_dir,
                             save_path=filestr,
                             thrust_limit=thrust_limit,
                             k_weight=kw,
                             p_weight=0.1,
                             f_weight=fw,
                             closest_knot=True,
                             view_distance=view_distance_str
                             )

if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == '-h': print('python pareto_grid.py thrust_limit soln_folder knot_weight_range fuel_weight_range',
                                                '\n  thrust_limit=0.5',
                                                '\n  soln_folder=pareto_front_solutions',
                                                '\n  knot_weight_range=(0.1, 100, 8)',
                                                '\n  fuel_weight_range=(0.1, 100, 8)')
    elif len(argv) > 1 and argv[1] == '-p':
        if len(argv) > 2: thrust_limit_input = float(argv[2])
        else: thrust_limit_input = 1.0
        if len(argv) > 3: soln_folder = argv[3]
        else: soln_folder = 'ck_1.0'
        if len(argv) > 6: knot_range_input = (float(argv[4]), float(argv[5]), int(argv[6]))
        else: knot_range_input = (0.01, 100, 9)
        if len(argv) > 9: fuel_range_input = (float(argv[7]), float(argv[8]), int(argv[9]))
        else: fuel_range_input = (0.01, 100, 9)
        generate_pareto_grid_parallel(knot_range=knot_range_input,
                                      fuel_range=fuel_range_input,
                                      thrust_limit=thrust_limit_input,
                                      save_dir=join(getcwd(), 'ocp_paths', soln_folder),
                                      view_distance_str='1.5m'
                                      )
    else:
        if len(argv) > 1: thrust_limit_input = float(argv[1])
        else: thrust_limit_input = 1.0
        if len(argv) > 2: soln_folder = argv[2]
        else: soln_folder = 'ck_1.0'
        if len(argv) > 5: knot_range_input = (float(argv[3]), float(argv[4]), int(argv[5]))
        else: knot_range_input = (0.01, 100, 9)
        if len(argv) > 8: fuel_range_input = (float(argv[6]), float(argv[7]), int(argv[8]))
        else: fuel_range_input = (0.01, 100, 9)
        generate_pareto_grid(knot_range=knot_range_input, fuel_range=fuel_range_input,
                             thrust_limit=thrust_limit_input, save_dir=join(getcwd(), 'ocp_paths', soln_folder))
