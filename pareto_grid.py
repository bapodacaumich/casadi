from casadi_opti_station import ocp_station_knot
import numpy as np
from os.path import join
from os import getcwd
from sys import argv
from utils import num2str

def generate_pareto_grid(knot_range=(0.1, 100, 11), 
                         fuel_range=(0.1, 100, 11),
                         thrust_limit=0.5,
                         save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions')):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.1, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.1 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
    """
    print('Knot Weight Range: ', knot_range)
    print('Fuel Weight Range: ', knot_range)

    knot_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    fuel_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])

    for kw in knot_weights:
        for fw in fuel_weights:
            # parse kw and fw into a filename string
            k_str = num2str(kw)
            f_str = num2str(fw)
            filestr = 'k_' + k_str + '_f_' + f_str
            print('Save path: ', join(save_dir, filestr + '_X.csv'))

            ocp_station_knot(save_folder=save_dir, 
                             save_path=filestr,
                             thrust_limit=thrust_limit, 
                             k_weight=kw, 
                             p_weight=0.1, 
                             f_weight=fw,
                             closest_knot=True
                             )

if __name__ == "__main__":
    if len(argv) > 1: thrust_limit_input = float(argv[1])
    else: thrust_limit_input = 0.5
    if len(argv) > 2: soln_folder = argv[2]
    else: soln_folder = 'pareto_front_solutions'
    if len(argv) > 5: knot_range_input = (float(argv[3]), float(argv[4]), int(argv[5]))
    else: knot_range_input = (0.1, 100, 11)
    if len(argv) > 8: fuel_range_input = (float(argv[6]), float(argv[7]), int(argv[8]))
    else: fuel_range_input = (0.1, 100, 11)
    generate_pareto_grid(knot_range=knot_range_input, fuel_range=fuel_range_input,
                         thrust_limit=thrust_limit_input, save_dir=join(getcwd(), 'ocp_paths', soln_folder))