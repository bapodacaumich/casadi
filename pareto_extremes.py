from casadi_opti_station import ocp_station_knot
import numpy as np
from os.path import join
from os import getcwd
from sys import argv
from utils import num2str

def run_extremes(knot_range=(0.001, 100),
                 fuel_range=(0.001, 100),
                 thrust_limit=1.0,
                 save_dir=join(getcwd(), 'ocp_paths', 'pareto_extremes'),
                 view_distance_str='1.5m'
                 ):
    """run the extremes of the pareto front in order to determine normalization factors for cost function terms.

    Args:
        knot_range (tuple, optional): _description_. Defaults to (0.001, 100).
        fuel_range (tuple, optional): _description_. Defaults to (0.001, 100).
        thrust_limit (float, optional): _description_. Defaults to 1.0.
        save_dir (_type_, optional): _description_. Defaults to join(getcwd(), 'ocp_paths', 'extremes').
        view_distance_str (str, optional): _description_. Defaults to '1.5m'.
    """
    print('Knot Weight Extremes: ', knot_range)
    print('Fuel Weight Extremes: ', fuel_range)

    knot_weights = np.array([knot_range[1], knot_range[1], knot_range[0], knot_range[0]])
    fuel_weights = np.array([fuel_range[0], fuel_range[1], fuel_range[1], fuel_range[0]])

    for kw, fw in zip(knot_weights, fuel_weights):
        k_str = num2str(kw)
        f_str = num2str(fw)

        filestr = 'k_' + k_str + '_f_' + f_str
        print('Save path: ', join(save_dir, filestr + '_X'))

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
    if len(argv) > 1: thrust_limit_input = float(argv[1])
    else: thrust_limit_input = 1.0
    if len(argv) > 2: soln_folder = argv[2]
    else: soln_folder = 'pareto_extremes'
    if len(argv) > 4: knot_range_input = (float(argv[3]), float(argv[4]))
    else: knot_range_input = (0.001, 100)
    if len(argv) > 6: fuel_range_input = (float(argv[5]), float(argv[6]))
    else: fuel_range_input = (0.001, 100)
    run_extremes(knot_range=knot_range_input, fuel_range=fuel_range_input,
                 thrust_limit=thrust_limit_input, save_dir=join(getcwd(), 'ocp_paths', soln_folder))