from casadi_opti_station import ocp_station_knot
import numpy as np
from os.path import join
from os import getcwd
from sys import argv

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

    knot_weights = np.linspace(knot_range[0], knot_range[1], knot_range[2])
    fuel_weights = np.linspace(fuel_range[0], fuel_range[1], fuel_range[2])

    for kw in knot_weights:
        for fw in fuel_weights:
            # parse kw and fw into a filename string
            if str(kw)[-1] == '0': k_str = str(kw)[:-2]
            else: k_str = str(kw)[:-2] + '_' + str(kw)[-1]
            if str(fw)[-1] == '0': f_str = str(fw)[:-2]
            else: f_str = str(fw)[:-2] + '_' + str(fw)[-1]
            filestr = k_str + '_' + f_str
            print('Save path: ', join(save_dir, filestr + '_X.csv'))

            ocp_station_knot(save_folder=save_dir, 
                             save_path=filestr,
                             thrust_limit=thrust_limit, 
                             k_weight=kw, 
                             p_weight=0.1, 
                             f_weight=fw)

if __name__ == "__main__":
    if len(argv) > 1: thrust_limit_input = float(argv[1])
    else: thrust_limit_input = 0.2
    generate_pareto_grid(thrust_limit=thrust_limit_input)