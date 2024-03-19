from casadi_opti_station import ocp_parallel
from os.path import join
from os import getcwd, listdir
from sys import argv
from utils import num2str
from multiprocessing import Pool

def generate_soln_parallel(thrust_limit=1.0,
                           knot_weight=10,
                           path_weight=0.1,
                           fuel_weight=10,
                           num_processes=4,
                           save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions'),
                           ):
    """generate a solution for each path file in ccp_paths

    Args:
        thrust_limit (float, optional): thrust limit for actions. Defaults to 1.0
        knot_weight (float, optional): weighting for the knot point cost term of the cost function
        path_weight (float, optional): weighting for the path length cost term of the cost function
        fuel_weight (float, optional): weighting for the fuel cost term of the cost function
        num_processes (int, optional): number of processes to run in parallel
        save_dir (path): path to save solutions
    """

    arg_list = []
    for file in listdir(join(getcwd(), 'ccp_paths')):
        input_local = (file[4] == '_')

        args = (save_dir,
                None,
                thrust_limit,
                knot_weight,
                path_weight,
                fuel_weight,
                True,
                file[:4],
                input_local
                )

        arg_list.append(args)

    with Pool(num_processes) as p:
        r = list(p.imap(ocp_parallel, arg_list))

if __name__ == "__main__":
    save_folder='all_ccp'
    generate_soln_parallel(thrust_limit=1.0,
                           knot_weight=10,
                           path_weight=0.1,
                           fuel_weight=10,
                           save_dir=join(getcwd(), 'ocp_paths', save_folder)
                           )