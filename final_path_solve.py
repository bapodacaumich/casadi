from casadi_opti_station import ocp_parallel
from os.path import join, exists
from os import getcwd, listdir
from sys import argv
from utils import num2str
from multiprocessing import Pool

def generate_soln_parallel(thrust_limit=1.0,
                           knot_weight_local=10,
                           fuel_weight_local=10,
                           knot_weight_global=10,
                           fuel_weight_global=10,
                           path_weight=0.1,
                           num_processes=1,
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

        if not file.startswith('0.5m_local'): continue

        if input_local:
            args = (save_dir,
                    None,
                    thrust_limit,
                    knot_weight_local,
                    path_weight,
                    fuel_weight_local,
                    True,
                    file[:4],
                    input_local
                    )
        else:
            args = (save_dir,
                    None,
                    thrust_limit,
                    knot_weight_global,
                    path_weight,
                    fuel_weight_global,
                    True,
                    file[:4],
                    input_local
                    )

        # if input_local:
        #     soln_path = join(save_dir, file[:4] + '_local_X.csv') 
        # else:
        #     soln_path = join(save_dir, file[:4] + '_X.csv')
        # if not exists(soln_path):
        #     # if file[:4] == '4.5m' or file[:4] == '0.5m':
        #     arg_list.append(args)
        # else: print('skipping: ', soln_path)
        # if file[:4] == '0.5m' and not input_local:
        arg_list.append(args)


    # run sequentially (one process at a time)
    for arg in arg_list:
        print('Current Process: ', arg)
        ocp_parallel(arg)
    # run in parallel
    # with Pool(num_processes) as p:
    #     r = list(p.imap(ocp_parallel, arg_list))

if __name__ == "__main__":
    save_folder='all_ccp'
    generate_soln_parallel(thrust_limit=1.0,
                           knot_weight_local=100,
                           fuel_weight_local=1,
                           knot_weight_global=10,
                           fuel_weight_global=10,
                           path_weight=0.1,
                           save_dir=join(getcwd(), 'ocp_paths', save_folder)
                           )