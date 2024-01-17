import numpy as np
import matplotlib.pyplot as plt
from os.path import join, normpath, basename, exists
from os import getcwd, mkdir
from utils import filter_path_na, compute_time_intervals
from sys import argv

def load_solution(file_dir=join(getcwd(), 'ocp_paths', 'thrust_test'),
                  thrust_str='0_80'
                  ):
    """
    load solution files -- X (state), U (actions), and t (time vector)
    filename - file and directory
    """

    # load solutions
    X = np.loadtxt(join(file_dir, '1.5m_X_' + thrust_str + '.csv'), delimiter=' ')
    U = np.loadtxt(join(file_dir, '1.5m_U_' + thrust_str + '.csv'), delimiter=' ')
    t = np.loadtxt(join(file_dir, '1.5m_t_' + thrust_str + '.csv'), delimiter=' ')

    return X, U, t

def compute_objective_costs(X,
                            U,
                            knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')
                            ):
    """
    compute objective costs for pareto function from state and actions
    X - state vector (N+1, 6) [x, y, z, xdot, ydot, zdot]
    U - action vector (N, 3) [Fx, Fy, Fz]
    """

    # knot cost
    velocity = 0.2
    n_timesteps = 400
    path = np.loadtxt(knotfile, delimiter=',') # load original knot file (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans
    _, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    knot_cost = 0
    for i, k in enumerate(knot_idx):
        knot_cost += np.sum((X[k,:3].T - knots[i,:3])**2)

    # path cost (path length)
    path_cost = np.sum(np.sqrt(np.sum((X[1:,:] - X[:-1,:])**2, axis=1)))

    # fuel cost
    g0 = 9.81
    Isp = 80
    fuel_cost = np.sum(U**2/g0/Isp)*1000 # convert kg to grams

    return fuel_cost, knot_cost, path_cost

def plot_pareto_front(fuel_costs, knot_costs, path_costs, thrust_values, save_file=None):
    """
    plotting pareto front from data
    """

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)

    # fuel_costs against knot_costs
    ax0.plot(fuel_costs, knot_costs, 'rx')
    ax0.set_title('Pareto Front: Fuel and Knot Point Costs \n with annotated Thrust Limits') 
    ax0.set_xlabel('Fuel Cost (g)')
    ax0.set_ylabel('Knot Point Cost (m)')
    annotate_thrust(ax0, fuel_costs, knot_costs, thrust_values)

    # fuel_costs against path_costs
    ax1.plot(fuel_costs, path_costs, 'rx')
    ax1.set_title('Pareto Front: Fuel and Path Costs \n with annotated Thrust Limits')
    ax1.set_xlabel('Fuel Cost (g)')
    ax1.set_ylabel('Path Length (m)')
    annotate_thrust(ax1, fuel_costs, path_costs, thrust_values)

    # knot_costs against path_costs
    ax2.plot(knot_costs, path_costs, 'rx')
    ax2.set_title('Pareto Front: Knot Point and Path Costs \n with annotated Thrust Limits')
    ax2.set_xlabel('Knot Point Cost (m)')
    ax2.set_ylabel('Path Length (m)')
    annotate_thrust(ax2, knot_costs, path_costs, thrust_values)

    plt.tight_layout()
    if save_file is not None: fig.savefig(save_file, dpi=300)
    plt.show()

def annotate_thrust(ax, x, y, thrust_values):
    """
    annotate thrust value at each location (list)
    """
    for i, tv in enumerate(thrust_values):
        ax.text(x[i], y[i], str(tv) + ' N')

def generate_pareto_front(knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                          solution_dir=join(getcwd(), 'ocp_paths', 'thrust_test_k_1_p_1_f_1'),
                          start_thrust=0.5,
                          end_thrust=1.5
                          ):
    """
    generate the pareto front for set of solutions
    """
    plot_file_save=join(getcwd(), 'pareto_front', basename(normpath(solution_dir)))
    fuel_costs = []
    knot_costs = []
    path_costs = []
    # thrust_iter = ['0_20', '0_40', '0_60', '0_80', '1_00']
    # thrust_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    thrust_values = [x/10 for x in range(int(start_thrust*10),int(end_thrust*10)+1)]
    thrust_iter = [str(x)[0] + '_' + str(round((x%1)*10))[0] + str(round(((x*10)%1)*10))[0] for x in thrust_values] # '0_20' through '1_00'
    for thrust_str_value in thrust_iter:
        X, U, t = load_solution(file_dir=solution_dir, thrust_str=thrust_str_value)
        fuel_cost, knot_cost, path_cost = compute_objective_costs(X, U, knotfile)
        fuel_costs.append(fuel_cost)
        knot_costs.append(knot_cost)
        path_costs.append(path_cost)

    fuel_costs = np.array(fuel_costs)
    knot_costs = np.array(knot_costs)
    path_costs = np.array(path_costs)

    if not exists(plot_file_save): mkdir(plot_file_save)
    plot_pareto_front(fuel_costs, knot_costs, path_costs, thrust_values,
                      save_file=join(plot_file_save, 'pareto_front_' + thrust_iter[0] + '_to_' + thrust_iter[-1]))
    return

if __name__ == '__main__':

    if len(argv) > 1: k_weight = argv[1] # string
    else: k_weight = '1'
    if len(argv) > 2: p_weight = argv[2] # string
    else: p_weight = '1'
    if len(argv) > 3: f_weight = argv[3] # string
    else: f_weight = '1'
    if len(argv) > 4: start_thrust_input = float(argv[4])
    else: start_thrust_input=0.2
    if len(argv) > 5: end_thrust_input = float(argv[5])
    else: end_thrust_input=1.0

    solution_directory = join(getcwd(),
                              'ocp_paths',
                              'thrust_test_k_' + k_weight + '_p_' + p_weight + '_f_' + f_weight
                              )

    generate_pareto_front(solution_dir=solution_directory,
                          start_thrust=start_thrust_input,
                          end_thrust=end_thrust_input)
