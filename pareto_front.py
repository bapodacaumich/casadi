import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import getcwd
from utils import filter_path_na, compute_time_intervals

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
    path_cost = np.sqrt(np.sum(X[1:,:] - X[:-1,:]))

    # fuel cost
    g0 = 9.81
    Isp = 80
    fuel_cost = np.sum(U**2/g0/Isp)

    return fuel_cost, knot_cost, path_cost

def plot_pareto_front(fuel_costs, knot_costs, path_costs, save_file=None):
    """
    plotting pareto front from data
    """

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)

    # fuel_costs against knot_costs
    ax0.plot(fuel_costs, knot_costs, 'rx')
    ax0.set_title('Pareto Front: Fuel and Knot Point Costs')
    ax0.set_xlabel('Fuel Cost (g)')
    ax0.set_ylabel('Knot Point Cost (m)')

    # fuel_costs against path_costs
    ax1.plot(fuel_costs, path_costs, 'rx')
    ax1.set_title('Pareto Front: Fuel and Path Costs')
    ax1.set_xlabel('Fuel Cost (g)')
    ax1.set_ylabel('Path Length (m)')

    # knot_costs against path_costs
    ax2.plot(fuel_costs, path_costs, 'rx')
    ax2.set_title('Pareto Front: Knot Point and Path Costs')
    ax2.set_xlabel('Knot Point Cost (m)')
    ax2.set_ylabel('Path Length (m)')

    plt.tight_layout()
    if save_file is not None: fig.savefig(save_file, dpi=300)
    plt.show()

def generate_pareto_front(knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                          solution_dir=join(getcwd(), 'ocp_paths', 'thrust_test'),
                          plot_file_save=join(getcwd(), 'pareto_front', 'thrust_test')
                          ):
    """
    generate the pareto front for set of solutions
    """
    fuel_costs = []
    knot_costs = []
    path_costs = []
    thrust_iter = ['0_20', '0_40', '0_60', '0_80', '1_00']
    for thrust_str_value in thrust_iter:
        X, U, t = load_solution(file_dir=solution_dir, thrust_str=thrust_str_value)
        fuel_cost, knot_cost, path_cost = compute_objective_costs(X, U, knotfile)
        fuel_costs.append(fuel_cost)
        knot_costs.append(knot_cost)
        path_costs.append(path_cost)

    fuel_costs = np.array(fuel_costs)
    knot_costs = np.array(knot_costs)
    path_costs = np.array(path_costs)

    plot_pareto_front(fuel_costs, knot_costs, path_costs, save_file=join(plot_file_save, 'pareto_front_' + thrust_iter[0] + '_to_' + thrust_iter[1]))

    return

if __name__ == '__main__':
    generate_pareto_front()
