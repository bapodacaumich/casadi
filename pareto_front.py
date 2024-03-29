import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from os.path import join, normpath, basename, exists
from os import getcwd, mkdir
from utils import filter_path_na, compute_time_intervals, compute_path_coverage, num2str
from sys import argv
from tqdm import tqdm

def load_solution(file_dir=join(getcwd(), 'ocp_paths', 'thrust_test'),
                  thrust_str=None,
                  kf_weights=None
                  ):
    """
    load solution files -- X (state), U (actions), and t (time vector)
    filename - file and directory
    """

    X, U, t = None, None, None
    # load solutions
    if thrust_str is not None:
        X = np.loadtxt(join(file_dir, '1.5m_X_' + thrust_str + '.csv'), delimiter=' ')
        U = np.loadtxt(join(file_dir, '1.5m_U_' + thrust_str + '.csv'), delimiter=' ')
        t = np.loadtxt(join(file_dir, '1.5m_t_' + thrust_str + '.csv'), delimiter=' ')
    else:
        X = np.loadtxt(join(file_dir, 'k_' + kf_weights[0] + '_f_' + kf_weights[1] + '_X.csv'), delimiter=' ')
        U = np.loadtxt(join(file_dir, 'k_' + kf_weights[0] + '_f_' + kf_weights[1] + '_U.csv'), delimiter=' ')
        t = np.loadtxt(join(file_dir, 'k_' + kf_weights[0] + '_f_' + kf_weights[1] + '_t.csv'), delimiter=' ')
        

    return X, U, t

def compute_knot_cost_numpy(X,
                            knots,
                            n_timesteps=400,
                            closest=True,
                            velocity=0.2,
                            square=False
                            ):
    """
    compute the distance between knot points and path using closest knot point formulation (sum of squares)

    Inputs
    ------

        X (numpy array): state vector size (N+1, 6) [x, y, z, xdot, ydot, zdot]

        knots (numpy array): knot points in array size (N_knots, 3) [x, y, z]

        closest (bool): if True use closest point to each knot point for distance computation, if False use old formulation

        velocity (float): assumed velocity for time interval computation

        square (bool): use sum of squares if true, else find actual distances

    Returns
    -------

        knot_cost (float): cumulative distance between path and knot points

    """
    _, knot_idx = compute_time_intervals(knots, velocity, n_timesteps)
    knot_cost = 0

    if closest:
        lastidx = 0
        for ki in range(len(knot_idx)-1):
            closest_dist = np.inf
            for idx in range(lastidx, (knot_idx[ki] + knot_idx[ki+1])//2+1):
                if square: dist = np.sum((knots[ki, :3].reshape((1,-1)) - X[idx, :3])**2) # compare state
                else: dist = np.sqrt(np.sum((knots[ki, :3].reshape((1,-1)) - X[idx, :3])**2)) # compare state
                closest_dist = min(closest_dist, dist)
            knot_cost += closest_dist

    else:
        for i, k in enumerate(knot_idx):
            if square: knot_cost += np.sum((X[k,:3].T - knots[i,:3])**2)
            else: knot_cost += np.sqrt(np.sum((X[k,:3].T - knots[i,:3])**2))

    return knot_cost


def compute_objective_costs(X,
                            U,
                            t,
                            knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                            compute_coverage=True,
                            velocity = 0.2,
                            square=False
                            ):
    """
    compute objective costs for pareto function from state and actions

    Inputs
    ------

        X (numpy array): state vector (N+1, 6) [x, y, z, xdot, ydot, zdot]

        U (numpy array): action vector (N, 3) [Fx, Fy, Fz]

        t (numpy array): time vector (N+1, 1)

        knotfile (String): filepath of knot file including cwd

        compute_coverage (bool): flag to compute coverage ratio

        velocity (float): assumed speed of inspector along path (for computing knot_idx)

        square (bool): if square, use sum of squares (like objective cost fn)

    Return
    ------

        fuel_cost (float): fuel cost in grams

        knot_cost (float): distance from knot points in meters

        path_cost (float): path length (m)

        coverage (float): ratio of station (convex hull) inspected by agent

        path_time (float): path traversal time given input velocity
    """

    # knot cost
    path = np.loadtxt(knotfile, delimiter=',') # load original knot file (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans

    knot_cost = compute_knot_cost_numpy(X, knots, closest=True, square=square)

    # path cost (path length)
    if square: path_cost = np.sum(np.sum((X[1:,:] - X[:-1,:])**2, axis=1))
    else: path_cost = np.sum(np.sqrt(np.sum((X[1:,:] - X[:-1,:])**2, axis=1)))

    # fuel cost
    g0 = 9.81 # acceleration due to gravity
    Isp = 80 # specific impulse of rocket engine
    m = 5.75 # mass of agent
    dt = np.diff(t)
    if square: fuel_cost = np.sum((np.sum(U**2, axis=1)/g0**2/Isp**2)*dt**2)   # convert kg to grams
    else: fuel_cost = np.sum((np.sqrt(np.sum(U**2, axis=1))/g0/Isp)*dt) * 1000 # convert kg to grams

    if compute_coverage: coverage = compute_path_coverage(knots, X, t)
    else: coverage=None

    path_time = t[-1]

    return fuel_cost, knot_cost, path_cost, coverage, path_time

def parse_pareto_front(x, y):
    """isolate datapoints in pareto front assuming lower is better

    Args:
        x (_type_): _description_
        y (_type_): _description_
    """
    xsortarg = np.argsort(x)
    xsort = x[xsortarg]
    ysort = y[xsortarg]
    front = [[xsort[0], ysort[0]]]
    front_idx = [xsortarg[0]]
    rest = []
    for i in range(1, len(x)):
        if ysort[i] < front[-1][1]:
            front.append([xsort[i], ysort[i]])
            front_idx.append(xsortarg[i])
        else: rest.append([xsort[i], ysort[i]])

    return np.array(front)[:,0], np.array(front)[:,1], np.array(rest)[:,0], np.array(rest)[:,1], front_idx

def plot_pareto_front(fuel_costs, knot_costs, path_costs, coverage, thrust_values=None, save_file=None):
    """
    plotting pareto front from data
    """

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(50,30))

    # preprocess
    fuel_fkfront, knot_fkfront, fuel_fkrest, knot_fkrest, fk_idx = parse_pareto_front(fuel_costs, knot_costs)
    fuel_fcfront, cov_fcfront, fuel_fcrest, cov_fcrest, fc_idx = parse_pareto_front(fuel_costs, 1-coverage)
    knot_kcfront, cov_kcfront, knot_kcrest, cov_kcrest, kc_idx = parse_pareto_front(knot_costs, 1-coverage)

    # fuel_costs against knot_costs
    ax0.plot(fuel_fkrest, knot_fkrest, 'rx', label='')
    ax0.plot(fuel_costs[fc_idx], knot_costs[fc_idx], 'bo-', label='fuel-coverage front')
    ax0.plot(fuel_costs[kc_idx], knot_costs[kc_idx], 'go-', label='knot-coverage front')
    ax0.plot(fuel_costs[fk_idx], knot_costs[fk_idx], 'ko-', label='fuel-knot front')
    ax0.set_title('Pareto Front: Fuel and Knot Point Costs with annotated Thrust Limits') 
    ax0.set_xlabel('Fuel Cost (g)')
    ax0.set_ylabel('Knot Point Cost (m)')
    if thrust_values is not None: annotate_thrust(ax0, fuel_costs, knot_costs, thrust_values)

    # fuel_costs against coverage
    ax1.plot(fuel_fcrest, 100*cov_fcrest, 'rx')
    ax1.plot(fuel_costs[fk_idx], 100*(1-coverage[fk_idx]), 'ko-', label='fuel-knot front')
    ax1.plot(fuel_costs[kc_idx], 100*(1-coverage[kc_idx]), 'go-', label='knot-coverage front')
    ax1.plot(fuel_costs[fc_idx], 100*(1-coverage[fc_idx]), 'bo-', label='fuel-coverage front')
    ax1.set_title('Pareto Front: Fuel Costs and Coverage Ratio with annotated Thrust Limits')
    ax1.set_xlabel('Fuel Cost (g)')
    ax1.set_ylabel('Missed Coverage (%)')
    if thrust_values is not None: annotate_thrust(ax1, fuel_costs, coverage, thrust_values)

    # knot_costs against coverage
    ax2.plot(knot_kcrest, 100*cov_kcrest, 'rx')
    ax2.plot(knot_costs[fc_idx], 100*(1-coverage[fc_idx]), 'bo-', label='fuel-coverage front')
    ax2.plot(knot_costs[fk_idx], 100*(1-coverage[fk_idx]), 'ko-', label='fuel-knot front')
    ax2.plot(knot_costs[kc_idx], 100*(1-coverage[kc_idx]), 'go-', label='knot-coverage front')
    ax2.set_title('Pareto Front: Knot Costs and Coverage Ratio with annotated Thrust Limits')
    ax2.set_xlabel('Knot Cost (m)')
    ax2.set_ylabel('Missed Coverage (%)')
    if thrust_values is not None: annotate_thrust(ax2, knot_costs, coverage, thrust_values)

    # # fuel_costs against path_costs
    # ax1.plot(fuel_costs, path_costs, 'rx')
    # ax1.set_title('Pareto Front: Fuel and Path Costs \n with annotated Thrust Limits')
    # ax1.set_xlabel('Fuel Cost (g)')
    # ax1.set_ylabel('Path Length (m)')
    # annotate_thrust(ax1, fuel_costs, path_costs, thrust_values)

    # # knot_costs against path_costs
    # ax2.plot(knot_costs, path_costs, 'rx')
    # ax2.set_title('Pareto Front: Knot Point and Path Costs \n with annotated Thrust Limits')
    # ax2.set_xlabel('Knot Point Cost (m)')
    # ax2.set_ylabel('Path Length (m)')
    # annotate_thrust(ax2, knot_costs, path_costs, thrust_values)

    fig.set_figwidth(15)
    fig.set_figheight(10)
    plt.tight_layout()
    if save_file is not None: fig.savefig(save_file, dpi=300)
    plt.show()
    return fk_idx, fc_idx, kc_idx

def annotate_thrust(ax, x, y, thrust_values):
    """
    annotate thrust value at each location (list)
    """
    for i, tv in enumerate(thrust_values):
        ax.text(x[i], y[i], str(tv) + ' N')

def compute_cost_single_soln(knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                             solution_dir=join(getcwd(), 'ocp_paths', 'pf_0.2'),
                             knot_weight=1,
                             fuel_weight=1,
                             compute_coverage=True
                             ):
    # build file path for queried solution
    kstr = num2str(knot_weight)
    fstr = num2str(fuel_weight)
    file_path = join(solution_dir, 'k_' + kstr + '_f_' + fstr + '_X.csv')

    # if the file exists, compute the costs
    if exists(file_path):
        X, U, t = load_solution(file_dir=solution_dir, kf_weights=(kstr, fstr))
        fuel_cost, knot_cost, path_cost, coverage_ratio, path_time = compute_objective_costs(X,
                                                                                             U,
                                                                                             t, 
                                                                                             knotfile,
                                                                                             compute_coverage=compute_coverage,
                                                                                             square=True
                                                                                             )

        print('Weights and costs: Knot = ', knot_weight, '| Fuel = ', fuel_weight, ' | Fuel Cost: ', fuel_cost, ' | Knot Cost: ', knot_cost, ' Coverage (%): ', coverage_ratio*100)
    else: print('File missing: ', file_path)

def generate_pareto_front_grid(knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'),
                               solution_dir=join(getcwd(), 'ocp_paths', 'pf_0.2'),
                               knot_range=(0.001, 100, 11),
                               fuel_range=(0.001, 100, 11),
                               coverage_input=True
                               ):
    """
    generate the pareto front for set of solutions
    """
    plot_file_save=join(getcwd(), 'pareto_front', basename(normpath(solution_dir)))
    # plot_file_save=join(getcwd(), 'pareto_front', 'pf_1.0_temp')
    if not exists(plot_file_save): mkdir(plot_file_save)
    fuel_costs = []
    knot_costs = []
    path_costs = []
    path_times = []
    coverage = []
    k_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    f_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])
    weight_combs = []
    # for kw in tqdm(k_weights, position=0):
    #     for fw in tqdm(f_weights, leave=False, position=1):
    for kw in k_weights:
        for fw in f_weights:
            file_path = join(solution_dir, 'k_' + num2str(kw) + '_f_' + num2str(fw) + '_X.csv')
            if exists(file_path):
                weight_combs.append((kw, fw))
                X, U, t = load_solution(file_dir=solution_dir, kf_weights=(num2str(kw), num2str(fw)))
                fuel_cost, knot_cost, path_cost, coverage_ratio, path_time = compute_objective_costs(X, U, t, knotfile, compute_coverage=coverage_input)
                # print('\n\nWeight combination (knot, fuel): ', kw, fw, ' Fuel cost: ', fuel_cost, ' Knot Cost: ', knot_cost, ' Coverage: ', coverage_ratio)
                fuel_costs.append(fuel_cost)
                knot_costs.append(knot_cost)
                path_costs.append(path_cost)
                coverage.append(coverage_ratio)
                path_times.append(path_time)
            else: print('File missing: ', file_path)

    fuel_costs = np.array(fuel_costs)
    knot_costs = np.array(knot_costs)
    path_costs = np.array(path_costs)
    if coverage[0] is not None: coverage = np.array(coverage)
    path_times = np.array(path_times)
    
    np.savetxt(join(plot_file_save, 'fcost.csv'), fuel_costs)
    np.savetxt(join(plot_file_save, 'kcost.csv'), knot_costs)
    np.savetxt(join(plot_file_save, 'pcost.csv'), path_costs)
    if coverage[0] is not None: np.savetxt(join(plot_file_save, 'cov.csv'), coverage)
    np.savetxt(join(plot_file_save, 'ptime.csv'), path_times)

    if not exists(plot_file_save): mkdir(plot_file_save)
    fk_idx, fc_idx, kc_idx = plot_pareto_front(fuel_costs, knot_costs, path_costs, coverage,
                                               save_file=join(plot_file_save, 'all_points'))
    return

def pareto_load_plot(cost_dir=join(getcwd(), 'pareto_front','pf_0.2'),
                     solution_dir=join(getcwd(), 'ocp_paths', 'pf_0.2'), 
                     knot_range=(0.001, 100, 11),
                     fuel_range=(0.001, 100, 11)):
    """load costs from pareto front solutions from cost_dir and plot

    Args:
        cost_dir (str, optional): _description_. Defaults to 'pareto_front_solutions_0.2'.
    """
    k_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    f_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])
    weight_combs = []
    for kw in k_weights:
        for fw in f_weights:
            file_str = 'k_' + num2str(kw) + '_f_' + num2str(fw) + '_X.csv'
            file_path = join(solution_dir, file_str)
            if exists(file_path):
                weight_combs.append('\n     '+ str(kw) + ',' + str(fw))

    plot_file_save = join(cost_dir)
    fuel_costs = np.loadtxt(join(plot_file_save, 'fcost.csv'))
    knot_costs = np.loadtxt(join(plot_file_save, 'kcost.csv'))
    path_costs = np.loadtxt(join(plot_file_save, 'pcost.csv'))
    coverage = np.loadtxt(join(plot_file_save, 'cov.csv'))

    print('\nObjective Metric Vector Lengths (should be the same): ')
    print(len(weight_combs))
    print(fuel_costs.shape)
    print(knot_costs.shape)
    print(coverage.shape)
    
    fk_idx, fc_idx, kc_idx = plot_pareto_front(fuel_costs, knot_costs, path_costs, coverage, save_file=join(cost_dir, 'plot_with_front.png'))
    print('\nBest Solution Indices:')
    print(' Fuel-Knot Front Indices: ', fk_idx)
    print(' Fuel-Coverage Front Indices: ', fc_idx)
    print(' Knot-Coverage Front Indices: ', kc_idx, '\n')
    out_str = [s + ',' + str(fc) + ',' + str(kc) + ',' + str(cov) for s, fc, kc, cov in zip(weight_combs, fuel_costs, knot_costs, coverage)]
    out_str = np.array(out_str)
    print('Fuel-Knot Front Weights: ', *out_str[fk_idx])
    print('Fuel-Coverage Front Weights: ', *out_str[fc_idx])
    print('Knot-Coverage Front Weights: ', *out_str[kc_idx], '\n')
    return

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
    coverage = []
    path_times = []
    # thrust_iter = ['0_20', '0_40', '0_60', '0_80', '1_00']
    # thrust_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    thrust_values = [x/10 for x in range(int(start_thrust*10),int(end_thrust*10)+1)]
    thrust_iter = [str(x)[0] + '_' + str(round((x%1)*10))[0] + str(round(((x*10)%1)*10))[0] for x in thrust_values] # '0_20' through '1_00'
    for thrust_str_value in tqdm(thrust_iter):
        X, U, t = load_solution(file_dir=solution_dir, thrust_str=thrust_str_value)
        fuel_cost, knot_cost, path_cost, coverage_ratio, path_time = compute_objective_costs(X, U, t, knotfile)
        fuel_costs.append(fuel_cost)
        knot_costs.append(knot_cost)
        path_costs.append(path_cost)
        coverage.append(coverage_ratio)
        path_times.append(path_time)

    fuel_costs = np.array(fuel_costs)
    knot_costs = np.array(knot_costs)
    path_costs = np.array(path_costs)
    coverage = np.array(coverage)
    path_times = np.array(path_times)

    if not exists(plot_file_save): mkdir(plot_file_save)
    plot_pareto_front(fuel_costs, knot_costs, path_costs, coverage, thrust_values,
                      save_file=join(plot_file_save, 'pareto_front_' + thrust_iter[0] + '_to_' + thrust_iter[-1]))
    return

if __name__ == '__main__':
    # python pareto_front.py 1 1 1 0.2 1.5
    # python pareto_front.py -grid pf_0.2

    if argv[1] == '-grid':
        if len(argv) > 2: save_dir=argv[2]
        else: save_dir='pf_0.2'
        if len(argv) > 3: compute_coverage=bool(argv[3])
        else: compute_coverage=True
        solution_directory = join(getcwd(), 'ocp_paths', save_dir)
        generate_pareto_front_grid(solution_dir=solution_directory, coverage_input=compute_coverage)
    elif argv[1] == '-load':
        if len(argv) > 2: save_dir=argv[2]
        else: save_dir='pf_0.2'
        # solution_directory = join(getcwd(), 'ocp_paths', 'pf_1.0')
        solution_directory = join(getcwd(), 'ocp_paths', save_dir)
        cost_directory = join(getcwd(), 'pareto_front', save_dir)
        pareto_load_plot(cost_dir=cost_directory,
                         solution_dir=solution_directory
                         )
    elif argv[1] == '-c': # -c for cost evaluation of a single solution
        soln_dir=argv[2]
        kw=float(argv[3])
        fw=float(argv[4])
        compute_cost_single_soln(knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv'), 
                                 solution_dir=join(getcwd(), 'ocp_paths', soln_dir),
                                 knot_weight=kw,
                                 fuel_weight=fw,
                                 compute_coverage=True
                                 )
    else:
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