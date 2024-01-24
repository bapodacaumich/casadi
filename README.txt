README

FILES:
camera.py: file that defines a camera object to detect points in a camera fov (for coverage)
casadi_opti_station.py: contains most recent optimal control problem formulation for point to point planning problems around many convex hulls
casadi_opti.py: a few other optimal control problem formulations using spheres and convex hulls [not kept up to date, potentially doesn't work]
constraints.py: functions which enforce constraints on the optimal control problem
initial_conditions.py: creates specific arrangements of obstacle formulations
obs.py: contains functions which define or load in obstacles
ode.py: ordinary differential equation function handles
package_path.py: used to convert paths from [x y z xdot ydot zdot] and knot points to [x y z xo yo zo] where (xo yo zo) is a unit vector representing orientation
pareto_front.py: used to generate pareto front plots from a directory of paths
pareto_grid.py: calls casadi_opti_station to produces many ocp's (optimal control problems) which sweep a number of weights used for the cost function
plot_mesh_debug.py: plots mesh and normals for debugging mesh import params
plot_solution.py: plots a mesh and path around mesh to view solutions in 3d
utils.py: various useful functions used in other parts of the code

DIRECTORIES:
./ocp_paths: contains solutions to ocp's
./ccp_paths: contains a list of various knot point files
./model: contains various mesh objects (convex hulls)
./pareto_front: plots of pareto front points for various solution sets
./path_figures: plots of paths