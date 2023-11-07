# based on shooting example
from casadi import *

m_I = 1
g0 = 9.81
Isp = 80

T = 10 # Time Horizon
N = 20 # control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x3 = MX.sym('x3')
x4 = MX.sym('x4')
x = vertcat(x1, x2, x3, x4)
u1 = MX.sym('u1')
u2 = MX.sym('u2')
u = vertcat(u1, u2)

# Model Equations
xdot = vertcat(x3, x4, u1/m_I, u2/m_I)

# Objective Term (no obstacle avoidance, just fuel)
L = sqrt(u1**2 + u2**2)/(g0*Isp)

# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [xdot, L])
X0 = MX.sym('X0', 4)
U = MX.sym('U',2)
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + DT/2 * k1, U)
    k3, k3_q = f(X + DT/2 * k2, U)
    k4, k4_q = f(X + DT * k3, U)
    X = X + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
    Q = Q + DT/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

# Evaluate at a test point
Fk = F(x0=[0,1,0,0], p=[0,0])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# formulate NLP
Xk = MX([0, 1])
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]

    # integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk['xf']
    J = J + Fk['qf']

    # add inequality constraint
    g += 