'''
ricattiSolver.py - 13/09/24

A script used in the publication: Modelling treatment response of heterogeneous
cell populations with optimal multiplicative control and drug-drug interactions

Example 1

Written by Samuel Johnson and Simon Martina-Perez

dNa / dt = 2Nb(1 - Up) - alpha Na(1 - Uc)

dNb / dt = -Nb(1 - Up) + alpha Na(1 - Uc)

Na = non-proliferative cells
Nb = proliferative cells
Up = paclitaxel concentration
Uc = cisplatin concentration

state vector (Na, Nb)
drug vector (Uc, Up)

In solver, flattened array formatted as [x...S...]

'''

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

################################################################################

#Non-dimensional model parameter
alpha = 0.5
T = 2.5

#Linear state coefficient matrix
def A():
    return np.array([[-alpha, 2], [alpha, -1]])

#Control cost matrix
def R():

    R = np.eye(2)

    return R

#State cost matrix
def Q():

    Q = np.eye(2)

    return Q

#Non-linear coefficient matrices
def C():

    #Two 2x2 matrices
    return np.array([[[alpha, 0], [-alpha, 0]], \
                     [[0, -2], [0, 1]]])

#Epsilon matrices
def E():
    e_1 = np.array([[1], [0]])
    e_2 = np.array([[0], [1]])

    #Each E[i] is 2x2
    return np.array([e_1 @ np.ones((1, 2)), e_2 @ np.ones((1, 2))])

#Compute control
def control(x, lmbd):
    #Column vector of ones
    ones = np.array([[1], [1]])

    #Compute inner sum term in optimal control formula
    sum_term = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)])

    #Compute optimal control and clip between (0, 1)
    u_star = -np.linalg.inv(R()) @ sum_term.T @ lmbd

    u_star = np.minimum(u_star, np.array([[1], [1]]))
    u_star = np.maximum(u_star, np.array([[0], [0]]))

    #Optimal control
    return u_star

################################################################################

#System dynamics for S(t)
def costate(t, master_flat):

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape S_flat back into a 2x2 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    u = control(x, lmbd)

    #Calculate lambda derivative
    ones = np.array([[1], [1]])
    e1 = np.array([[1], [0]])
    e2 = np.array([[0], [1]])
    eArr = [e1, e2]
    sum_term = sum([eArr[i] @ (C()[i] @ u).T @ lmbd for i in range(2)])

    # print((A()).T @ lmbd)
    # print(sum_term)

    M = -Q() @ x - (A()).T @ lmbd - sum_term

    return M.flatten()

#System dynamics for the state vector x(t)
def state(t,  master_flat):

    #Column vector of ones
    ones = np.array([[1], [1]])

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x2 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate optimal control from Ricatti equation
    u_star = control(x, lmbd)

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u_star

    #Flattened state dynamics matrix
    return(((A() @ x + L).flatten()))

#Derivative of both state vector and S(x)
def total_derivative(t, master_flat):

    #Return time derivatives of S and x
    return list(state(t, master_flat)) + list(costate(t, master_flat))

#Derivative of both state vector and S(x), vectorised

def total_derivative_vec(t, y):

    #Return time derivatives of S and x
    return np.array([np.array(list(state(t, y[:,i])) + list(costate(t, y[:,i])))\
                        for i in range(y.shape[1])]).T


#Evaluation of the boundary conditions
def bc(ya,yb):

    return np.array([ya[0] - x0[0][0], ya[1]-x0[1][0], yb[2], yb[3]])


################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, lmbd0, t_span):
    #Flatten initial conditions
    #lmbd0_flat = list(lmbd0.flatten())
    #x0_flat = list(x0.flatten())

    #master0_flat = x0_flat + lmbd0_flat

    #Solve for S(t) using solve_ivp
    # sol_master = solve_ivp(total_derivative, t_span, master0_flat, \
    #              method='LSODA')

    sol_master = solve_bvp(total_derivative_vec, bc, x = tmsh,
                            y = np.zeros((4,tmsh.shape[0])), verbose=2,
                            tol=1e-4, max_nodes = 1e4)

    sol_x = sol_master.y[0:2, :]
    sol_lmbd = sol_master.y[2:, :]
    t = sol_master.x

    return t, sol_x, sol_lmbd

#Initial conditions
x0 = np.array([[1], [1]])
lmbd0 = np.array([[0], [0]])
t_span = (0, T)
tmsh = np.linspace(0, T, 1000)

t, sol_x, sol_lmbd = solve_optimal_control(x0, lmbd0, t_span)



################################################################################

sol_u = [control(sol_x[:, i].reshape((2, 1)), \
        sol_lmbd[:, i].reshape((2, 1))) for i in range(len(t))]

################################################################################

sol_u = np.array(sol_u).T

#Visualization function
def plot_results(t, sol_x, sol_lmbd):

    #Plot state trajectory
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, sol_x[0, :], label='$x_1(t)$')
    plt.plot(t, sol_x[1, :], label='$x_2(t)$')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Trajectory')

    #Plot control input over time
    plt.subplot(1, 2, 2)
    plt.plot(t, sol_u[0][0], label='$u_1(t)$')
    plt.plot(t, sol_u[0][1], label='$u_2(t)$')
    plt.xlabel('Time')
    plt.ylabel('Control Input')
    plt.title('Control Input')
    plt.legend()

    plt.show()

plot_results(t, sol_x, sol_lmbd)
