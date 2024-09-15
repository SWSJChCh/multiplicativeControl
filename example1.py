'''
example1.py - 15/09/24

A script used in the publication: Modelling treatment response of heterogeneous
cell populations with optimal multiplicative control and drug-drug interactions

Example 1

Written by Samuel Johnson and Simon Martina-Perez

dNa / dt = 2(1 - Up) * Nb - Na * (1 - Uc) - alpha * Uc * Na

dNb / dt = -(1 - Up) * Nb + Na * (1 - Uc) - beta * Up * Nb

Na = non-proliferative cells
Nb = proliferative cells
Uc = cisplatin concentration
Up = paclitaxel concentration

state vector (Na, Nb)
drug vector (Uc, Up)

In solver, flattened array formatted as [x1, x2, lmbd1, lmbd2]

'''

alpha = 1
beta = 1

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

################################################################################

#Non-dimensional model parameter
T = 2.5

#Linear state coefficient matrix
def A():
    return np.array([[-1, 2], [1, -1]])

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
    return np.array([[[1 - alpha, 0], [-1, 0]], \
                     [[0, -2], [0, 1 - beta]]])

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

#System dynamics
def costate(t, master_flat):

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x1 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate control from state and adjoint
    u = control(x, lmbd)

    #Calculate lmbd derivative
    ones = np.array([[1], [1]])
    e1 = np.array([[1], [0]])
    e2 = np.array([[0], [1]])
    eArr = [e1, e2]
    sum_term = sum([eArr[i] @ (C()[i] @ u).T @ lmbd for i in range(2)])

    M = -Q() @ x - (A()).T @ lmbd - sum_term

    #Return flattened state vector time derivative
    return M.flatten()

#System dynamics for the state vector x(t)
def state(t,  master_flat):

    #Column vector of ones
    ones = np.array([[1], [1]])

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x1 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate optimal control
    u_star = control(x, lmbd)

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u_star

    #Flattened state dynamics matrix
    return(((A() @ x + L).flatten()))


#Derivative of both state vector and adjoint (vectorised)
def total_derivative_vec(t, y):

    #Return time derivatives of x and lmbd
    return np.array([np.array(list(state(t, y[:,i])) + list(costate(t, y[:,i])))\
                        for i in range(y.shape[1])]).T

#Evaluation of the boundary conditions
def bc(ya,yb):

    return np.array([ya[0] - x0[0][0], ya[1]-x0[1][0], yb[2], yb[3]])


################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, lmbd0, t_span):

    #Solve BVP for state and adjoint
    sol_master = solve_bvp(total_derivative_vec, bc, x = tmsh,
                            y = np.zeros((4,tmsh.shape[0])), verbose=2,
                            tol=1e-4, max_nodes = 1e4)

    #Extract state and adjoint dynamics
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

#Calculate control from state and adjoint
sol_u = [control(sol_x[:, i].reshape((2, 1)), \
        sol_lmbd[:, i].reshape((2, 1))) for i in range(len(t))]

#Reshape control vector for plotting
sol_u = np.array(sol_u).T

################################################################################

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
