'''
ricattiSolver.py - 09/09/24

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
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

################################################################################

#Non-dimensional model parameter
alpha = 2
T = 20

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

#Compute R1 (control-dependent term)
def R1(x, S):
    ones = np.array([[1], [1]])
    sum_term = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for i in range(2)])
    return -sum_term @ np.linalg.inv(R()) @ sum_term.T @ S

#Compute R2 (control-dependent term)
def R2(x, S):
    ones = np.array([[1], [1]])
    e1 = np.array([[1], [0]])
    e2 = np.array([[0], [1]])
    eArr = [e1, e2]
    inner_sum_term = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)])
    sum_term = sum([eArr[i] @ (x.T @ S @ inner_sum_term) @ (np.linalg.inv(R())).T \
                          @ C()[i].T for i in range(2)])
    return -sum_term @ S

#Compute control
def control(x, S):
    #Column vector of ones
    ones = np.array([[1], [1]])

    #Compute inner sum term in optimal control formula
    sum_term = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)])

    #Compute optimal control and clip between (0, 1)
    u_star = np.linalg.inv(R()) @ sum_term.T @ S @ x

    u_star = np.minimum(u_star, np.array([[1], [1]]))
    u_star = np.maximum(u_star, np.array([[0], [0]]))

    #Optimal control
    return u_star

################################################################################

#System dynamics for S(t)
def riccati(t, master_flat):

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape S_flat back into a 2x2 matrix
    S = master_flat[2:].reshape((len(x), len(x)))

    #Compute the matrix M in the Riccati equation
    R1_val = R1(x, S)
    R2_val = R2(x, S)

    #Riccati equation: M = -SA - R1 - Q - A^T S + R2
    M = -S @ A() + R1_val - Q() - A().T @ S + R2_val

    return M.flatten()

#System dynamics for the state vector x(t)
def state(t,  master_flat):

    #Column vector of ones
    ones = np.array([[1], [1]])

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape S_flat back into a 2x2 matrix
    S = master_flat[2:].reshape((len(x), len(x)))

    #Calculate optimal control from Ricatti equation
    u_star = control(x, S)

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u_star

    #Flattened state dynamics matrix
    return(((A() @ x + L).flatten()))

#Derivative of both state vector and S(x)
def total_derivative(t, master_flat):

    #Return time derivatives of S and x
    return list(state(t, master_flat)) + list(riccati(t, master_flat))

################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, S0, t_span):
    #Flatten initial conditions
    S0_flat = list(S0.flatten())
    x0_flat = list(x0.flatten())

    master0_flat = x0_flat + S0_flat

    #Solve for S(t) using solve_ivp
    sol_master = solve_ivp(total_derivative, t_span, master0_flat, \
                 method='LSODA')

    sol_x = sol_master.y[0:2, :]
    sol_S = sol_master.y[2:, :]
    t = sol_master.t

    return t, sol_S, sol_x

#Initial conditions
x0 = np.array([[1], [1]])
S0 = np.array([[0, 0], [0, 0]])
t_span = (0, T)

t, sol_S, sol_x = solve_optimal_control(x0, S0, t_span)

sol_u = [control(sol_x[:, i].reshape((2, 1)), \
        sol_S[:, i].reshape((2, 2))) for i in range(len(t))]

sol_u = np.array(sol_u).T

#Visualization function
def plot_results(t, sol_S, sol_x):

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


#Example usage
x0 = np.array([[1], [1]])  #Initial condition for state vector
S0 = np.array([[0, 0], [0, 0]])  #Initial condition for S
t_span = (0, T)  #Time interval

plot_results(t, sol_S, sol_x)
