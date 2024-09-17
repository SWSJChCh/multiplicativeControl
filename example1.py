'''
example1-Fig1.py - 17/09/24

A script used in the publication: Modelling treatment response of heterogeneous
cell populations with optimal multiplicative control and drug-drug interactions

FIGURE 1 PLOTTER

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

import numpy as np
import sys
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

################################################################################

#Non-dimensional model parameter
T = 7

#Linear state coefficient matrix
def A():
    return np.array([[-1, 2], [1, -1]])

#Control cost matrix
def R():

    R = np.eye(2) * 1e-1

    return R

#State cost matrix
def Q():

    Q = np.eye(2)

    return Q

#Non-linear coefficient matrices
def C(alpha, beta):

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
def control(x, lmbd, alpha, beta):
    #Column vector of ones
    ones = np.array([[1], [1]])

    #Compute inner sum term in optimal control formula
    sum_term = sum([np.multiply(C(alpha, beta)[i], ones @ x.T @ E()[i]) for \
                          i in range(2)])

    #Compute optimal control and clip between (0, 1)
    u_star = -np.linalg.inv(R()) @ sum_term.T @ lmbd

    u_star = np.minimum(u_star, np.array([[1], [1]]))
    u_star = np.maximum(u_star, np.array([[0], [0]]))

    #Optimal control
    return u_star

################################################################################

#System dynamics
def costate(t, master_flat, alpha, beta):

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x1 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate control from state and adjoint
    u = control(x, lmbd, alpha, beta)

    #Calculate lmbd derivative
    ones = np.array([[1], [1]])
    e1 = np.array([[1], [0]])
    e2 = np.array([[0], [1]])
    eArr = [e1, e2]
    sum_term = sum([eArr[i] @ (C(alpha, beta)[i] @ u).T @ \
                    lmbd for i in range(2)])

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
    u = control(x, lmbd, alpha, beta)

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C(alpha, beta)[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u

    #Flattened state dynamics matrix
    return(((A() @ x + L).flatten()))


#Derivative of both state vector and adjoint (vectorised)
def total_derivative_vec(t, y, alpha, beta):

    #Return time derivatives of x and lmbd
    return np.array([np.array(list(state(t, y[:,i])) + \
    list(costate(t, y[:,i], alpha, beta))) for i in range(y.shape[1])]).T

#Evaluation of the boundary conditions
def bc(ya,yb):

    return np.array([ya[0] - x0[0][0], ya[1]-x0[1][0], yb[2], yb[3]])


################################################################################
#                         MEAN CONTROL FUNCTION                                #
################################################################################

#System dynamics
def costate_mean(t, master_flat, alpha, beta, u1, u2):

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x1 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate control from state and adjoint
    u = np.array([[u1], [u2]])

    #Calculate lmbd derivative
    ones = np.array([[1], [1]])
    e1 = np.array([[1], [0]])
    e2 = np.array([[0], [1]])
    eArr = [e1, e2]
    sum_term = sum([eArr[i] @ (C(alpha, beta)[i] @ u).T @ \
                    lmbd for i in range(2)])

    M = -Q() @ x - (A()).T @ lmbd - sum_term

    #Return flattened state vector time derivative
    return M.flatten()

#System dynamics for the state vector x(t)
def state_mean(t, master_flat, u1, u2):

    #Column vector of ones
    ones = np.array([[1], [1]])

    #Reshape x_flat back into 2x1 matrix
    x = master_flat[0:2].reshape((2, 1))

    #Reshape lmbd_flat back into a 2x1 matrix
    lmbd = master_flat[2:].reshape((2, 1))

    #Calculate optimal control
    u = np.array([[u1], [u2]])

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C(alpha, beta)[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u

    #Flattened state dynamics matrix
    return(((A() @ x + L).flatten()))


#Derivative of both state vector and adjoint (vectorised)
def total_derivative_vec_mean(t, y, alpha, beta, u1, u2):

    #Return time derivatives of x and lmbd
    return np.array([np.array(list(state_mean(t, y[:,i], u1, u2)) + \
    list(costate_mean(t, y[:,i], alpha, beta, u1, u2))) for i in \
         range(y.shape[1])]).T

#Evaluation of the boundary conditions
def bc(ya,yb):

    return np.array([ya[0] - x0[0][0], ya[1]-x0[1][0], yb[2], yb[3]])


################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, lmbd0, t_span, alpha, beta):

    #Solve BVP for state and adjoint
    sol_master = solve_bvp(lambda t, y: total_derivative_vec(t, y, alpha, \
                           beta), bc, x = tmsh, y = np.zeros((4,\
                           tmsh.shape[0])), verbose=2, tol=1e-4, max_nodes = 1e4)

    #Extract state and adjoint dynamics
    sol_x = sol_master.y[0:2, :]
    sol_lmbd = sol_master.y[2:, :]
    t = sol_master.x

    return t, sol_x, sol_lmbd

#Solve the system of ODEs
def solve_optimal_control_mean(x0, lmbd0, t_span, alpha, beta, u1, u2):

    #Solve BVP for state and adjoint
    sol_master = solve_bvp(lambda t, y: total_derivative_vec_mean(t, y, \
                            alpha, beta, u1, u2), bc, x = tmsh, y = \
                            np.zeros((4,tmsh.shape[0])), verbose=2, tol=1e-4, \
                            max_nodes = 2.5e4)

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

alpha = float(sys.argv[1])
beta = float(sys.argv[2])

################################################################################

t1, sol_x, sol_lmbd = solve_optimal_control(x0, lmbd0, t_span, alpha, beta)

#Calculate control from state and adjoint
sol_u = [control(sol_x[:, i].reshape((2, 1)), \
sol_lmbd[:, i].reshape((2, 1)), alpha, beta) for i in range(len(t1))]

#Reshape control vector for plotting
sol_u = np.array(sol_u).T

#Cell count at end in optimal regime
x1_end = sol_x[0, :][-1]
x2_end = sol_x[1, :][-1]

#Total cost calculation
total_cost = 0.1 * np.trapz(np.array(sol_u[0][0])**2 + \
                   np.array(sol_u)[0][1]**2, t1)

total_u1 = np.trapz(np.array(sol_u[0][0]), t1)
total_u2 = np.trapz(np.array(sol_u[0][1]), t1)

total_cost_cis = 0.1 * np.trapz(np.array(sol_u[0][0])**2, t1)
total_cost_pac = 0.1 * np.trapz(np.array(sol_u[0][1])**2, t1)

#Total cost per time
total_cost_per_time = total_cost / T

#Total drugs per time
total_u1_per_time = total_u1 / T
total_u2_per_time = total_u2 / T

#Drug ratios
drug_ratio = total_u1 / total_u2

#List of drugs per time
drug_list = [total_u1_per_time, total_u2_per_time]

if drug_list[0] > drug_list[1]:
    total_u1_per_time = min([total_u1_per_time, 1])
    total_u2_per_time = total_u1_per_time / drug_ratio
else:
    total_u2_per_time = min([total_u2_per_time, 1])
    total_u1_per_time = total_u2_per_time * drug_ratio

#Solve optimal control in mean dose regime
t2, sol_x_mean, sol_lmbd_mean = solve_optimal_control_mean(x0, \
lmbd0, t_span, alpha, beta, total_u1_per_time, total_u2_per_time)

x1_mean_end = sol_x_mean[0, :][-1]
x2_mean_end = sol_x_mean[1, :][-1]

ratio = (x1_mean_end + x2_mean_end) / (x1_end + x2_end)

#Create data file for output
file = open('a={}-b={}.txt'.format(float(sys.argv[1]), float(sys.argv[2])), 'a')

#Write data to .txt. file in columns
file.write(str((x1_mean_end + x2_mean_end)) + " " + \
           str((x1_end + x2_end)) + " " + \
           str(ratio) + " " + \
           str(total_cost) + " " + \
           str(total_cost_cis) + " " + \
           str(total_cost_pac))

#Close file after writing
file.close()
