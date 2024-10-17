'''
Figure5+6.py - 07/10/2024

Optimal combination therapy for heterogeneous cell populations
with drug synergies

Written by Samuel Johnson and Simon Martina-Perez

dnI_dt = (r - 2) * nI + nN + nS \
         - (delta * u_RA + 2 * r * u_chemo \
         + delta * (1 - u_trk) * u_NGF) * nI \
         + delta * (1 - u_trk) * nN

dnN_dt = -2 * nN + nI + nS \
         + delta * (u_RA + (1 - u_trk) * \
         u_NGF) * nI \
         - (delta_apop * (1 - u_NGF) * (1 - u_trk) \
         + 2 * delta * (1 - u_trk)) * nN

dnS_dt = -2 * nS + nI + nN \
         + delta * (1 - u_trk) * nN


nI = sympathoblasts
nN = adrenergic cells
nS = mesenchymal cells
u_RA = retinoic acid
u_chemo = chemotherapeutic agent
u_trk = track inhibitor 
u_NGF = nerve growth factor

state vector (nI, nN, nS)
drug vector (u_RA, u_chemo, u_trk, u_NGF)

In solver, flattened array formatted as [x1, x2, x3, lmbd1, lmbd2, lmbd3]

'''

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

################################################################################

#Non-dimensional model parameters
T = 7
r = float(sys.argv[1])
delta = float(sys.argv[2])
deltaApop = float(sys.argv[3])

#Linear state coefficient matrix
def A():
    return np.array([[r - 2, 1 + delta, 1],\
                    [1, - 2 - 2 * delta - deltaApop, 1], \
                    [1, 1 + delta, -2]])

#Control cost matrix
def R():

    R = 0.1 * np.eye(4)

    return R

#State cost matrix
def Q():

    Q = np.eye(3)

    return Q

#Non-linear coefficient matrices (I then N)
def C():

    #Two 3x4 matrices
    return np.array([np.array([[-delta, -2 * r, 0, -delta],\
                    [delta, 0, 0, delta], \
                    [0, 0, 0, 0]]), \
                    np.array([[0, 0, -delta, 0],\
                    [0, 0, deltaApop + 2 * delta, deltaApop], \
                    [0, 0, -delta, 0]])])

#Multiplication matrices (II then IN then NN)
def D():

    #Two 4x4 matrices
    return np.array([np.array([[0, 0, 0, 0],\
                    [0, 0, 0, 0], \
                    [0, 0, 0, delta], \
                    [0, 0, 0, 0]]), \
                    np.array([[0, 0, 0, 0],\
                    [0, 0, 0, 0], \
                    [0, 0, 0, -delta], \
                    [0, 0, 0, 0]]), \
                    np.array([[0, 0, 0, 0],\
                    [0, 0, 0, 0], \
                    [0, 0, 0, -deltaApop], \
                    [0, 0, 0, 0]])])


#Epsilon matrices
def E():
    e_1 = np.array([[1], [0], [0]])
    e_2 = np.array([[0], [1], [0]])
    e_3 = np.array([[0], [0], [1]])

    #Each E[i] is 3x4
    return np.array([e_1 @ np.ones((1, 4)), e_2 @ np.ones((1, 4)), \
    e_3 @ np.ones((1, 4))])

#Compute control
def control(x, lmbd):
    #Basis vectors
    e_1 = np.array([[1], [0], [0]])
    e_2 = np.array([[0], [1], [0]])
    e_3 = np.array([[0], [0], [1]])

    #Column vector of ones
    ones = np.array([[1], [1], [1]])

    #Compute inner sum term in optimal control formula
    term1 = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)])

    t2 = R() + e_1.T @ x * lmbd.T @ e_1 * (D()[0]) + \
                              e_1.T @ x * lmbd.T @ e_2 * (D()[1]) + \
                              e_2.T @ x * lmbd.T @ e_2 * (D()[2])

    if np.linalg.cond(t2) < 1 / sys.float_info.epsilon :
        
        #Compute term 2 in optimal control formula
        term2 = np.linalg.inv(R() + e_1.T @ x * lmbd.T @ e_1 * (D()[0]) + \
                                  e_1.T @ x * lmbd.T @ e_2 * (D()[1]) + \
                                  e_2.T @ x * lmbd.T @ e_2 * (D()[2]))

        #Compute optimal control and clip between (0, 1)
        u_star = -term2 @ term1.T @ lmbd
        u_star = np.minimum(u_star, np.array([[1], [1], [1], [1]]))
        u_star = np.maximum(u_star, np.array([[0], [0], [0], [0]]))

    else:
        dummyControl = np.array([[1], [1], [1], [1]])
        dudH = R() @ dummyControl + sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                              i in range(2)]).T @ lmbd + (e_1.T @ x * lmbd.T @ e_1 * (D()[0]) + \
                                                        e_1.T @ x * lmbd.T @ e_2 * (D()[1]) + \
                                                        e_2.T @ x * lmbd.T @ e_2 * (D()[2])) @ dummyControl    

        u_star = (dudH > 0)*1.

    #Optimal control
    return u_star

################################################################################

#System dynamics for S(t)
def costate(t, master_flat):

    #Reshape x_flat back into 3x1 matrix
    x = master_flat[0:3].reshape((3, 1))

    #Reshape lmbd_flat back into a 4x1 matrix
    lmbd = master_flat[3:].reshape((3, 1))

    u = control(x, lmbd)

    #Calculate lambda derivative
    ones = np.array([[1], [1], [1], [1]])
    e_1 = np.array([[1], [0], [0]])
    e_2 = np.array([[0], [1], [0]])
    e_3 = np.array([[0], [0], [1]])
    eArr = [e_1, e_2, e_3]

    sum_term = sum([eArr[i] @ (C()[i] @ u).T @ lmbd for i in range(2)])

    sum_term += e_1 * (e_1 @ ones.T @ np.multiply(D()[0], u @ ones.T) @ u).T @ lmbd 

    sum_term += e_1 * (e_2 @ ones.T @ np.multiply(D()[1], u @ ones.T) @ u).T @ lmbd 

    sum_term += e_2 * (e_2 @ ones.T @ np.multiply(D()[2], u @ ones.T) @ u).T @ lmbd 

    M = -Q() @ x - (A()).T @ lmbd - sum_term

    return M.flatten()

#System dynamics for the state vector x(t)
def state(t,  master_flat):

    ones = np.array([[1], [1], [1], [1]])
    e_1 = np.array([[1], [0], [0]])
    e_2 = np.array([[0], [1], [0]])
    e_3 = np.array([[0], [0], [1]])

    #Column vector of ones
    ones = np.array([[1], [1], [1]])

    #Reshape x_flat back into 3x1 matrix
    x = master_flat[0:3].reshape((3, 1))

    #Reshape lmbd_flat back into a 4x1 matrix
    lmbd = master_flat[3:].reshape((3, 1))

    #Calculate optimal control from Ricatti equation
    u_star = control(x, lmbd)

    #Compute the nonlinear state dynamics
    L = sum([np.multiply(C()[i], ones @ x.T @ E()[i]) for \
                          i in range(2)]) @ u_star

    #Column vector of ones
    ones = np.array([[1], [1], [1], [1]])

    #Compute multiplicative drug dynamics
    J = (e_1 @ x.T @ E()[0]) @ (np.multiply(D()[0], u_star @ ones.T) @ u_star)

    J += (e_2 @ x.T @ E()[0]) @ (np.multiply(D()[1], u_star @ ones.T) @ u_star)

    J += (e_2 @ x.T @ E()[1]) @ (np.multiply(D()[2], u_star @ ones.T) @ u_star)

    #Flattened state dynamics matrix
    return(((A() @ x + L + J).flatten()))

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

    return np.array([ya[0] - x0[0][0],
                    ya[1]-x0[1][0],
                    ya[2]-x0[2][0],
                    yb[3],
                    yb[4],
                    yb[5]])

################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, lmbd0, t_span):

    sol_master = solve_bvp(total_derivative_vec, bc, x = tmsh,
                            y = np.ones((6,tmsh.shape[0])), verbose=2,
                            tol=1e-2, max_nodes = 1e4)

    sol_x = sol_master.y[0:3, :]
    sol_lmbd = sol_master.y[3:, :]
    t = sol_master.x

    return t, sol_x, sol_lmbd

#Initial conditions
x0 = np.array([[1], [1], [1]])
lmbd0 = np.array([[0], [0], [0], [0]])
t_span = (0, T)
tmsh = np.linspace(0, T, 1000)

t, sol_x, sol_lmbd = solve_optimal_control(x0, lmbd0, t_span)

################################################################################

sol_u = [control(sol_x[:, i].reshape((3, 1)), \
        sol_lmbd[:, i].reshape((3, 1))) for i in range(len(t))]

################################################################################

sol_u = np.array(sol_u).T

################################################################################

#Total cost calculation
total_cost = np.trapz(R()[0][0] * np.array(sol_u[0][0])**2 + \
                   R()[1][1] * np.array(sol_u)[0][1]**2 + \
                   R()[2][2] * np.array(sol_u)[0][2]**2 + \
                   R()[3][3] * np.array(sol_u)[0][3]**2, t)

total_RA = np.trapz(np.array(sol_u[0][0]), t)
total_chem = np.trapz(np.array(sol_u[0][1]), t)
total_trk = np.trapz(np.array(sol_u[0][2]), t)
total_NGF = np.trapz(np.array(sol_u[0][3]), t)

total_cost_RA = np.trapz(R()[0][0] * np.array(sol_u[0][0])**2, t)
total_cost_chem = np.trapz(R()[1][1] * np.array(sol_u[0][1])**2, t)
total_cost_trk = np.trapz(R()[2][2] * np.array(sol_u[0][2])**2, t)
total_cost_NGF = np.trapz(R()[3][3] * np.array(sol_u[0][3])**2, t)

#Create data file for output
file = open('lmbd={}-delta={}-deltaAPOP={}.txt'.format(float(sys.argv[1]), \
             float(sys.argv[2]), float(sys.argv[3])), 'a')

#Write data to .txt. file in columns
file.write(str(total_RA) + " " + \
           str(total_chem) + " " + \
           str(total_trk) + " " + \
           str(total_NGF) + " " + \
           str(total_cost_RA) + " " + \
           str(total_cost_chem) + " " + \
           str(total_cost_trk) + " " + \
           str(total_cost_NGF))

#Close file after writing
file.close()
