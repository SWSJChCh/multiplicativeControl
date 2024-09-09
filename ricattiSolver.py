import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

################################################################################

#Model parameters
lambdaA = 1e-5
lambdaB = 1e-5
kAB = 1e-3
kBA = 1e-1
T = 10

################################################################################

def A():
    return np.array([[lambdaA - kAB, kBA], [kAB, lambdaB - kBA]])

#Control cost matrix
def R():
    return np.eye(2)

#State cost matrix
def Q():
    return np.eye(2)

def C():
    #Two 2x2 matrices
    return np.array([[[0, 0], [0, 0]], \
                     [[0, -kBA], [-lambdaB, kBA]]])

def E():
    e_1 = np.array([[1], [0]])
    e_2 = np.array([[0], [1]])
    #Each E[i] is 2x2
    return np.array([e_1 @ np.ones((1, 2)), e_2 @ np.ones((1, 2))])

#Compute R1 (control-dependent term)
def R1(x, S):
    sum_term = sum([C()[i] @ np.outer(x, np.ones(len(x))) @ E()[i] for i in range(len(C()))])
    return -sum_term @ np.linalg.inv(R()) @ sum_term.T @ S

#Compute R2 (control-dependent term)
def R2(x, S):
    sum_term = sum([np.outer(np.ones(len(x)), C()[i].T @ S @ x) for i in range(len(C()))])
    return -sum_term @ np.linalg.inv(R().T) @ sum_term.T @ S

################################################################################

#System dynamics for S(t)
def riccati_ode(t, S_flat, x):
    #Reshape S_flat back into a 2x2 matrix
    S = S_flat.reshape((len(x), len(x)))

    #Compute the matrix M in the Riccati equation
    R1_val = R1(x, S)
    R2_val = R2(x, S)

    #Riccati equation: M = -SA - R1 - Q - A^T S + R2
    M = -S @ A() + R1_val - Q() - A().T @ S + R2_val
    return M.flatten()  #Flatten the matrix to use in ODE solvers

#Feedback control
def control(t, x, S):
    sum_term = sum([C()[i] @ np.outer(np.ones(len(x)), x) @ E()[i] for i in range(len(C()))])
    u_star = -np.linalg.inv(R()) @ sum_term.T @ S @ x
    u_star = np.minimum(u_star, np.array([1, 1]))
    u_star = np.maximum(u_star, np.array([0, 0]))
    return -u_star

#System dynamics for the state vector x(t)
def state_dynamics(t, x, S_func):
    #Get S(t) from the interpolated solution
    S = S_func(t).reshape((len(x), len(x)))
    u_star = control(t, x, S)

    #Compute the nonlinear state dynamics
    L = np.zeros_like(x)
    for i in range(len(C())):
        L += C()[i] @ u_star

    return A() @ x + L

################################################################################

#Solve the system of ODEs
def solve_optimal_control(x0, S0, t_span):
    S0_flat = S0.flatten()

    #Solve for S(t) using solve_ivp
    sol_S = solve_ivp(riccati_ode, t_span, S0_flat, args=(x0,), method='RK45')

    #Use interp1d to interpolate each element of S separately
    S_interp_funcs = [interp1d(sol_S.t, sol_S.y[i], kind='linear', \
                    fill_value="extrapolate") for i in range(sol_S.y.shape[0])]

    #Define a function to get the full S matrix at time t
    def S_func(t):
        S_flat = np.array([interp_func(t) for interp_func in S_interp_funcs])
        return S_flat

    #Use the solved S(t) to compute the optimal state trajectory
    sol_x = solve_ivp(state_dynamics, t_span, x0, args=(S_func,), method='RK45')

    return sol_S, sol_x

#Example usage
x0 = np.array([1, 1])  #Initial condition for x
S0 = np.array([[0, 0], [0, 0]])  #Initial condition for S (symmetric matrix)
t_span = (0, T)  #Time interval

sol_S, sol_x = solve_optimal_control(x0, S0, t_span)

#Compute the control input
def compute_control(t, x, S_func):
    S = S_func(t).reshape((2, 2))  #Reshape S from flattened form
    sum_term = sum([C()[i] @ np.outer(np.ones(len(x)), x) @ E()[i] for i in range(len(C()))])
    u_star = -np.linalg.inv(R()) @ sum_term.T @ S @ x
    u_star = np.minimum(u_star, np.array([1, 1]))
    u_star = np.maximum(u_star, np.array([0, 0]))
    return u_star

#Visualization function
def plot_results(sol_S, sol_x, T):
    #Time points
    t = sol_x.t

    #State trajectory
    x = sol_x.y.T

    #Interpolate S(t) from sol_S
    S_interp_funcs = [interp1d(sol_S.t, sol_S.y[i], kind='linear', fill_value="extrapolate") for i in range(sol_S.y.shape[0])]
    def S_func(t):
        S_flat = np.array([interp_func(t) for interp_func in S_interp_funcs])
        return S_flat.reshape((2, 2))

    S = np.array([S_func(ti) for ti in t])

    #Compute control input over time
    u = np.array([compute_control(ti, x_i, S_func) for ti, x_i in zip(t, x)])

    # Plot state trajectory
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(t, x[:, 0], label='$x_1(t)$')
    plt.plot(t, x[:, 1], label='$x_2(t)$')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Trajectory')
    plt.legend()

    #Plot S(t) over time
    plt.subplot(1, 3, 2)
    plt.plot(t, [S[i, 0, 0] for i in range(len(t))], label='$S_{{11}}(t)$')
    plt.plot(t, [S[i, 0, 1] for i in range(len(t))], label='$S_{{12}}(t)$')
    plt.plot(t, [S[i, 1, 0] for i in range(len(t))], label='$S_{{21}}(t)$')
    plt.plot(t, [S[i, 1, 1] for i in range(len(t))], label='$S_{{22}}(t)$')
    plt.xlabel('Time')
    plt.ylabel('Matrix Elements')
    plt.title('Matrix $S(t)$')
    plt.legend()

    #Plot control input over time
    plt.subplot(1, 3, 3)
    plt.plot(t, u[:, 0], label='$u_1(t)$')
    plt.plot(t, u[:, 1], label='$u_2(t)$')
    plt.xlabel('Time')
    plt.ylabel('Control Input')
    plt.title('Control Input')
    plt.legend()

    plt.tight_layout()
    plt.show()

#Example usage
x0 = np.array([1, 1])  #Initial condition for state vector
S0 = np.array([[0, 0], [0, 0]])  #Initial condition for S
t_span = (0, T)  #Time interval

sol_S, sol_x = solve_optimal_control(x0, S0, t_span)
plot_results(sol_S, sol_x, T)
