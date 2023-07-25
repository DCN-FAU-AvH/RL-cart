import numpy as np
import params as p


# Cost

def cost(x, x_dot, u):
    """ Approximation of the cost function evaluated in a given trajectory-control pair (x, u), using trapezoidal rule. """
    return p.LAMBDA_P*x[-1]**2 + p.LAMBDA_V*x_dot[-1]**2 + np.trapz(u**2, dx=p.T/u.shape[0])


def J(u, x_0, m, dt=None):
    """ Cost as a function of piece-wise u (after discretization in time) for initial condition x_0, for a given mass m. """
    dt = p.DT if dt is None else dt
    x_N = x_0 + dt**2/m * np.sum(np.arange(p.N, 0, -1)*u)
    v_N = dt/m * np.sum(u)
    return p.LAMBDA_P*x_N**2 + p.LAMBDA_V*v_N**2 + dt*np.sum(u**2)



# State dynamics of the system

def dynamics(x, v, u):
    """ Takes position and action ((x, v), u) and returns a new state x_new. """
    return x + p.DT*v + p.DT**2/(2*p.M_REAL)*u, v + p.DT/p.M_REAL*u


def simulator(x_0, policy):
    """ Runs a simulation of the system between times 0 and T.
    
    :param float x_0: Initial positioni in [0, 1].
    :param function policy: Function that given a position, velocity and time returns an action.
    :return: The couple (x, u) where x is the resulting trajectory and u the control which are 2 np.arrays of size (N+1) and (N) respectively, where N=T/DT.
    """
    x = np.zeros(p.N+1)
    u = np.zeros(p.N)
    x[0] = x_0
    v = 0
    for n in range(p.N):
        u[n] = policy(n*p.DT, x[n], v)
        x[n+1], v = dynamics(x[n], v, u[n])
    return x, u
