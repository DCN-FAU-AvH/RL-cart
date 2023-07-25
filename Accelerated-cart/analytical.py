import numpy as np
import params as p


def ground_truth(x_0, t):
    """ Analytical expressions of the trajectory, its derivative and the control for the continuous control problem. """
    C1 = 1 + (p.LAMBDA_P*p.T**3)/(3*p.M_REAL**2)
    C2 = (p.LAMBDA_P*p.T**2)/(2*p.M_REAL**2)
    C3 = (p.LAMBDA_V*p.T**2)/(2*p.M_REAL**2)
    C4 = 1 + (p.LAMBDA_V*p.T)/(p.M_REAL**2)
    p1 = (2*C4)/(C1*C4-C2*C3)*p.LAMBDA_P*x_0
    p2 = -(2*C3)/(C1*C4-C2*C3)*p.LAMBDA_P*x_0
    x = p1/(12*p.M_REAL**2)*(p.T**3 - (p.T-t)**3) - p2/(4*p.M_REAL**2)*t**2 - (p.T**2*p1)/(4*p.M_REAL**2)*t + x_0
    v = p1/(4*p.M_REAL**2)*(p.T-t)**2 - p2/(2*p.M_REAL**2)*t - (p.T**2*p1)/(4*p.M_REAL**2)
    u = -1/(2*p.M_REAL)*(p1*(p.T-t) + p2)
    return x, v, u


def ground_truth_sample(x_0, res=10000):
    """ Samples the analytical trajectory, its derivative and the analytical control for the continuous control problem. """
    t = np.linspace(0, p.T, res)
    return ground_truth(x_0, t)
