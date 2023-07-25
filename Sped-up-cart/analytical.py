import numpy as np

from params import *



def ground_truth(x0, t, lamda=LAMBDA):
    """ Analytical expressions of the trajectory and the control for the continuous control problem. """
    sqrtl = np.sqrt(lamda)
    cosht = np.cosh(sqrtl*T)
    return (x0/cosht)*np.cosh(sqrtl*(T-t)), -(sqrtl*x0/cosht*np.sinh(sqrtl*(T-t)))


def ground_truth_sample(x0, res=100000, lamda=LAMBDA):
    """ Sample of the trajectory and control solution to the continuous control problem. """
    t = np.linspace(0, T, res)
    return ground_truth(x0, t, lamda=lamda)
