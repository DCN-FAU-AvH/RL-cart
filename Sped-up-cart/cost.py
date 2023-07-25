import numpy as np
from scipy import integrate

from params import *


def cost(x, u):
    """ Compute an approximation of the cost function evaluated in a given trajectory-control pair (x, u), using Simpson's method. """
    # return lamda*np.trapz(x**2, dx=T/x.shape[0]) + np.trapz(u**2, dx=T/u.shape[0])
    return LAMBDA*integrate.simpson(x**2, dx=T/x.shape[0]) + integrate.simpson(u**2, dx=T/u.shape[0])

# Precisely compute the cost for a piece-wise constant control

def piece_wise_cosntant_u_sample(u, res=10000):
    """ Takes the sample of a control u and returns a new sample of different resolution, interpolating u as a piece-wise constant function. """
    new_u_indices = np.arange(res)
    return u[np.floor(u.shape[0]/res*new_u_indices).astype(int)]

def piece_wise_linear_x_sample(x, res_mul=1000):
    """ Takes the sample of a trajectory x and returns a new sample of different resolution, interpolating u as a piece-wise linear function. """
    new_x = np.zeros(res_mul*(x.shape[0]-1)+1)
    dt = 1/res_mul
    for current_segment in range(x.shape[0]-1):
        for k in range(res_mul):
            new_x[current_segment*res_mul+k] = x[current_segment] + dt*k*(x[current_segment+1] - x[current_segment])
    new_x[-1] = x[-1]
    return new_x

def piece_wise_constant_u_cost(x, u, res_u=10000, res_mul_x=1000):
    """ Compute an approximation of the cost function evaluated in a given trajectory-control pair (x, u) with u piece-wise constant, using trapezoidal rule. u is interpolated as a piece-wise constant function and sampled using resolution `res`. """
    new_u = piece_wise_cosntant_u_sample(u, res=res_u)
    new_x = piece_wise_linear_x_sample(x, res_mul=res_mul_x)
    return cost(new_x, new_u)
