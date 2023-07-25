# Cost parameter
LAMBDA = 10  # Target trajectory penalization

# Discretization parameters
T = 1  # Time horizon
N = 20  # Number of time steps
DT = T/N  # Time step
Ul = -8; Ur = 8  # Bounds between which we look for u
OMEGAl = -1 + Ul*T; OMEGAr = Ur*T  # Bounds between which x lives
N_U = 5  # Resolution of the discretization on u
Du = 1/N_U  # Disctretization step on u
N_OMEGA = int(N_U/DT)  #  Resolution of the discretization on x
Dx = 1/N_OMEGA  # Disctretization step on x
