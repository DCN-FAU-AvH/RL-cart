# Physical problem
M_REAL = 1      # Mass of the cart
LAMBDA_P = 100  # Penalization for not reaching the target at time T
LAMBDA_V = 50   # Penalization for not having null velocity at time T
T = 1           # Time horizon of the problem

# Time discretization
N = 50    # Number of time steps for the simulator
DT = T/N  # Length of a time step

# Space and action discretizations
U_L, U_R = -6, 6  # Bounds for the control
N_U = 4  # Resolution of the control
DU = 1/N_U  # Step between two possible values of the control
V_L, V_R = U_L/M_REAL*T, U_R/M_REAL*T  # Bounds for the velocity
DV = DU/M_REAL*DT  # Step between two possible values of the velocity
N_V = round(1/DV)  # Resolution of the velocity
X_L, X_R = -1+U_L/(2*M_REAL)*T**2, U_R/(2*M_REAL)*T**2  # Bounds for the position
DX = (DT**2)/(2*M_REAL)*DU  # Step between two possible values of the position
N_X = round(1/DX)  # Resolution of the position


def print_current_parameters(physical_problem=True, time_discretization=False, space_action_discretizations=False):
    """ Prints current global parameters for the program.

    :param (bool, optional) physical_problem: Whether to print parameters for the physical problem. Defaults to True.
    :param (bool, optional) time_discretization: Whether to print parameters for discretization in time. Defaults to False.
    :param (bool, optional) space_action_discretizations: Whether to print parameters for discretization in space and action. Defaults to False.
    """
    physical_problem_parameters = {
        "M_REAL": M_REAL,
        "LAMBDA_P": LAMBDA_P,
        "LAMBDA_V": LAMBDA_V,
        "T": T
    }
    time_parameters = {"N": N, "DT": DT}
    space_action_problem_parameters = {
        "U_L": U_L,
        "U_R": U_R,
        "N_U": N_U,
        "V_L": V_L,
        "V_R": V_R,
        "N_V": N_V,
        "X_L": X_L,
        "X_R": X_R,
        "N_X": N_X
    }
    if physical_problem:
        for item, value in physical_problem_parameters.items():
            print(f"{item}:", value)
    if time_discretization:
        for item, value in time_parameters.items():
            print(f"{item}:", value)
    if space_action_discretizations:
        for item, value in space_action_problem_parameters.items():
            print(f"{item}:", value)


def update_action_space_parameters(new_N_U=N_U, new_U_L=U_L, new_U_R=U_R):
    """ Updates the action (and space) parameters with a new value for U_L, U_R and N_U. """
    global U_L, U_R, N_U, DU, V_L, V_R, DV, N_V, X_L, X_R, DX, N_X
    U_L, U_R = new_U_L, new_U_R
    N_U = new_N_U
    DU = 1/N_U
    V_L, V_R = U_L/M_REAL*T, U_R/M_REAL*T
    DV = DU/M_REAL*DT
    N_V = round(1/DV)
    X_L, X_R = -1+U_L/(2*M_REAL)*T**2, U_R/(2*M_REAL)*T**2
    DX = (DT**2)/(2*M_REAL)*DU
    N_X = round(1/DX)


def update_time_parameters(new_N=N):
    """ Updates the time parameters with a new value for N. """
    global N, DT
    N = new_N
    DT = T/N
    update_action_space_parameters(N_U, U_L, U_R)


def update_physical_parameters(new_M_REAL=M_REAL, new_LAMBDA_P=LAMBDA_P, new_LAMBDA_V=LAMBDA_V, new_T=T):
    """ Updates the time parameters with a new value for N. """
    global M_REAL, LAMBDA_P, LAMBDA_V, T
    M_REAL = new_M_REAL
    LAMBDA_P = new_LAMBDA_P
    LAMBDA_V = new_LAMBDA_V
    T = new_T
    update_time_parameters(new_N=N)
