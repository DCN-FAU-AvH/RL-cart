import matplotlib.pyplot as plt
import numpy as np

from params import *


# Plotting functions

def plot_trajectory(predicted_x=None, gt_x=None, crop_y=False):
    """ Plots the ground truth trajectory vs the predicted trajectory on a (t, x) plot. """
    plt.figure()
    if predicted_x is not None:
        t_prediction = np.linspace(0, T, predicted_x.shape[0])
        plt.plot(t_prediction, predicted_x, label="Prediction", color="tab:blue")
    if gt_x is not None:
        t_gt = np.linspace(0, T, gt_x.shape[0])
        plt.plot(t_gt, gt_x, label="Ground truth", color="tab:orange")
    if crop_y:
        plt.ylim([-1, 0])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Trajectories")
    plt.legend()
    plt.grid()
    plt.show()
    

def plot_control(predicted_u=None, gt_u=None):
    """ Plots the ground truth control vs the predicted control on a (t, u) plot. """
    plt.figure()
    if predicted_u is not None:
        t_prediction = np.linspace(0, T, predicted_u.shape[0])
        plt.scatter(t_prediction, predicted_u, label="Prediction", color="tab:blue", marker="x")
    if gt_u is not None:
        t_gt = np.linspace(0, T, gt_u.shape[0])
        plt.plot(t_gt, gt_u, label="Ground truth", color="tab:orange")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("Controls")
    plt.legend()
    plt.grid()
    plt.show()
    
    
def plot_reward_trajectory(reward_trajectory):
    """ Plots the reward at each time step on a (t, r) plot. """
    plt.figure()
    t = np.linspace(0, T, N+1)
    plt.plot(t, reward_trajectory)
    plt.xlabel("t")
    plt.ylabel("r")
    plt.title("Reward trajectory")
    plt.grid()
    plt.show()


def plot_loss_history(loss_history, evaluate_every=1, function_name="loss"):
    """ Plots the loss history over episodes, using the indicated function name for the title of the plot. """
    plt.figure()
    plt.plot(np.arange(0, evaluate_every*len(loss_history), evaluate_every), loss_history)
    plt.title(f"History of the {function_name} during training")
    plt.xlabel("Episodes")
    plt.ylabel(function_name)
    plt.grid()
    plt.show()

def plot_reward_history(reward_history, evaluate_every=1, crop_reward=None):
    """
    Plots the reward_history history over episodes, with min, max and std.
    The crop_reward indicates the interval of reward values to show on the y axis. If None, scale automatically.
    """
    plt.figure()
    episodes = np.arange(0, evaluate_every*len(reward_history), evaluate_every)
    reward_history = np.array(reward_history)
    plt.plot(episodes, reward_history.mean(axis=1), label="mean")
    plt.fill_between(episodes, reward_history.mean(axis=1) - reward_history.std(axis=1), reward_history.mean(axis=1) + reward_history.std(axis=1), alpha=0.4, label="std", color="gray")
    plt.fill_between(episodes, reward_history.min(axis=1), reward_history.max(axis=1), alpha=0.2, label="min-max", color="gray")
    plt.title("History of the reward during training")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    if crop_reward is not None:
        plt.ylim(crop_reward)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    

def plot_Q(Q, n, state_zoom=None, colorbar_limits=None, log_scale=False):
    """
    Plots the heatmap of the weights of Q (represented as a matrix) at time n*\Delta_t. Note that for convenience, Q is transposed for the representation.
    The x and y axes are scaled to represent the actions and states in U and \Omega, not U^{RL} and \Omega^{RL}.
    
    Parameters
    ----------
    Q: np.array of size (T, |\Omega^RL|, |A|) with \Omega^RL and A the spatial-state and action spaces
        The array representing the Q function.
    n: int
        The time step at which we represent Q.
    state_zoom: None | tuple(int, int)
        Interval of the state to show on the heatmap. If None, shows the entire state space.
    colorbar_limits: None | tuple(float, float)
        Limits for the colorbar: values of Q above these bounds will be truncated to the closest bound. If None, no limit is applied.
    log_scale: bool
        If True, use a log scale for the color bar; otherwise use linear scale. Defaults to True.
    """
    Q = Q[n]
    state_zoom = state_zoom if state_zoom else (0, Q.shape[0]-1)
    colorbar_limits = (None, None) if colorbar_limits is None else colorbar_limits
    truncated_Q = Q[state_zoom[0]:state_zoom[1]+1, :]
    plt.figure()
    if log_scale:
        plt.imshow(np.transpose(truncated_Q), norm=SymLogNorm(linthresh=np.min(-truncated_Q)/10))
    else:
        plt.imshow(np.transpose(truncated_Q), vmin=colorbar_limits[0], vmax=colorbar_limits[1])
    xticks = np.append(np.arange(0, truncated_Q.shape[0], truncated_Q.shape[0]//5), (x_to_x_RL(0)-state_zoom[0]))
    plt.xticks(ticks=xticks, labels=[f"{x:.2f}" for x in x_RL_to_x(np.arange(state_zoom[0], state_zoom[1]+1, truncated_Q.shape[0]//5))]+[0])
    yticks = np.array([0, u_to_u_RL(0), Q.shape[1]-1])
    plt.yticks(ticks=yticks, labels=[Ul, 0, Ur])
    plt.colorbar(location="bottom")
    plt.title(f"Q function at time step {n}")
    plt.xlabel("s")
    plt.ylabel("a")
    plt.show()


# Transforms

def transform_interval(a, b, c, d, x):
    """ Affine transformation that maps an interval [a, b] to an interval [c, d]. """
    return (d-c)/(b-a)*x + (c*b-d*a)/(b-a)

u_to_u_RL = lambda u: round(transform_interval(Ul, Ur, 0, (Ur-Ul)*N_U, u))  # This is to convert a control to an index for the np.array Q
u_RL_to_u = lambda u: transform_interval(0, (Ur-Ul)*N_U, Ul, Ur, u)  # Inverse of the previous 
x_to_x_RL = lambda x: round(transform_interval(OMEGAl, OMEGAr, 0, (OMEGAr-OMEGAl)*N_OMEGA, x))  # This is to convert a state to an index for the np.array Q
x_RL_to_x = lambda x: transform_interval(0, (OMEGAr-OMEGAl)*N_OMEGA, OMEGAl, OMEGAr, x)  # Inverse of the previous 
