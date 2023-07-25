import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import params as p


# Plotting

def plot_trajectory(predicted_x=None, gt_x=None, crop_y=False, title="position"):
    """ Plots the ground truth trajectory vs the predicted trajectory on a (t, x) plot.
    
    :param predicted_x: Regularly spaced samples of the predicted trajectory. If None, this trajectory is not plotted.
    :type predicted_x: np.ndarray[float], optional
    :param gt_x: Regularly spaced samples of the ground truth trajectory. If None, this trajectory is not plotted.
    :type gt_x: np.ndarray[float], optional
    :param bool crop_y: If True, crop the y axis to zoom on [-1, 0].
    :param string title: the type of trajectory that is plotted, as displayed in the title of the plot.
    """
    plt.figure()
    plt.grid()
    if predicted_x is not None:
        t_prediction = np.linspace(0, p.T, predicted_x.shape[0])
        plt.plot(t_prediction, predicted_x, label="Prediction", color="tab:blue")
    if gt_x is not None:
        t_gt = np.linspace(0, p.T, gt_x.shape[0])
        plt.plot(t_gt, gt_x, label="Ground truth", color="tab:orange")
    if crop_y:
        plt.ylim([-1, 0])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Trajectories ({title})")
    plt.legend()
    plt.show()


def plot_control(predicted_u=None, gt_u=None):
    """ Plots the ground truth control vs the predicted control on a (t, u) plot.
    
    :param predicted_u: Regularly spaced samples of the predicted control. If None, this control is not plotted.
    :type predicted_u: np.ndarray[float], optional
    :param gt_u: Regularly spaced samples of the ground truth control. If None, this control is not plotted.
    :type gt_u: np.ndarray[float], optional
    """
    plt.figure()
    plt.grid()
    if predicted_u is not None:
        # We should be careful about the fact that the last action is taken at the last step but one.
        dt_prediction = p.T/predicted_u.shape[0]
        t_prediction = np.linspace(0, p.T-dt_prediction, predicted_u.shape[0])
        plt.scatter(t_prediction, predicted_u, label="Prediction", color="tab:blue", marker="x")
    if gt_u is not None:
        t_gt = np.linspace(0, p.T, gt_u.shape[0])
        plt.plot(t_gt, gt_u, label="Ground truth", color="tab:orange")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("Controls")
    plt.legend()
    plt.show()


def plot_V_x(V, v=0, state_zoom=None, log_scale=True, zoom_time=0):
    """ Plots the value function on an (x, n) axis (where n is the time step) for a fixed velocity v.
    
    :param V: The value function to plot.
    :type V: np.ndarray[float] of dimension N x ((p.X_R-p.X_L)*p.N_X+1) x ((p.V_R-p.V_L)*p.N_V+1).
    :param float v: The fixed velocity at which to plot V.
    :param np.ndarray[float] state_zoom: Values of V for states smaller or greater than the boundaries of this state window are not plotted. If None, takes the whole state space.
    :param bool log_scale: Whether to use a log scale for the colorbar. Defaults to True.
    :param int zoom_time: The distance between 2 ticks on the n axis is multiplied by 2^(zoom_time), to make the plot more readable. Otherwise, the plot might be too streched in x.
    """
    state_zoom = state_zoom if state_zoom else (0, V.shape[1]-1)
    truncated_V = V[:,state_zoom[0]:state_zoom[1]+1, to_arr_v(v)]
    truncated_V = np.expand_dims(truncated_V, axis=1)
    for _ in range(zoom_time):
        truncated_V = np.concatenate([truncated_V, truncated_V] , axis=1)
    truncated_V = np.reshape(truncated_V, [truncated_V.shape[0]*truncated_V.shape[1], truncated_V.shape[2]])
    plt.figure()
    if log_scale:
        plt.imshow(truncated_V, norm=LogNorm())
    else:
        plt.imshow(truncated_V)
    xticks = np.append(np.arange(0, truncated_V.shape[1], truncated_V.shape[1]//5), (to_arr_x(0)-state_zoom[0]))
    plt.xticks(ticks=xticks, labels=[f"{x:.2f}" for x in from_arr_x(np.arange(state_zoom[0], state_zoom[1]+1, truncated_V.shape[1]//5))]+[0])
    yticks = np.arange(0, truncated_V.shape[0]+1, truncated_V.shape[0]//2)
    plt.yticks(ticks=yticks, labels=[n/(2**zoom_time) for n in yticks])
    plt.colorbar(location="bottom")
    plt.title(f"Value function ($v$={v})")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.show()
    
    
def plot_V_v(V, x=0, state_zoom=None, log_scale=True, zoom_time=0):
    """ Plots the value function on an (v, n) axis (where n is the time step) for a fixed position x.
    
    :param V: The value function to plot.
    :type V: np.ndarray[float] of dimension N x ((p.X_R-p.X_L)*p.N_X+1) x ((p.V_R-p.V_L)*p.N_V+1).
    :param float x: The fixed position at which to plot V.
    :param np.ndarray[float] state_zoom: Values of V for states smaller or greater than the boundaries of this state window are not plotted. If None, takes the whole state space.
    :param bool log_scale: Whether to use a log scale for the colorbar. Defaults to True.
    :param int zoom_time: The distance between 2 ticks on the n axis is multiplied by 2^(zoom_time), to make the plot more readable. Otherwise, the plot might be too streched in x.
    """
    state_zoom = state_zoom if state_zoom else (0, V.shape[2]-1)
    truncated_V = V[:, to_arr_x(x), state_zoom[0]:state_zoom[1]+1]
    truncated_V = np.expand_dims(truncated_V, axis=1)
    for _ in range(zoom_time):
        truncated_V = np.concatenate([truncated_V, truncated_V] , axis=1)
    truncated_V = np.reshape(truncated_V, [truncated_V.shape[0]*truncated_V.shape[1], truncated_V.shape[2]])
    plt.figure()
    if log_scale:
        plt.imshow(truncated_V, norm=LogNorm())
    else:
        plt.imshow(truncated_V)
    xticks = np.append(np.arange(0, truncated_V.shape[1], truncated_V.shape[1]//5), (to_arr_v(0)-state_zoom[0]))
    plt.xticks(ticks=xticks, labels=[f"{v:.2f}" for v in from_arr_v(np.arange(state_zoom[0], state_zoom[1]+1, truncated_V.shape[1]//5))]+[0])
    yticks = np.arange(0, truncated_V.shape[0]+1, truncated_V.shape[0]//2)
    plt.yticks(ticks=yticks, labels=[n/(2**zoom_time) for n in yticks])
    plt.colorbar(location="bottom")
    plt.title(f"Value function ($x$={x})")
    plt.xlabel("v")
    plt.ylabel("n")
    plt.show()


def plot_training_evaluations(log_folder, reference_reward=None, crop_reward=None):
    """ Plots the evaluation scores obtained over training steps.
    An evaluation score is the average reward obtained by the RL agent over a number of runs that was chosen during training.

    :param string log_folder: path to the folder containing the evaluation logs.
    :param (float, optional) reference_reward: A reward used as a reference, typically the optimal reward in the worse case where x_0=-1. Defaults to None.
    :param (np.array(float, float), optional) crop_reward: Interval of reward values that will be shown on the y-axis. If None, it is set automatically to fit extremal values of the data. Defaults to None.
    """
    log = os.path.join(log_folder, "evaluations.npz")
    plt.figure("Evaluations during training")
    evaluation_data = np.load(log)
    timesteps = evaluation_data["timesteps"]
    accumulated_rewards = evaluation_data["results"]
    plt.plot(timesteps, accumulated_rewards.mean(axis=1), label="mean")
    plt.fill_between(timesteps, accumulated_rewards.mean(axis=1) - accumulated_rewards.std(axis=1), accumulated_rewards.mean(axis=1) + accumulated_rewards.std(axis=1), alpha=0.4, label="std", color="gray")
    plt.fill_between(timesteps, accumulated_rewards.min(axis=1), accumulated_rewards.max(axis=1), alpha=0.2, label="min-max", color="gray")
    if reference_reward is not None:
        plt.plot([0, timesteps[-1]], [reference_reward, reference_reward], color="r", label="Reference reward")
    plt.title("Estimated average accumulated reward over training timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    if crop_reward is not None:
        plt.ylim(crop_reward)
    plt.legend()
    plt.grid()
    plt.show()


# Conversion from float to array indices

def transform_interval(a, b, c, d, x):
    """ Affine transformation that maps an interval [a, b] to an interval [c, d].
    
    :param int a: Lower bound of the departure interval.
    :param int b: Upper bound of the departure interval.
    :param int c: Lower bound of the departure interval.
    :param int d: Upper bound of the departure interval.
    :param float x: The point to map from [a, b] to [c, d].
    :return float: The point of [c, d] that is the image of x by the affine transformation.
    """
    return (d-c)/(b-a)*x + (c*b-d*a)/(b-a)

to_arr_x = lambda x: round(transform_interval(p.X_L, p.X_R, 0, (p.X_R-p.X_L)*p.N_X, x))  # This is to convert a position to an index for value functions and policies
from_arr_x = lambda x: transform_interval(0, (p.X_R-p.X_L)*p.N_X, p.X_L, p.X_R, x)  # Inverse of the previous
to_arr_v = lambda v: round(transform_interval(p.V_L, p.V_R, 0, (p.V_R-p.V_L)*p.N_V, v))  # This is to convert a velocity to an index for value functions and policies
from_arr_v = lambda v: transform_interval(0, (p.V_R-p.V_L)*p.N_V, p.V_L, p.V_R, v)  # Inverse of the previous
to_arr_a = lambda a: round(transform_interval(p.U_L, p.U_R, 0, (p.U_R-p.U_L)*p.N_U, a))  # This is to convert an action to an index for value functions and policies
from_arr_a = lambda a: transform_interval(0, (p.U_R-p.U_L)*p.N_U, p.U_L, p.U_R, a)  # Inverse of the previous
