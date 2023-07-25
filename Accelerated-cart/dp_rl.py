import os
import json
import gymnasium as gym
import numpy as np
import params as p
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# Costs
def  running_cost(x_n, v_n, a_n):
    """ Running cost for the dynamic programming formulation. """
    return p.DT*a_n**2

def  final_cost(x_N, v_N):
    """ Final cost for the dynamic programming formulation. """
    return p.LAMBDA_P*x_N**2 + p.LAMBDA_V*v_N**2


# Reachable states for dynamic programming
def reachable_x(t):
    """ Returns the set of positions that are reachable in time t from x_0 varying in [-1, 0].
    Useful in dynamic programming for computing the value function only on relevant states. """
    return np.arange(-1 + p.U_L*(t**2)/(2*p.M_REAL), p.U_R*(t**2)/(2*p.M_REAL) + p.DX/2, p.DX)

def reachable_v(t):
    """ Returns the set of velocities that are reachable in time t from a null velocity at time 0.
    Useful in dynamic programming for computing the value function only on relevant states. """
    return np.arange(p.U_L/p.M_REAL*t, p.U_R/p.M_REAL*t + p.DV/2, p.DV)


# Reinforcement learning utilitaries
def replace_best_model(model_path):
    """ When resuming the training of an RL model, this function is used to update the best stored 
    model with the one obtained with the new training, if necessary. """
    previous_best_score = np.max(np.mean(np.load(os.path.join(model_path, "old_evaluations.npz"))["results"], axis=-1))
    new_best_score = np.max(np.mean(np.load(os.path.join(model_path, "evaluations.npz"))["results"], axis=-1))
    if new_best_score <= previous_best_score:
        print("No improvement made from previous training.")
        os.remove(os.path.join(model_path, "best_model.zip"))
        os.rename(os.path.join(model_path, "old_best_model.zip"),os.path.join(model_path, "best_model.zip"))
    else:
        print("Improvement made from previous training.")
        os.remove(os.path.join(model_path, "old_best_model.zip"))


def merge_eval_logs(model_path):
    """ When resuming the training of an RL model, this function is used to update the previous training logs
    with the logs obtained from the new training. """
    old_evaluations = np.load(os.path.join(model_path, "old_evaluations.npz"))
    new_evaluations = np.load(os.path.join(model_path, "evaluations.npz"))
    np.savez(
        os.path.join(model_path, "evaluations"),
        timesteps=np.append(old_evaluations["timesteps"], new_evaluations["timesteps"]),
        results=np.append(old_evaluations["results"], new_evaluations["results"], axis=0),
        ep_lengths=np.append(old_evaluations["ep_lengths"], new_evaluations["ep_lengths"], axis=0)
    )
    os.remove(os.path.join(model_path, "old_evaluations.npz"))


def instantiate_model(model_name, Algo, hyperparameters, n_envs=None, verbose=0):
    """ Creates a folder for the RL model to be trained in the "Agents" folder. This folder will contain the model itself (architecture, weights; best model and latest model),
    the parameters for the cart problem used, the hyperparameters used, monitoring logs for the training and logs for the evaluations made throughout training.
    If the folder already exists, this function simply reloads the existing model.
    
    :param string model_name: The name of the model. This will also be the name of the RL model.
    :param Algo: the RL algorithm to use.
    :type Algo: stable_baselines3 class for the agorithm. 
    :param dict hyperparameters: A dictionary containing the hyperparameters to use for the model.
    :param (int, optional) n_envs: The number of envs to use for parallelized training. If None, no parallelization is applied. Defaults to None.
    :param (int, optional) verbose: If 1, will print info on the training process. If 0, only prints evaluation results throughout the training. Defaults to 0.
    :return: The model path, created/loaded model and env used for model instanciation.
    :rtype: string, Algo, gym.Env.
    """
    model_path = os.path.join("Agents", model_name)
    existing_model = os.path.exists(model_path)
    current_problem_parameters = {
        "M_REAL": p.M_REAL,
        "LAMBDA_P": p.LAMBDA_P,
        "LAMBDA_V": p.LAMBDA_V,
        "T": p.T,
        "N": p.N,
        "U_L": p.U_L,
        "U_R": p.U_R,
        "N_U": p.N_U,
        "V_L": p.V_L,
        "V_R": p.V_R,
        "N_V": p.N_V,
        "X_L": p.X_L,
        "X_R": p.X_R,
        "N_X": p.N_X
    }

    if existing_model:
        print("Loading a pre-existing model.")
        if n_envs is None:
            env = Monitor(gym.make("AcceleratedCart-v1", render_mode=None), filename=model_path, override_existing=False)
        else:
            env = make_vec_env("AcceleratedCart-v1", n_envs=n_envs, monitor_dir=model_path, monitor_kwargs=dict(override_existing=False), env_kwargs=dict(render_mode=None))
        model = Algo.load(os.path.join(model_path, "latest_model"), env=env, verbose=verbose)
        
        with open(os.path.join(model_path, "hyperparameters.json"), "r") as hyperparameters_file:
            hyperparameters = json.load(hyperparameters_file)
        with open(os.path.join(model_path, "problem.json"), "r") as problems_parameters_file:
            if current_problem_parameters != json.load(problems_parameters_file):
                print("WARNING: The loaded model was trained for a different set of problem parameters!")
        
    else:
        print("Creating a new model.")
        os.makedirs(model_path)
        if n_envs is None:
            env = Monitor(gym.make("AcceleratedCart-v1", render_mode=None), filename=model_path, override_existing=True)
        else:
            env = make_vec_env("AcceleratedCart-v1", n_envs=n_envs, monitor_dir=model_path, monitor_kwargs=dict(override_existing=True), env_kwargs=dict(render_mode=None))

        with open(os.path.join(model_path, "hyperparameters.json"), "w") as hyperparameters_file:
            # TODO: parse hyperparameters dictionary
            saved_hyperparamaters = hyperparameters.copy()
            # saved_hyperparamaters.pop("learning_rate")  # These can't be saved simply in the the json file, needs parsing 
            # saved_hyperparamaters.pop("clip_range")  # Idem
            json.dump(saved_hyperparamaters, hyperparameters_file, indent=4)
        with open(os.path.join(model_path, "problem.json"), "w") as problems_parameters:
            json.dump(current_problem_parameters, problems_parameters, indent=4)

        model = Algo("MultiInputPolicy", env, **hyperparameters, verbose=verbose)
    
    return model_path, model, env


def train_model(model, model_path, training_steps=50_000, eval_freq=500, n_envs=1):
    """ Trains the specified model (name and path), running n_envs training episodes simultaneously.
    Training stops after training_steps time steps have been simulated, counting each parallel episode independantly.
    Evaluations are carried out every n_envs*eval_freq time steps, evaluating on 50 episodes. """
    old_eval_logs = os.path.exists(os.path.join(model_path, "evaluations.npz")) and os.path.exists(os.path.join(model_path, "best_model.zip"))
    assert os.path.exists(os.path.join(model_path, "best_model.zip")) == old_eval_logs and os.path.exists(os.path.join(model_path, "evaluations.npz")) == old_eval_logs
    if old_eval_logs:
        os.rename(os.path.join(model_path, "evaluations.npz"), os.path.join(model_path, "old_evaluations.npz"))
        os.rename(os.path.join(model_path, "best_model.zip"), os.path.join(model_path, "old_best_model.zip"))
        model.env = make_vec_env("AcceleratedCart-v1", n_envs=n_envs, monitor_dir=model_path, monitor_kwargs=dict(override_existing=False), env_kwargs=dict(render_mode=None))

    # Separate evaluation env
    eval_env = make_vec_env("AcceleratedCart-v1", n_envs=1, env_kwargs=dict(render_mode=None))
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path, log_path=model_path, eval_freq=eval_freq, n_eval_episodes=50, deterministic=True, render=False)
    try:
        model.learn(training_steps, eval_callback, reset_num_timesteps=not old_eval_logs, log_interval=None)
    except KeyboardInterrupt:
        print("Training interrupted by keyboard.")
    finally:
        print("End of learning.")

        if old_eval_logs:
            replace_best_model(model_path)
            merge_eval_logs(model_path)

        # Save latest model, for example to resume trainng later.
        model.save(os.path.join(model_path, "latest_model"))
