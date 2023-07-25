# Reinforcement learning for optimal control

This code is the implementation of the following post: https://dcn.nat.fau.eu/reinforcement-learning-as-a-new-perspective-into-controlling-physical-systems/

This repo contains the implementation of reinforcement learning (RL) algorithms for two linear-quadratic optimal control problems. Both consist in pushing a cart along a 1D-axis from a random initial position to a fixed target position; but each one illustrates different aspects of RL:
1. The `Sped-up-cart` folder involves a reduced version of the problem for visualization purposes. It implements the _Q-learning_ algorithm and visualizes the `Q` array for various parameters of the algorithm. It illustrates the _exploration-exploitation_ dilemma in RL.
2. The `Accelerated-cart` folder solves the full-fledged problem by training pre-implemented RL algorithms from the [`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/) library. The problem is also solved using an adjoint method and the two approaches are compared.

Each of the two projects consists of source code and a notebook which defines and calls the main functions and gives comments on the results. The `params.py` file of each folder also defines the default parameters for the problem and the algorithms.

## Requirements

To run the code, it is advised to create a new virtual Python environment and to install [Jupyter](https://jupyter.org/install).

Then, the required libraries are listed in the `requirements.txt` file of each project folder. To install them, simply enter the corresponding folder and run
```bash
pip install -r requirements.txt
```
