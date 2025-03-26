import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchrl.envs.utils import set_exploration_type, ExplorationType

from relapse.train.relapse_train import actor
from relapse.env.relapse_env import env


def validate(path_to_actor: str):
    valenv = env
    valenv.validation = True
    
    
    save_dir = Path("resplots")
    save_dir.mkdir(exist_ok=True)
    
    # Initialize the environment
    actor.load_state_dict(torch.load(path_to_actor))
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            
            for i in range(50):
                # Perform the rollout
                rollout = valenv.rollout(50,policy=actor)
                
                # Get states
                states = rollout["observation"].cpu().numpy()
                measurements, stop = rollout["action"].cpu().numpy().T
                
                # Get the first index where stop is larger than 0.5
                stop = np.where(stop > 0.5)[0]
                
                states[:,0] /= valenv.normalizer
                measurements = measurements * env.max_measurement_gap
                
                positions = np.zeros(np.size(measurements)) + np.cumsum(measurements)
                
                # Get measurement funciton
                measure = valenv.measure
                
                # get measurement y
                measurement_y = [measure(_p) for _p in positions]
                
                # Get the relapse point
                relapse_point = valenv._get_relapse_point()
                
                # Get the relapse y
                relapse_y = valenv.relapse_y
                
                # Sample points for plotting
                _x = np.linspace(0,env.relapse_upper_bound + 100,200)
                
                # Plot the target function
                f, ax = plt.subplots()
                ax.plot(_x,measure(_x),label="Target function", color="#bc6c25")
                ax.hlines(
                    relapse_y,
                    0,
                    1000,
                    linestyles="--",
                    label="Relapse Time",
                    color="#780000"
                )
                
                # Plot the measurements
                ax.vlines(
                    positions[:-1],
                    np.array(measurement_y)[:-1] - 15,
                    np.array(measurement_y)[:-1]
                    ,linestyles="-",
                    label="Measurements",
                    color="#283618"
                )
                ax.scatter(
                    positions[:-1],
                    measurement_y[:-1],
                    color="#283618"
                )
                ax.vlines(
                    positions[stop],
                    np.array(measurement_y)[stop] - 15,
                    np.array(measurement_y)[stop],
                    linestyles="--",
                    label="Stop",
                    color="r"
                )
                ax.scatter(
                    positions[stop],
                    np.array(measurement_y)[stop],
                    color="r"
                )
                
                ax.text(
                    200,
                    20,
                    "Cumulative Reward: {:.2f}".format(rollout["next", "reward"].sum().item()),
                    color="#780000"
                )
                
                ax.set_xlabel("Days")
                ax.set_ylabel("Measurement")
                ax.set_xlim(-20,500)
                ax.set_ylim(-20,200)
                
                plt.legend(loc = "upper right",ncol=2)
                plt.savefig(f"{save_dir}/relapse_{i}.png")
                plt.close()