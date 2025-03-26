from collections import defaultdict
from matplotlib import pyplot as plt
import torch
import gym.envs
from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
import torch.nn as nn
import multiprocessing
import gym
from tqdm import tqdm
from torchrl.envs import GymWrapper, StepCounter, TransformedEnv, InitTracker, Compose
from torchrl.envs import check_env_specs
from torchrl.objectives import DDPGLoss, SoftUpdate, SACLoss
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl._utils import logger as torchrl_logger
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, ValueOperator, TanhNormal, LSTMModule

from relapse.env.relapse_env import RelapseEnv

# Use matplotlib style `bmh`
plt.style.use('bmh')
plt.rcParams["font.family"] = "monospace"

# Check if GPU (CUDA) access is available
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# Global configuration variables
buffer_size = 5e5
max_steps = 2e6 
lr = 1e-4
init_random_steps = 5_000

# Environment initialization
env = GymWrapper(RelapseEnv(100,300))
check_env_specs(env)
env = TransformedEnv(
    env, 
    Compose(
        StepCounter(),
    )
)   

actor_mlp = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
    nn.Tanh(),
    NormalParamExtractor()
).to(device)

actor_mlp_mod = Mod(
    actor_mlp,
    in_keys=["observation"],
    out_keys=["loc","scale"]
)

actor = ProbabilisticActor(
    actor_mlp_mod,
    in_keys=["loc","scale"],
    out_keys=["action"],
    spec=env.action_spec,
    distribution_kwargs = {"low":0.0,"high":1.0},
    distribution_class = TanhNormal,
    safe = True
)

class ValueMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        """
        Initialize the ValueMLP network.

        Args:
            input_dim (int): Number of input features. Defaults to 3, i.e., 2+1.
            hidden_dim (int): Number of neurons in the hidden layers. Defaults to 64.
        """
        super(ValueMLP, self).__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, action):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.cat([x, action], dim=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    
value_net = Mod(ValueMLP(),in_keys=["observation","action"],out_keys=["state_action_value"])

qvalue = ValueOperator(
    value_net.to(device),
    in_keys=["observation","action"], 
    out_keys=["state_action_value"]
)

# Set the uninitialized parameters
actor(env.reset().to(device))
td = env.rollout(3).to(device)
qvalue(td)

loss = SACLoss( # DDPGLoss for DDPG
    actor_network=actor,
    qvalue_network=qvalue,
)


# Set update rule for the target network
updater = SoftUpdate(loss,eps=0.95)

# Replay Buffer to store transitions
memory = ReplayBuffer(
   storage=LazyTensorStorage(buffer_size)
)

# Synchronized Data collector
fpb = 100
collector = SyncDataCollector(
    env,
    policy=actor,
    frames_per_batch=fpb,
    total_frames=max_steps,
    init_random_frames=init_random_steps,
    device=device,
)

# Optimizer
optim = torch.optim.Adam(loss.parameters(),lr)

# Main Loop
logs = defaultdict(list)


# Progress bar
pbar = tqdm(total=max_steps, desc="Training", unit=" steps")
def train():
    for i, data in enumerate(collector):
        
        pbar.update(fpb)
        memory.extend(data)
        if len(memory) > init_random_steps:
            avg_length: torch.Tensor = memory[:]["next", "step_count"]
            average_step_count = avg_length.type(torch.float).mean().item()
            sample = memory.sample(128)
            for _ in range(5): # optim loop for sample efficiency
                loss_vals = loss(sample.to(device))
                tot_loss = loss_vals["loss_actor"] + loss_vals["loss_qvalue"] + loss_vals["loss_alpha"]
                tot_loss.backward()
                optim.step()
                optim.zero_grad()
                # updater.step() # uncomment for soft update when using DDPG
            if i % 50 == 0:
                # Validate the policy
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(50,policy=actor)
                    torch.save(actor.state_dict(), "actor.pth")
                    # print(
                    #     f"Min action: {(eval_rollout['action'].min().item()):.2f}, "
                    #     f"Max action: {(eval_rollout['action'].max().item()):.2f}"
                    # )
                    logs["stepno."].append(i*fpb)
                    logs["eval reward"].append(
                        eval_rollout["next", "reward"].mean().item()
                    )
                    logs["eval step count"].append(
                        eval_rollout["next", "step_count"].type(torch.float).mean().item()
                    )
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["average step count"].append(average_step_count)
                    eval_str = (
                        f"eval cum. reward: {logs['eval reward (sum)'][-1]: 4.3f}|"
                        f"average step count: {logs['average step count'][-1]: 4.3f}|"
                        f"average eval step count: {logs['eval step count'][-1]: 4.3f}"
                    )
                    fig, ax = plt.subplots()
                    pbar.desc = eval_str
                    ax.plot(
                        logs["stepno."], 
                        logs["eval reward (sum)"], 
                        label="Cum. eval reward"
                    )
                    ax.plot(
                        logs["stepno."], 
                        logs["average step count"], 
                        label="Average step count"
                    )
                    ax.legend()
                    ax.set_xlabel("Step number")
                    ax.set_ylim(-10, 50)
                    plt.savefig("eval_reward.png")
                    plt.close()
                    #torchrl_logger.info(eval_str)
                    del eval_rollout