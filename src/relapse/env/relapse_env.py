import gym 
import torch
import numpy as np
from gym.spaces import Box
from typing import Callable

State = np.ndarray
Reward = float
Done = bool
Truncated = bool
Info = dict[str,str]


class RelapseEnv(gym.Env):
    def __init__(self,
                 relapse_lower_bound = 100,
                 relapse_upper_bound = 200,
                 validation = False):
        super().__init__()
        
        self.action_space = Box(
            low=0,
            high=1,
            shape=(2,)
        )
        
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(4,)
        )
        
        # Relapse bounds in days (earliest and latest possible relapse)
        self.relapse_lower_bound = relapse_lower_bound
        self.relapse_upper_bound = relapse_upper_bound
        
        # Relapse measurement y value
        self.relapse_y = 100
        self.normalizer = 1 / self.relapse_y # Normalize the measurement ot [0,1]
        
        # Neural networks can struggle with large differneces in observation scales, 
        # so we scale the position to be 'closer' to the measurement 
        # (however the scaling is arbitrary)
        self.position_scaling = 1 / 300 

        self.validation = validation

        # Parameter schedulders for curriculum reward shaping
        #self.variance_scheduler = ValueScheduler(0.1, 0.5, 1e6)
        self.max_measurement_gap = 100 # days
        
        # Stop the episode after 20 steps
        self.episode_length = 20
    
    def _sample_random_exponential(self) -> tuple[Callable[[float],float], float, float]:
        """
        Returns a randomly parameterized exponential function of the form:
            f(x) = exp(rate * x) + offset
        
        Returns:
            function: A function f(x) that computes exp(rate * x).
        """
        # Sample the offset from a uniform distribution
        offset = np.random.uniform(0,self.relapse_y/2)

        # Construct lower and upper bounds for the rate,
        # such that the relapse point is within the bounds.
        # (I just solved the exponential function for the rate,
        # given the relapse point and the bounds)
        # rate = log(y-offset) / x
        lower_rate = np.log(self.relapse_y-offset) / self.relapse_upper_bound
        upper_rate = np.log(self.relapse_y-offset) / self.relapse_lower_bound
        rate_range = (lower_rate, upper_rate)
        # Randomly sample rate from uniform distribution over the given range
        rate = np.random.uniform(*rate_range)
        
        
        # Optional: print the chosen parameters for debugging purposes
        # print(f"Selected rate: {rate:.3f}, offset: {offset:.3f}")
        
        # Define the exponential function
        def f(x: float) -> float:
            return np.exp(rate * x) + offset
        
        return f, rate, offset
    
    def reset(self) -> tuple[State, Info]:
        (
            self.measure, 
            self.target_rate, 
            self.measurement
        ) = self._sample_random_exponential()
        
        self.position = 0 # Position in days
        self.step_number = 0
        self.state = np.array(
            [
                # t-1 Measurement
                0.0,
                self.position * self.position_scaling,
                # t Measurement
                self.measurement * self.normalizer,
                self.position * self.position_scaling
            ]
        )
        
        return self.state, {"info": "Environment reset"}
    
    def step(self, action) -> tuple[State, Reward, Done, Truncated, Info]:
        """
        Execute one time step within the environment.
        
        Takes an action from the agent, updates the environment state,
        and returns the next observation, reward, done flag, and additional info.
        
        The action consists of two components:
        1. step_width: How far to advance in time [0,1], scaled by max_measurement_gap
        2. stop: Whether to stop the episode (interpreted as stop if > 0.5)
        
        The environment moves forward by the specified step_width, takes a new
        (noisy) measurement at that position, and calculates the reward based on
        waiting time, proximity to relapse point, and stopping decision.
        
        Args:
            action: Either torch.Tensor or numpy.ndarray with shape (2,)
                   containing [step_width, stop] values
        
        Returns:
            tuple containing:
                state (np.ndarray): New environment observation after taking the action
                reward (float): Reward achieved by the action
                done (bool): Whether the episode has ended
                truncated (bool): Whether the episode was truncated (always False)
                info (dict): Additional information about the step
        """
        old_measurement = self.state[2]
        old_position = self.position
        
        self.step_number += 1
        # Update the state
        if isinstance(action, torch.Tensor):
            step_width, stop = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            step_width, stop = action
        else:
            raise ValueError("Action must be a torch.Tensor or numpy.ndarray")
            
        next_position = self.position + round(float(step_width * self.max_measurement_gap))
        reward = self._get_reward(step_width, next_position, stop)
        done = stop > 0.5 or self.step_number >= self.episode_length
        
        # Update measurement position
        self.position += round(float(step_width * self.max_measurement_gap))

        self.state = np.array(
            [   
                # t-1 Measurement
                old_measurement,
                old_position * self.position_scaling,
                # Noisy measurement
                (self.measure(next_position)+np.random.normal(scale=2))*self.normalizer, 
                self.position * self.position_scaling
            ]
        )
        #self.variance_scheduler.next()
        
        # Update the curriculum parameters
        #                                  v----Trunctation
        return self.state, reward, done, False,{"info": "Step completed"}

    
    def _get_reward(self, wait_time, next_position, stop):
        """
        Calculates the reward based on wait time, position, and stopping decision.
        
        The reward consists of three components:
        1. wait_time_reward: Quadratic reward for waiting longer between measurements
        2. closeness_reward: Reward based on proximity to the relapse point
        3. perfect_stop: Bonus reward for correctly stopping near the relapse point
        
        Args:
            wait_time (float): Time fraction to wait before taking the next measurement [0,1]
            next_position (float): The next position in days after taking this action
            stop (bool): Whether the agent chooses to stop at this position
        
        Returns:
            float: The combined reward for the current action
        """
        perfect_stop = (
            10 if stop and 
            next_position *0.95 <= self._get_relapse_point() <= next_position * 1.05 
            else 0
        )
        
        wait_time_reward = 2*wait_time**2
        
        return wait_time_reward + self._closeness_reward(next_position) + perfect_stop
    
    def _closeness_reward(self, position):
        """
        Calculates a proximity-based reward relative to the relapse point.
        
        Uses an asymmetric Gaussian-like function that gives higher rewards
        when the position is close to the relapse point. The function has 
        different parameters before vs. after the relapse point, with a 
        steeper penalty for measurements taken after the relapse.
        
        Args:
            position (float): The current position in days to evaluate
        
        Returns:
            float: A reward value that peaks near the relapse point and
                  decreases as the position gets farther from it
        """
        rp = self._get_relapse_point()
        before_relapse = rp-position > 0
        dpx = position - rp # distance proxy to relapse
        m,var,s = (5.5,0.001,0.5) if before_relapse else (5,0.1,0.1)
        falloff = s if before_relapse else s*dpx
        return m*np.exp(-var*dpx**2) - falloff
        
        
    def _get_relapse_point(self):
        """
        Returns the x coordinate of the relapse point (days)
        
        Returns:
            float: The x coordinate of the relapse point.
        """
        return np.log(self.relapse_y - self.measurement) / self.target_rate
    
class ValueScheduler:
    """
    Linear schedule generator for arbitrary parameters.
    Linearly increases/decreases the value from start 
    to target over steps. Returns "target" after 
    "max_steps" is reached.
    """
    def __init__(self, 
                 start: float, 
                 target: float, 
                 max_steps: int,
                 validate = False):
        self.start = start
        self.target = target
        self.max_steps = max_steps
        self.step = 0
        self._value = start
        self.delta = (target - start) / max_steps
        self.validate = validate
        
    def next(self):
        if self.validate:
            return self.target
        if self.step < self.max_steps:
            self.step += 1
            self._value = self.start + self.delta * self.step
            return self._value
        return self.target
    
    @property
    def value(self):
        if self.validate:
            return self.target
        return self._value

if __name__ == "__main__":
    env = RelapseEnv()
    env.reset()
    done = False
    while not done:
        state, reward, done, _, _ = env.step(torch.tensor([0.5,0.0]))
        print(f"State: {state}, Reward: {reward}, Done: {done}")