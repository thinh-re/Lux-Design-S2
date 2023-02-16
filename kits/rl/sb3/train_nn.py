import torch as th
import torch.nn as nn
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Note: Copy to nn.py before submit!
# TODO: Create share sub-module that includes custom network
# and import it in here and nn
class CustomNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 12):
        super().__init__(observation_space, features_dim)
        self.action_dims = features_dim
        self.mlp = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(128, self.action_dims),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.mlp(x)
        return x
