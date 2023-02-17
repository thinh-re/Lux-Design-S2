from typing import Dict
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

# TODO:
# ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 12):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

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

    def forward(self, observations: Dict) -> th.Tensor:
        
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
