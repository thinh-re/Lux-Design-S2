from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import count_parameters
from wrappers.observations import Board


# Note: Copy to nn.py before submit!
# TODO: Create share sub-module that includes custom network
# and import it in here and nn
class CustomNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, 
        observation_space: spaces.Box, # SimpleUnitObservationWrapper.observation_space
        action_space: spaces.MultiDiscrete, # SimpleUnitDiscreteController.action_space
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.features_dim = features_dim
        self.action_dims: int = action_space.shape[0]
        self.observation_space_shape: int = observation_space.shape[0]
        
        # Board
        self.c, self.h, self.w = Board.numpy_shape # (6, 48, 48)
        self.board_region: int = self.c * self.h * self.w
        self.cnn = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                # th.as_tensor(observation_space.sample()[None]).float()
                th.randn((1, self.c, self.h, self.w))
            ).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU(),
        )
        
        self.n_others = self.observation_space_shape - self.board_region
        self.mlp = nn.Sequential(
            nn.Linear(self.n_others, features_dim),
            nn.Tanh(),
            nn.Linear(features_dim, features_dim),
            nn.Tanh(),
        )
        
        self.final = nn.Sequential(
            nn.Linear(features_dim*2, features_dim),
            nn.Tanh(),
        )
        print('No. parameters:', count_parameters(self))
    

    def forward(self, x: th.Tensor) -> th.Tensor:
        board = x[:, :self.board_region].reshape((-1, self.c, self.h, self.w))
        board = self.linear(self.cnn(board))

        others = x[:, self.board_region:]
        others = self.mlp(others)
        
        rs = th.cat([board, others], axis=1)
        rs = self.final(rs)
        return rs

# TODO:
# ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Dict, features_dim: int = 12):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super().__init__(observation_space, features_dim=1)

#         self.action_dims = features_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(observation_space.shape[0], 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#         )
#         self.action_net = nn.Sequential(
#             nn.Linear(128, self.action_dims),
#         )

#     def forward(self, observations: Dict) -> th.Tensor:
        
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)
