"""
Code for neural network inference and loading SB3 model weights
"""
import io
import sys
import zipfile
from typing import Dict

import torch as th
from gym import spaces
from torch import Tensor, nn
from wrappers.obs_wrappers import ObservationWrapper
from wrappers.observations import Board
from wrappers.controllers_wrapper import ControllerWrapper

class Net(nn.Module):
    def __init__(
        self, 
        observation_space: spaces.Box, # SimpleUnitObservationWrapper.observation_space
        action_space: spaces.MultiDiscrete, # SimpleUnitDiscreteController.action_space
        features_dim: int = 128,
    ):
        super(Net, self).__init__()
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
        # print('No. parameters:', count_parameters(self))
        
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, self.action_dims),
        )

    def act(self, x: Tensor, action_masks: Tensor, deterministic: bool = False):
        latent_pi = self.forward(x)
        action_logits: Tensor = self.action_net(latent_pi)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = th.distributions.Categorical(logits=action_logits)
        if not deterministic:
            return dist.sample()
        else:
            return dist.mode

    def forward(self, x: th.Tensor) -> th.Tensor:
        board = x[:, :self.board_region].reshape((-1, self.c, self.h, self.w))
        board = self.linear(self.cnn(board))

        others = x[:, self.board_region:]
        others = self.mlp(others)
        
        rs = th.cat([board, others], axis=1)
        rs = self.final(rs)
        return rs

def load_policy(model_path: str, observation_wrapper: ObservationWrapper, controller_wrapper: ControllerWrapper) -> Net:
    # load .pth or .zip
    if model_path[-4:] == ".zip":
        with zipfile.ZipFile(model_path) as archive:
            file_path = "policy.pth"
            with archive.open(file_path, mode="r") as param_file:
                file_content = io.BytesIO()
                file_content.write(param_file.read())
                file_content.seek(0)
                sb3_state_dict: Dict = th.load(file_content, map_location="cpu")
    else:
        sb3_state_dict: Dict = th.load(model_path, map_location="cpu")

    model = Net(
        observation_space=observation_wrapper.observation_space,
        action_space=controller_wrapper.action_space,
    )
    loaded_state_dict = {}

    # this code here works assuming the first keys in the sb3 state dict 
    # are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), model.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)

    model.load_state_dict(loaded_state_dict)
    return model
