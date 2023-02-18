from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.observations import Factory, Observation, Board, Unit
from config import OurEnvConfig


class ObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_size = np.product(Board.numpy_shape) \
            + OurEnvConfig.MAX_FACTORIES_IN_OBSERVATION * Factory.numpy_shape * 2 \
            + OurEnvConfig.MAX_UNITS_IN_OBSERVATION * Unit.numpy_shape(OurEnvConfig.MAX_ACTIONS_PER_UNIT_IN_OBSERVATION) * 2
        self.observation_space = spaces.Box(
            -9999, 9999, shape=(self.observation_size,)
        )

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return ObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: EnvConfig) -> Dict[str, npt.NDArray]:
        '''
        Returns 
        {
            'player_0': array([...]),
            'player_1': array([...]),
        }
        '''
        observation_obj = Observation(obs)
        mapped_observation = dict()
        
        # since both players observe the same game state
        observation_player = observation_obj.player_0
        for agent in obs.keys():
            board = observation_player.board.numpy()
            factories = observation_player.factories.numpy(
                agent, max_factories=OurEnvConfig.MAX_FACTORIES_IN_OBSERVATION
            )
            units = observation_player.units.numpy(
                agent, max_units=OurEnvConfig.MAX_UNITS_IN_OBSERVATION,
                max_actions=OurEnvConfig.MAX_ACTIONS_PER_UNIT_IN_OBSERVATION,
            )
            mapped_observation[agent] = np.concatenate([
                board.reshape(-1), factories.reshape(-1), units.reshape(-1)
            ], axis=0)

        return mapped_observation
