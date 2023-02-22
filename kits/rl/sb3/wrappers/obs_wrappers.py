from typing import Any, Dict, List

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.observations import Factory, Observation, Board, Player, Unit
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
        self.env_cfg: EnvConfig = env.env_cfg
        # np.product(Board.numpy_shape)
        self.observation_size = OurEnvConfig.MAX_FACTORIES_IN_OBSERVATION * Factory.numpy_shape * 2 \
            + OurEnvConfig.MAX_UNITS_IN_OBSERVATION * Unit.numpy_shape(OurEnvConfig.MAX_ACTIONS_PER_UNIT_IN_OBSERVATION) * 2 \
            + Player.real_env_steps_numpy_shape(self.env_cfg)
        self.observation_space = spaces.Box(
            -9999, 9999, shape=(self.observation_size,)
        )

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return ObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: EnvConfig) -> Dict[str, npt.NDArray]:
        """Convert raw observation into numerical observation for training
        Assumptions:
            1. We are player_0
            2. Player player_1 is kept alive with no actions

        Args:
            obs (Dict[str, Any]): raw observation
            env_cfg (EnvConfig): environment config

        Returns:
            Dict[str, npt.NDArray]: Dict of each player's observation
                Ex:
                {
                    'player_0': array([...]),
                    'player_1': array([...]),
                }
        """
        observation_obj = Observation(obs, env_cfg)
        mapped_observation = dict()
        
        # since both players observe the same game state
        observation_player = observation_obj.player_0
        
        ice_map = observation_player.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        
        ore_map = observation_player.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)
        
        # Assume we are "player_0"
        our_factory_map = np.array([factory.pos for factory in observation_obj.player_0.factories.player_0_factories])
        opponent_factory_map = np.array([factory.pos for factory in observation_obj.player_0.factories.player_1_factories])
        
        real_env_steps_numpy = observation_player.real_env_steps_numpy()
        
        for agent in obs.keys():
            # board = observation_player.board.numpy()
            factories = observation_player.factories.numpy(
                agent, max_factories=OurEnvConfig.MAX_FACTORIES_IN_OBSERVATION
            )
            units = observation_player.units.numpy(
                agent, max_units=OurEnvConfig.MAX_UNITS_IN_OBSERVATION,
                max_actions=OurEnvConfig.MAX_ACTIONS_PER_UNIT_IN_OBSERVATION,
                ice_map=ice_tile_locations, 
                ore_map=ore_tile_locations,
                our_factory_map=our_factory_map,
                opponent_factory_map=opponent_factory_map,
                env_config=env_cfg,
            )
            mapped_observation[agent] = np.concatenate([
                real_env_steps_numpy,
                # board.reshape(-1), 
                factories.reshape(-1),
                units.reshape(-1)
            ], axis=0)
        return mapped_observation
