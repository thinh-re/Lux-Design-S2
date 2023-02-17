from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.observations import Observation


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
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
        self.observation_space = spaces.Box(-9999, 9999, shape=(13936,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

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
        observation = dict()
        
        if observation_obj.player_0.real_env_steps > 10:
            pass
        
        # shared_obs = obs["player_0"]
        # ice_map = shared_obs["board"]["ice"]
        # ice_tile_locations = np.argwhere(ice_map == 1)

        # for agent in obs.keys():
        #     obs_vec = np.zeros(
        #         13,
        #     )

        #     factories = shared_obs["factories"][agent]
        #     factory_vec = np.zeros(2)
        #     for k in factories.keys():
        #         # here we track a normalized position of the first friendly factory
        #         factory = factories[k]
        #         factory_vec = np.array(factory["pos"]) / env_cfg.map_size
        #         break
        #     units = shared_obs["units"][agent]
        #     for k in units.keys():
        #         unit = units[k]

        #         # store cargo+power values scaled to [0, 1]
        #         cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
        #         battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
        #         cargo_vec = np.array(
        #             [
        #                 unit["power"] / battery_cap,
        #                 unit["cargo"]["ice"] / cargo_space,
        #                 unit["cargo"]["ore"] / cargo_space,
        #                 unit["cargo"]["water"] / cargo_space,
        #                 unit["cargo"]["metal"] / cargo_space,
        #             ]
        #         )
        #         unit_type = (
        #             0 if unit["unit_type"] == "LIGHT" else 1
        #         )  # note that build actions use 0 to encode Light
        #         # normalize the unit position
        #         pos = np.array(unit["pos"]) / env_cfg.map_size
        #         unit_vec = np.concatenate(
        #             [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
        #         )

        #         # we add some engineered features down here
        #         # compute closest ice tile
        #         ice_tile_distances = np.mean(
        #             (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
        #         )
        #         # normalize the ice tile location
        #         closest_ice_tile = (
        #             ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
        #         )
        #         obs_vec = np.concatenate(
        #             [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
        #         )
        #         break
        #     observation[agent] = obs_vec
            
        # since both players observe the same game state
        observation_player = observation_obj.player_0
        for agent in obs.keys():
            board = observation_player.board.numpy()
            factories = observation_player.factories.numpy(agent)
            units = observation_player.units.numpy(agent)
            observation[agent] = np.concatenate([
                board.reshape(-1), factories.reshape(-1), units.reshape(-1)
            ], axis=0)

        return observation
