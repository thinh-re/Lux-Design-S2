from typing import Tuple, Union

import gym
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wrappers import ControllerWrapper, CustomEnvWrapper, ObservationWrapper
from wrappers.sb3_wrapper import SB3Wrapper

from luxai_s2.env import LuxAI_S2
from luxai_s2.utils.heuristics.factory_placement import (
    place_near_random_ice, random_factory_placement)


def make_env(
    env_id: str, 
    rank: int, 
    seed: int = 0, 
    max_episode_steps=100,
    returns_controller_observation=False,
    is_random_policy=False,
):
    def _init() -> Union[
        LuxAI_S2, 
        Tuple[LuxAI_S2, ControllerWrapper, ObservationWrapper],
    ]:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 2 for simplification and keeping returns consistent 
        # as we survive longer if there are more initial resources
        env: LuxAI_S2 = gym.make(
            env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2
        )

        controller_wrapper = ControllerWrapper(env.env_cfg)
        
        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.
        env = SB3Wrapper(
            env,
            factory_placement_policy=place_near_random_ice if not is_random_policy else random_factory_placement,
            controller=controller_wrapper,
        )
        observation_wrapper = ObservationWrapper(env)  # changes observation to include a few simple features
        env = CustomEnvWrapper(observation_wrapper)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        
        if returns_controller_observation:
            return env, controller_wrapper, observation_wrapper
        else:
            return env

    return _init
