import copy
from typing import Dict, Tuple

import gym
import numpy as np

from luxai_s2.state import StatsStateDict

from .obs_wrappers import SimpleUnitObservationWrapper


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment 
        into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.env: SimpleUnitObservationWrapper

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """

        Args:
            action (int): 

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: obs, reward, done, info
            
            Example:
            (
                array([0.66666667, 0.3125    , 1.        , 0.17      , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        , 0.02083333, 0.04166667]), # observation
                0, # no reward
                False, # game not done
                {'metrics': {'ice_dug': 0, 'water_produced': 0, 'action_queue_updates_success': 0, 'action_queue_updates_total': 0}} # more info
            )
                
        """
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        # this will call SimpleUnitDiscreteController.action_to_lux_action and then LuxAI_S2.step()
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs
