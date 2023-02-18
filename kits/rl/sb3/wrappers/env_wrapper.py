import copy
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from wrappers.obs_wrappers import ObservationWrapper
from wrappers.observations import State
from lux.team import Team

from luxai_s2.state.state import State as GameState


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment 
        into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.env: ObservationWrapper
        
        self.max_states: int = 5
        self.prev_states: List[State] = []
        self.prev_game_states: List[GameState] = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """

        Args:
            action (np.ndarray): see SimpleUnitDiscreteController.action_space 

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
        
        current_game_state: GameState = self.env.state

        self.__opponent(current_game_state, opp_agent)

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        # this will call SimpleUnitDiscreteController.action_to_lux_action and then LuxAI_S2.step()
        obs, _, done, _ = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        current_state = State(current_game_state.stats[agent])
        
        self.__reset_game_states(current_game_state)

        if len(self.prev_states) > 0:
            prev_state = self.prev_states[-1]
            team: Team = current_game_state.teams[agent]
            reward = self.reward_function(
                current_game_state,
                current_state, 
                prev_state, 
                team.factories_to_place,
            )
        else:
            reward = 0
        
        self.__backup_game_states(current_game_state, current_state)
        return obs, reward, done, self.__info(current_state)
    
    def __backup_game_states(self, current_game_state: GameState, current_state: GameState) -> None:
        self.prev_states.append(copy.deepcopy(current_state))
        self.prev_game_states.append(copy.deepcopy(current_game_state))
        if len(self.prev_states) > self.max_states:
            self.prev_states.pop(0)
            self.prev_game_states.pop(0)
            
    def __reset_game_states(self, current_game_state: GameState) -> None:
        if current_game_state.real_env_steps == 1:
            self.prev_game_states = []
            self.prev_states = []
    
    def __opponent(self, current_game_state: GameState, opp_agent: str) -> None:
        # TODO: if passing opponent agent, ignore these code
        opp_factories = current_game_state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000
    
    def __consumption_state(self, s: State) -> np.ndarray:
        return np.array([
            # Ice
            s.consumption.ice.HEAVY + s.consumption.ice.LIGHT,
            
            # Metal
            s.consumption.metal,
            
            # Ore
            s.consumption.ore.HEAVY + s.consumption.ore.LIGHT,
            
            # Power
            s.consumption.power.HEAVY + \
                s.consumption.power.LIGHT + \
                s.consumption.power.FACTORY,
        ])
        
    def __info(self, current_state: State) -> Dict[str, float]:
        info = dict()
        metrics = dict()
        metrics["ice_dug"] = current_state.generation.ice.HEAVY + current_state.generation.ice.LIGHT
        metrics["water_produced"] = current_state.generation.water

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = current_state.action_queue_updates_success
        metrics["action_queue_updates_total"] = current_state.action_queue_updates_total

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics
        return info
    
    def __consumption_reward(self, current_state: State, prev_state: State) -> float:
        current_state_np = self.__consumption_state(current_state)
        prev_state_np = self.__consumption_state(prev_state)
        return np.sum(prev_state_np - current_state_np) / 100
    
    def __destroyed_state(self, s: State) -> np.ndarray:
        return np.array([
            # Factory
            s.destroyed.FACTORY,
            
            # Heavy robots
            s.destroyed.HEAVY,
            
            # Light robots
            s.destroyed.LIGHT,
            
            # Lichen
            s.destroyed.lichen.HEAVY + s.destroyed.lichen.LIGHT,
            
            # Rubble
            s.destroyed.rubble.HEAVY + s.destroyed.rubble.LIGHT,
        ])
        
    def __destroyed_reward(
        self, 
        current_state: State, 
        prev_state: State,
        factories_to_place: int,
    ) -> float:
        current_state_np = self.__destroyed_state(current_state)
        prev_state_np = self.__destroyed_state(prev_state)
        ratio = np.array([
            100 / factories_to_place, # factories
            5, # heavy robots
            1, # light robots
            0.02, # lichen
            0, # rubble
        ])
        return np.sum((prev_state_np - current_state_np) * ratio)
    
    def __generation_state(self, s: State) -> np.ndarray:
        return np.array([
            # Ice
            s.generation.ice.HEAVY + s.generation.ice.LIGHT,
            
            # Metal
            s.generation.metal,
            
            # Ore
            s.generation.ore.HEAVY + s.generation.ore.LIGHT,
            
            # Power
            s.generation.power.HEAVY + \
                s.generation.power.LIGHT + \
                s.generation.power.FACTORY,
            
            # Lichen
            s.generation.lichen,
            
            # Water
            s.generation.water,
            
            # Heavy robots
            s.generation.build.HEAVY,
            
            # Light robots
            s.generation.build.LIGHT,
        ])

    def __generation_reward(
        self, 
        current_state: State, 
        prev_state: State,
    ) -> float:
        current_state_np = self.__generation_state(current_state)
        prev_state_np = self.__generation_state(prev_state)
        ratio = np.array([
            0.02,
            0.02,
            0.02,
            0.02,
            0.03,
            0.03,
            5.5,
            1.5,
        ])
        return np.sum((current_state_np - prev_state_np) * ratio)

    
    def reward_function(
        self, 
        current_game_state: GameState,
        current_state: State, 
        prev_state: State, 
        factories_to_place: int,
    ) -> float:
        consumption_reward = self.__consumption_reward(current_state, prev_state)
        destroyed_reward = self.__destroyed_reward(current_state, prev_state, factories_to_place)
        generation_reward = self.__generation_reward(current_state, prev_state)
        # if generation_reward < 0:
        #     pass

        rewards = sum([
            consumption_reward,
            destroyed_reward,
            generation_reward,
            # TODO: rewards for pickup, transfer
        ])
        # print(current_game_state.real_env_steps, ':', consumption_reward, destroyed_reward, generation_reward, rewards)
        return rewards
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs
