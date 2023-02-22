import copy
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from lux.team import Team
from wrappers.actions import Action, CheckUselessActionInput
from wrappers.controllers_wrapper import ControllerWrapper
from wrappers.obs_wrappers import ObservationWrapper
from wrappers.observations import Observation, Player, State, Unit, Units

from luxai_s2.state.state import State as GameState


def find_unit(agent: str, key: str, units: List[Unit]) -> Unit:
    for u in units:
        if u.unit_id == key:
            return u
    raise ValueError(f"No unit with id {key} found for agent {agent}")


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, controller_wrapper: ControllerWrapper) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment
        into a single-agent environment for easy training
        """
        super().__init__(env)
        self.controller_wrapper = controller_wrapper
        self.prev_step_metrics = None
        self.env: ObservationWrapper
        self.env_cfg = self.env.env_cfg

        self.max_states: int = 2
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

        prev_game_state: GameState = self.env.state
        self.__opponent(prev_game_state, opp_agent)

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        # this will call SimpleUnitDiscreteController.action_to_lux_action and then LuxAI_S2.step()
        obs, _, done, _ = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        current_game_state: GameState = self.env.state

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        current_state = State(current_game_state.stats[agent])

        self.__reset_game_states(current_game_state)

        # self.__check_useless_actions(action, current_state)

        if done:
            reward = 0.0
            return obs, reward, done, self.__info(current_state)

        prev_state: Optional[State] = (
            self.prev_states[-1] if len(self.prev_states) > 0 else None
        )
        team: Team = current_game_state.teams[agent]
        reward = self.reward_function(
            agent,
            current_game_state,
            current_state,
            prev_state,
            team.factories_to_place,
            action[agent],
        )

        self.__backup_game_states(current_game_state, current_state)
        return obs, reward, done, self.__info(current_state)

    def __useless_actions_reward(
        self,
        agent: str,
        action: np.ndarray,
        current_game_state: GameState,
        state: State,
    ) -> float:
        # TODO: improve it. This is not efficient to convert actions to lux actions again
        raw_obs = {
            "player_0": current_game_state.get_obs(),
            "player_1": current_game_state.get_obs(),
        }
        obs_obj = Observation(raw_obs, self.env_cfg)
        actions: Dict[str, Any] = self.controller_wrapper.action_to_lux_action(
            agent, raw_obs, action
        )
        units = obs_obj.player_0.units.get_units_of_agent(agent)
        factories = obs_obj.player_0.factories.get_factories_of_agent(agent)
        rewards = 0.0
        for key, value in actions.items():
            if key.startswith("unit"):
                unit = find_unit(agent, key, units)

                check_useless_action = CheckUselessActionInput(
                    unit, value, state, units, factories
                )

                if self.__is_useless_action(check_useless_action):
                    rewards -= 1e-4
        return rewards

    def __is_useless_action(self, input: CheckUselessActionInput) -> bool:
        for action in self.controller_wrapper.actions:
            action: Action
            useless, msg = action.is_useless_action(input)
            if useless:
                # print("useless action", msg)
                return useless
        return False

    def __backup_game_states(
        self, current_game_state: GameState, current_state: GameState
    ) -> None:
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

    def __consumption_state(self, s: Optional[State]) -> np.ndarray:
        if s is not None:
            return np.array(
                [
                    # Ice
                    s.consumption.ice.HEAVY + s.consumption.ice.LIGHT,
                    # Metal
                    s.consumption.metal,
                    # Ore
                    s.consumption.ore.HEAVY + s.consumption.ore.LIGHT,
                    # Power
                    s.consumption.power.HEAVY
                    + s.consumption.power.LIGHT
                    + s.consumption.power.FACTORY,
                ]
            )
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])

    def __info(self, current_state: State) -> Dict[str, float]:
        info = dict()
        metrics = dict()
        metrics["produced_ice"] = (
            current_state.generation.ice.HEAVY + current_state.generation.ice.LIGHT
        )
        metrics["produced_water"] = current_state.generation.water
        metrics["produced_ore"] = (
            current_state.generation.ore.HEAVY + current_state.generation.ore.LIGHT
        )

        metrics["robots_light"] = current_state.generation.build.LIGHT
        metrics["robots_heavy"] = current_state.generation.build.HEAVY

        metrics["destroyed_factories"] = current_state.destroyed.FACTORY
        metrics["destroyed_heavy_robots"] = current_state.destroyed.HEAVY
        metrics["destroyed_light_robots"] = current_state.destroyed.LIGHT

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics[
            "action_queue_updates_success"
        ] = current_state.action_queue_updates_success
        metrics["action_queue_updates_total"] = current_state.action_queue_updates_total

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics
        return info

    def __consumption_reward(self, current_state: State, prev_state: State) -> float:
        current_state_np = self.__consumption_state(current_state)
        prev_state_np = self.__consumption_state(prev_state)
        return np.sum(prev_state_np - current_state_np) / 100

    def __destroyed_state(self, s: Optional[State]) -> np.ndarray:
        if s is not None:
            return np.array(
                [
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
                ]
            )
        else:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def __destroyed_reward(
        self,
        current_state: State,
        prev_state: Optional[State],
        factories_to_place: int,
    ) -> float:
        current_state_np = self.__destroyed_state(current_state)
        prev_state_np = self.__destroyed_state(prev_state)
        ratio = np.array(
            [
                0,  # 200 / factories_to_place, # factories
                1.0,  # 6, # heavy robots
                0.5,  # light robots
                0.0,  # lichen
                0,  # rubble
            ]
        )
        return np.sum((prev_state_np - current_state_np) * ratio)

    def __generation_state(self, s: Optional[State]) -> np.ndarray:
        if s is not None:
            return np.array(
                [
                    # Ice
                    s.generation.ice.HEAVY + s.generation.ice.LIGHT,
                    # Metal
                    s.generation.metal,
                    # Ore
                    s.generation.ore.HEAVY + s.generation.ore.LIGHT,
                    # Power
                    s.generation.power.HEAVY
                    + s.generation.power.LIGHT
                    + s.generation.power.FACTORY,
                    # Lichen
                    s.generation.lichen,
                    # Water
                    s.generation.water,
                    # Heavy robots
                    s.generation.build.HEAVY,
                    # Light robots
                    s.generation.build.LIGHT,
                ]
            )
        else:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __spawning_robots_reward(self, agent: str, game_state: GameState) -> float:
        if len(game_state.units[agent].keys()) == 0:
            return -0.1
        else:
            return 0.0

    def __generation_reward(
        self,
        current_state: State,
        prev_state: Optional[State],
    ) -> float:
        current_state_np = self.__generation_state(current_state)
        prev_state_np = self.__generation_state(prev_state)
        ratio = np.array(
            [
                0.01,  # ice
                0.01,  # metal
                0.01,  # ore
                0.0,  # power
                0.0,  # lichen
                1.0,  # water
                0.0,  # heavy robots
                0.0,  # light robots
            ]
        )
        return np.sum((current_state_np - prev_state_np) * ratio)

    def __updates_reward(
        self, current_state: State, prev_state: Optional[State]
    ) -> float:
        """Encourage successful updates, penalizing failed updates

        Args:
            current_state (State): current state
            prev_state (Optional[State]): previous state before updating

        Returns:
            float: reward
        """
        if prev_state is None:
            new_updates = current_state.action_queue_updates_total
            successful_updates = current_state.action_queue_updates_success
        else:
            new_updates = (
                current_state.action_queue_updates_total
                - prev_state.action_queue_updates_total
            )
            successful_updates = (
                current_state.action_queue_updates_success
                - prev_state.action_queue_updates_success
            )
        failed_updates = new_updates - successful_updates
        return -failed_updates * 1e-5

    def reward_function(
        self,
        agent: str,
        current_game_state: GameState,
        current_state: State,
        prev_state: Optional[State],
        factories_to_place: int,
        action: np.ndarray,
    ) -> float:
        # consumption_reward = self.__consumption_reward(current_state, prev_state)
        destroyed_reward = self.__destroyed_reward(
            current_state, prev_state, factories_to_place
        )
        generation_reward = self.__generation_reward(current_state, prev_state)
        updates_reward = self.__updates_reward(current_state, prev_state)
        spawning_robots_reward = self.__spawning_robots_reward(
            agent, current_game_state
        )
        useless_actions_reward = self.__useless_actions_reward(
            agent, action, current_game_state, current_state
        )

        rewards = (
            sum(
                [
                    # consumption_reward,
                    destroyed_reward,
                    generation_reward,
                    updates_reward,
                    useless_actions_reward
                    # TODO: rewards for pickup, transfer
                ]
            )
            + 1e-5
            + spawning_robots_reward
        )
        # print(
        #     current_game_state.real_env_steps,
        #     ":",
        #     destroyed_reward,
        #     updates_reward,
        #     useless_actions_reward,
        #     generation_reward,
        #     rewards,
        # )
        return rewards

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs
