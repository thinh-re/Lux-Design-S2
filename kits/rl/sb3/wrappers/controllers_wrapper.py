from typing import Any, Dict, List

import gym
import numpy as np
import numpy.typing as npt
from config import OurEnvConfig
from gym import spaces
from lux.config import EnvConfig
from lux.factory import Factory
from lux.kit import GameState, obs_to_game_state
from wrappers.actions import Action, DigAction, MoveAction, PickupAction, TransferAction
from wrappers.observations import Unit


def get_valid_actions_of_factory(
    factory: Factory,
    game_state: GameState,
    env_cfg: EnvConfig,
) -> List[int]:
    valid_actions: List[int] = []

    # is there enough resource to build LIGHT robot
    if (
        factory.cargo.metal >= env_cfg.ROBOTS["LIGHT"].METAL_COST
        and factory.power >= env_cfg.ROBOTS["LIGHT"].POWER_COST
    ):
        valid_actions.append(0)

    # is there enough resource to build HEAVY robot
    if (
        factory.cargo.metal >= env_cfg.ROBOTS["HEAVY"].METAL_COST
        and factory.power >= env_cfg.ROBOTS["HEAVY"].POWER_COST
    ):
        valid_actions.append(1)

    # is it okay to water?
    # TODO:
    valid_actions.append(2)

    return valid_actions


def get_valid_actions_of_units(
    agent: str, unit: Unit, game_state: GameState, env_cfg: EnvConfig
) -> List[int]:
    valid_actions: List[int] = []


def get_valid_actions_of_agent(
    agent: str,
    game_state: GameState,
    env_cfg: EnvConfig,
) -> np.ndarray:
    valid_actions: List[List[int]] = []

    for _, factory in zip(
        range(OurEnvConfig.MAX_FACTORIES_IN_ACTION_SPACES),
        game_state.factories[agent].values(),
    ):
        valid_actions.append(
            get_valid_actions_of_factory(
                factory,
                game_state,
                env_cfg,
            )
        )

    for _, unit in zip(
        range(OurEnvConfig.MAX_ACTIONS_PER_UNIT_IN_ACTION_SPACES),
        game_state.units[agent].values(),
    ):
        pass


def mask_fn(env: gym.Env) -> np.ndarray:
    game_state: GameState = env.env.state
    env_cfg = env.env_cfg
    rs = np.array(
        [0, 0, 1, 0] * OurEnvConfig.MAX_FACTORIES_IN_ACTION_SPACES
        + [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
        * OurEnvConfig.MAX_UNITS_IN_ACTION_SPACES
    )
    return rs


# Controller class copied here since you won't have access to the luxai_s2
# package directly on the competition server
class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()


class ControllerWrapper(Controller):
    def __init__(self, env_cfg: EnvConfig) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer actions: each resource in 4 cardinal directions or center (5)*2
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/act_space.py

        """
        self.env_cfg = env_cfg
        self.actions: List[Action] = []
        i = 0
        for action_cls in [MoveAction, TransferAction, PickupAction, DigAction]:
            action: Action = action_cls(env_cfg)
            self.actions.append(action)
            action.update_id_range(i)
            i += action.dim
        self.total_actions_per_unit = (
            sum([action.dim for action in self.actions]) + 1
        )  # no op

        self.actions_per_factory = 4
        self.max_factories = OurEnvConfig.MAX_FACTORIES_IN_ACTION_SPACES
        self.max_units = OurEnvConfig.MAX_UNITS_IN_ACTION_SPACES
        action_space = spaces.MultiDiscrete(
            [self.actions_per_factory] * self.max_factories
            + [self.total_actions_per_unit] * self.max_units
        )  # shape = (n,)

        super().__init__(action_space)

    def action_to_lux_action(
        self,
        agent: str,
        obs: Dict[str, Any],
        action: npt.NDArray,
    ) -> Dict:
        """
        Returns: {'factory_0': 0 or 1 or 2}
        """
        game_state = obs_to_game_state(self.env_cfg, obs)
        lux_action = dict()
        units = game_state.units_lst(agent)
        for i, unit in enumerate(units[: self.max_units], start=self.max_factories):
            self.__unit_action(unit, action[i], lux_action)

        factories = game_state.factories_lst(agent)
        for i, factory in enumerate(factories[: self.max_factories], start=0):
            if action[i] == 3:
                continue
            lux_action[factory.unit_id] = action[i]

        return lux_action

    def __unit_action(
        self,
        unit: Unit,
        id: int,
        lux_action: Dict[str, Any],
    ) -> None:
        action_queue = []

        for action_ctl in self.actions:
            action = action_ctl.get_action(id)
            if action is not None:
                action_queue = [action]
                break

        if len(action_queue) > 0:
            lux_action[unit.unit_id] = action_queue

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Note: this function is not used anywhere!
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """
        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask
