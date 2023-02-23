from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from lux.config import EnvConfig
from lux.kit import GameState
from wrappers.observations import Factory, Unit

FACTORY_AREA = [
    np.array([0, 0]),
    np.array([0, -1]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, -1]),
    np.array([1, 1]),
    np.array([-1, 0]),
    np.array([-1, -1]),
    np.array([-1, 1]),
]


def is_elm_in_array(arr: np.ndarray, elm: np.ndarray) -> bool:
    return (arr == elm).all(1).any()


def get_factories_pos(factories: List[Factory]) -> np.ndarray:
    factories_pos_lst: List[np.ndarray] = []
    for factory in factories:
        factories_pos_lst.extend([factory.pos + p for p in FACTORY_AREA])
    return np.array([factories_pos_lst])


class CheckUselessActionInput:
    def __init__(
        self,
        unit: Unit,
        action: np.ndarray,
        game_state: GameState,
        units: List[Unit],
        factories: List[Factory],
    ) -> None:
        self.unit = unit
        self.action = action
        self.game_state = game_state
        self.units = units  # does not include this `unit`
        self.factories = factories


class Action:
    def __init__(
        self,
        action_type_id: int,
        dim: int,
        env_cfg: EnvConfig,
    ):
        """
        Abstract class Action for all action types

        Note: Currently not consider [..., repeat, n]
        so [..., 0, 1]

        Args:
            dim (int): _description_
        """
        self.dim = dim
        self.env_cfg = env_cfg
        self.start_id: int = -1
        self.end_id: int = -1
        self.action_type_id = action_type_id

    def update_id_range(self, start_id: int) -> None:
        self.start_id = start_id
        self.end_id = start_id + self.dim - 1

    def get_action(self, id: int) -> Optional[np.ndarray]:
        if id > self.end_id or id < self.start_id:
            return None
        else:
            return self._get_action(id - self.start_id)

    @abstractmethod
    def _get_action(self, id: int) -> np.ndarray:
        """Convert each action type to numpy action

        Args:
            id (int): Always relative to start_id

        Returns:
            np.ndarray: action numpy
                [action_type, direction, resource_type, amount, repeat, n]
        """
        raise NotImplemented()

    def is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        """Check whether the action is useless
        Like transfer to somewhere not our factorys or robots

        Args:
            unit (Unit): which unit will take this action
            action (np.ndarray): [action_type, direction, resource_type, amount, repeat, n]
            game_state (GameState): previous game state before taking action

        Returns:
            bool: True if the action is useless, False otherwise
        """
        if input.action[0][0] != self.action_type_id:
            return False, ""

        return self._is_useless_action(input)

    @abstractmethod
    def _is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        return False, ""


class MoveAction(Action):
    def __init__(self, env_cfg: EnvConfig):
        super().__init__(action_type_id=0, dim=4, env_cfg=env_cfg)
        # 4 directions: up, right, down, left (ignore move center)

    def _get_action(self, id: int) -> np.ndarray:
        return np.array(
            [
                self.action_type_id,  # action_type
                id + 1,  # direction
                0,  # resource_type
                0,  # amount
                0,  # repeat
                1,  # n
            ]
        )

    def _is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        """Useless actions are:
        - Move when power not enough

        Note: We also need to care about
        whether the results of moving action turn out
        to be good or bad:
        1. Good: kill other enemy's robot
        2. Bad: kill our robot
        """
        if input.unit.power < self.env_cfg.ROBOTS[input.unit.unit_type].MOVE_COST:
            return True, "Move when power not enough"
        else:
            return False, ""


class TransferAction(Action):
    def __init__(self, env_cfg: EnvConfig):
        super().__init__(action_type_id=1, dim=5 * 2, env_cfg=env_cfg)
        # ore(5), ice(5)
        self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

    def _get_action(self, id: int) -> np.ndarray:
        # [1, direction, resource_type, self.env_cfg.max_transfer_amount, 0, 1]
        return np.array(
            [
                self.action_type_id,  # action_type
                id % 5,  # direction
                id // 5,  # resource_type
                self.env_cfg.max_transfer_amount,  # amount
                0,  # repeat
                1,  # n
            ]
        )

    def _is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        """Useless actions are:
        - Transfer when no resource is available
        - Transfer to the cell where no factories or robots are available
        """
        if input.action[0][2] == 0:
            # Transfer ice
            if input.unit.cargo.ice == 0:
                return True, "Transfer ice but there is no ice available"
            else:
                return False, ""

        if input.action[0][2] == 1:
            # Transfer ore
            if input.unit.cargo.ore == 0:
                return True, "Transfer ore but there is no ore available"
            else:
                return False, ""

        direction = self.move_deltas[input.action[0][1]]
        destination = input.unit.pos + direction

        units_pos = np.array(
            [unit.pos for unit in input.units if unit.pos != input.unit.pos]
        )
        factories_pos = get_factories_pos(input.factories)

        if not (
            is_elm_in_array(units_pos, destination)
            or is_elm_in_array(factories_pos, destination)
        ):
            return (
                True,
                "Transfer to the cell where no factories or robots are available",
            )
        else:
            return False, ""


class PickupAction(Action):
    def __init__(self, env_cfg: EnvConfig):
        super().__init__(action_type_id=2, dim=1, env_cfg=env_cfg)

    def _get_action(self, id: int) -> np.ndarray:
        # Only pickup power
        # [2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1]
        return np.array(
            [
                self.action_type_id,  # action_type
                0,  # direction
                4,  # resource_type
                self.env_cfg.max_transfer_amount,  # amount
                0,  # repeat
                1,  # n
            ]
        )

    def _is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        """Useless actions are:
        - Pickup power when not in factories location
        """
        factories_pos = get_factories_pos(input.factories)

        if not is_elm_in_array(factories_pos, input.unit.pos):
            return True, "Pickup power when not in factories location"
        else:
            return False, ""


class DigAction(Action):
    def __init__(self, env_cfg: EnvConfig):
        super().__init__(action_type_id=3, dim=1, env_cfg=env_cfg)

    def _get_action(self, id: int) -> np.ndarray:
        # Only pickup power
        # [3, 0, 0, 0, 0, 1]
        return np.array(
            [
                self.action_type_id,  # action_type
                0,  # direction
                0,  # resource_type
                0,  # amount
                0,  # repeat
                1,  # n
            ]
        )

    def _is_useless_action(
        self,
        input: CheckUselessActionInput,
    ) -> Tuple[bool, str]:
        """Useless actions are:
        -
        """
        return False, ""
