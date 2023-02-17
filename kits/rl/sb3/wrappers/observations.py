from typing import Any, Dict, List, Optional, Tuple

import numpy as np

'''Convert observation dict to observation object
for the sake of debugging

observation_object = Observation(observation_dict)
'''

class PlayerTeam:
    def __init__(self, raw_player_team_obs: Dict[str, Any]) -> None:
        self.bid: int = raw_player_team_obs['bid']
        self.faction: str = raw_player_team_obs['faction']
        self.factories_to_place: int = raw_player_team_obs['factories_to_place']
        self.factory_strains: List[int] = raw_player_team_obs['factory_strains']
        self.metal: int = raw_player_team_obs['metal']
        self.place_first: bool = raw_player_team_obs['place_first']
        self.team_id: int = raw_player_team_obs['team_id']
        self.water: int = raw_player_team_obs['water']

class Team:
    def __init__(self, raw_teams_player_obs: Dict[str, Any]) -> None:
        self.player_0 = PlayerTeam(raw_teams_player_obs['player_0'])
        self.player_1 = PlayerTeam(raw_teams_player_obs['player_1'])

class Board:
    numpy_shape = (6, 48, 48)
    
    def __init__(self, raw_board_obs: Dict[str, Any]) -> None:
        self.factories_per_team: int = raw_board_obs['factories_per_team']
        
        self.ice: np.ndarray = raw_board_obs['ice']
        self.lichen: np.ndarray = raw_board_obs['lichen']
        self.lichen_strains: np.ndarray = raw_board_obs['lichen_strains']
        self.ore: np.ndarray = raw_board_obs['ore']
        self.rubble: np.ndarray = raw_board_obs['rubble']
        self.valid_spawns_mask: np.ndarray = raw_board_obs['valid_spawns_mask']
        
    def numpy(self) -> np.ndarray:
        """Merge 6 board together
        Return shape 6 x H x W
        """
        return np.stack([
            self.ice,
            self.lichen,
            self.lichen_strains,
            self.ore,
            self.rubble,
            self.valid_spawns_mask.astype(np.int64),
        ])

class Cargo:
    def __init__(self, raw_cargo_obs: Optional[Dict[str, Any]] = None) -> None:
        if raw_cargo_obs is None:
            self.ice: int = 0
            self.metal: int = 0
            self.ore: int = 0
            self.water: int = 0
        else:
            self.ice: int = raw_cargo_obs['ice']
            self.metal: int = raw_cargo_obs['metal']
            self.ore: int = raw_cargo_obs['ore']
            self.water: int = raw_cargo_obs['water']
    
    def numpy(self) -> np.array:
        return np.array([
            self.ice, self.metal, self.ore, self.water
        ])

class Factory:
    def __init__(self, raw_factory_obs: Optional[Dict[str, Any]] = None) -> None:
        if raw_factory_obs is None:
            self.cargo = Cargo()
            self.pos = np.array([-1, -1])
            self.power = 0
            self.strain_id = -1
            self.team_id = -1
            self.unit_id = "None"
        else:
            self.cargo = Cargo(raw_factory_obs['cargo'])
            self.pos: np.ndarray = raw_factory_obs['pos'] # array([10, 10])
            self.power: int = raw_factory_obs['power']
            self.strain_id: int = raw_factory_obs['strain_id']
            self.team_id: int = raw_factory_obs['team_id']
            self.unit_id: str = raw_factory_obs['unit_id']

    def numpy(self) -> np.ndarray:
        return np.concatenate([
            self.cargo.numpy(),
            self.pos,
            np.array([
                self.power,
                self.strain_id,
                # self.team_id,
            ])
        ], axis=0)
        
class Factories:
    def __init__(self, raw_factories_obs: Dict[str, Dict]) -> None:
        self.player_0_factories: List[Factory] = self.__convert_factoriesdict_to_factorieslist(
            raw_factories_obs['player_0']
        )
        self.player_1_factories: List[Factory] = self.__convert_factoriesdict_to_factorieslist(
            raw_factories_obs['player_1']
        )
            
    def get_factories_of_agent(self, agent: str) -> List[Factory]:
        if agent == 'player_0':
            return self.player_0_factories
        else:
            return self.player_1_factories
            
    def __convert_factoriesdict_to_factorieslist(self, factories_dict: Dict[str, Dict]) -> List[Factory]:
        lst: List[Factory] = []
        for factory in factories_dict.values():
            lst.append(Factory(factory))
        return lst
    
    def numpy(self, agent: str, max_factories: int = 2) -> np.ndarray:
        # The order depends on which agent
        if agent == 'player_0':
            return np.stack([
                self.factories_numpy(self.player_0_factories, max_factories),
                self.factories_numpy(self.player_1_factories, max_factories),
            ], axis=0)
        else:
            return np.stack([
                self.factories_numpy(self.player_1_factories, max_factories),
                self.factories_numpy(self.player_0_factories, max_factories),
            ], axis=0)
        
    def factories_numpy(self, factories: List[Factory], max_factories: int = 2) -> np.ndarray:
        assert max_factories >= 1, "Max number of factories must be at least 1"
        if len(factories) < max_factories:
            factories += [Factory()] * (max_factories - len(factories))
        return np.stack([factory.numpy() for factory in factories[:max_factories]], axis=0)

class Unit:
    def __init__(self, raw_unit_obs: Optional[Dict[str, Any]] = None) -> None:
        if raw_unit_obs is None:
            # [array([0, 4, 0, 0, 0, 1]), ...]
            self.action_queue: List[np.ndarray] = []
            
            self.cargo = Cargo()
            self.pos: np.ndarray = np.array([-1, -1]) # array([30, 11])
            self.power: int = 0
            self.team_id: int = -1
            self.unit_id: str = -1
            self.unit_type: str = "NONE" # HEAVY, LIGHT
        else:
            # [array([0, 4, 0, 0, 0, 1]), ...]
            self.action_queue: List[np.ndarray] = raw_unit_obs['action_queue']
            
            self.cargo = Cargo(raw_unit_obs['cargo'])
            self.pos: np.ndarray = raw_unit_obs['pos'] # array([30, 11])
            self.power: int = raw_unit_obs['power']
            self.team_id: int = raw_unit_obs['team_id']
            self.unit_id: str = raw_unit_obs['unit_id']
            self.unit_type: str = raw_unit_obs['unit_type'] # HEAVY, LIGHT
        
    def actions_numpy(self, max_actions: int = 2):
        n = len(self.action_queue)
        
        action_queue = self.action_queue
        if n < max_actions:
            action_queue += [np.array([0,0,0,0,0,0])] * (max_actions - n)
        
        return np.concatenate(action_queue[:max_actions], axis=0)
        
    def numpy(self, max_actions: int = 2) -> np.ndarray:
        return np.concatenate([
            self.cargo.numpy(),
            self.pos,
            np.array([
                self.power,
                # self.unit_id,
                1 if self.unit_type == "HEAVY" \
                    else -1 if self.unit_type == "LIGHT" \
                        else 0,
            ]),
            self.actions_numpy(max_actions),
        ])

class Units:
    def __init__(self, raw_units_obs: Dict[str, Dict]) -> None:
        self.player_0 = self.__convert_unitsdict_to_unitslist(
            raw_units_obs['player_0']
        )
        self.player_1 = self.__convert_unitsdict_to_unitslist(
            raw_units_obs['player_1']
        )
    
    def get_units_of_agent(self, agent: str) -> List[Unit]:
        if agent == 'player_0':
            return self.player_0
        else:
            return self.player_1
    
    def __convert_unitsdict_to_unitslist(self, units_dict: Dict[str, Dict]) -> List[Unit]:
        lst: List[Unit] = []
        for unit in units_dict.values():
            lst.append(Unit(unit))
        return lst
    
    def numpy(self, agent: str, max_units: int = 2) -> np.ndarray:
        # The order depends on which agent
        if agent == 'player_0':
            return np.stack([
                self.units_numpy(self.player_0, max_units),
                self.units_numpy(self.player_1, max_units),
            ])
        else:
            return np.stack([
                self.units_numpy(self.player_1, max_units),
                self.units_numpy(self.player_0, max_units),
            ])
        
    def units_numpy(self, units: List[Unit], max_units: int = 2) -> np.ndarray:
        assert max_units >= 1, "Max number of units must be at least 1"
        if len(units) < max_units:
            units += [Unit()] * (max_units - len(units))
        return np.stack([unit.numpy() for unit in units[:max_units]], axis=0)

class Player:
    def __init__(self, raw_player_obs: Dict[str, Any]) -> None:
        self.units = Units(raw_player_obs['units'])
        self.teams = Team(raw_player_obs['teams'])
        self.factories = Factories(raw_player_obs['factories'])
        self.board = Board(raw_player_obs['board'])
        self.real_env_steps: int = raw_player_obs['real_env_steps']
        self.global_id: int = raw_player_obs['global_id']

class Observation:
    def __init__(self, raw_obs: Dict[str, Dict]) -> None:
        self.player_0: Player = Player(raw_obs['player_0'])
        self.player_1: Player = Player(raw_obs['player_1'])
