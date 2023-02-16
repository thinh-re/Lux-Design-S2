from typing import Any, Dict, List

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
    def __init__(self, raw_board_obs: Dict[str, Any]) -> None:
        self.factories_per_team: int = raw_board_obs['factories_per_team']
        self.ice: np.ndarray = raw_board_obs['ice']
        self.lichen: np.ndarray = raw_board_obs['lichen']
        self.lichen_strains: np.ndarray = raw_board_obs['lichen_strains']
        self.ore: np.ndarray = raw_board_obs['ore']
        self.rubble: np.ndarray = raw_board_obs['rubble']
        self.valid_spawns_mask: np.ndarray = raw_board_obs['valid_spawns_mask']

class Cargo:
    def __init__(self, raw_cargo_obs: Dict[str, Any]) -> None:
        self.ice: int = raw_cargo_obs['ice']
        self.metal: int = raw_cargo_obs['metal']
        self.ore: int = raw_cargo_obs['ore']
        self.water: int = raw_cargo_obs['water']

class Factory:
    def __init__(self, raw_factory_obs: Dict[str, Any]) -> None:
        self.cargo = Cargo(raw_factory_obs['cargo'])
        self.pos: np.ndarray = raw_factory_obs['pos'] # array([10, 10])
        self.power: int = raw_factory_obs['power']
        self.strain_id: int = raw_factory_obs['strain_id']
        self.team_id: int = raw_factory_obs['team_id']
        self.unit_id: str = raw_factory_obs['unit_id']

class Factories:
    def __init__(self, raw_factories_obs: Dict[str, Dict]) -> None:
        self.player_0_factories: List[Factory] = self.__convert_factoriesdict_to_factorieslist(
            raw_factories_obs['player_0']
        )
        self.player_1_factories: List[Factory] = self.__convert_factoriesdict_to_factorieslist(
            raw_factories_obs['player_1']
        )
            
    def __convert_factoriesdict_to_factorieslist(self, factories_dict: Dict[str, Dict]) -> List[Factory]:
        lst: List[Factory] = []
        for factory in factories_dict.values():
            lst.append(Factory(factory))
        return lst

class Unit:
    def __init__(self, raw_unit_obs: Dict[str, Any]) -> None:
        # [array([0, 4, 0, 0, 0, 1]), ...]
        self.action_queue: List[np.ndarray] = raw_unit_obs['action_queue']
        
        self.cargo = Cargo(raw_unit_obs['cargo'])
        self.pos: np.ndarray = raw_unit_obs['pos'] # array([30, 11])
        self.power: int = raw_unit_obs['power']
        self.team_id: int = raw_unit_obs['team_id']
        self.unit_id: str = raw_unit_obs['unit_id']
        self.unit_type: str = raw_unit_obs['unit_type'] # HEAVY, LIGHT

class Units:
    def __init__(self, raw_units_obs: Dict[str, Dict]) -> None:
        self.player_0 = self.__convert_unitsdict_to_unitslist(
            raw_units_obs['player_0']
        )
        self.player_1 = self.__convert_unitsdict_to_unitslist(
            raw_units_obs['player_1']
        )
    
    def __convert_unitsdict_to_unitslist(self, units_dict: Dict[str, Dict]) -> List[Unit]:
        lst: List[Unit] = []
        for unit in units_dict.values():
            lst.append(Unit(unit))
        return lst

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
