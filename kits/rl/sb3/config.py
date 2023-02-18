from dataclasses import dataclass

@dataclass
class OurEnvConfig:
    MAX_FACTORIES_IN_OBSERVATION: int = 5
    MAX_UNITS_IN_OBSERVATION: int = 20
    MAX_ACTIONS_PER_UNIT_IN_OBSERVATION: int = 20

    MAX_FACTORIES_IN_ACTION_SPACES: int = 5
    MAX_UNITS_IN_ACTION_SPACES: int = 20
    MAX_ACTIONS_PER_UNIT_IN_ACTION_SPACES: int = 5 # TODO: currently not used, only 1 action per unit
