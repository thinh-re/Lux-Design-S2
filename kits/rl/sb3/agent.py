"""
This file is where your agent's logic is kept. 
Define a bidding policy, factory placement policy, 
as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase 
and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error 
e.g. print("message", file=sys.stderr)
"""

import os.path as osp

import numpy as np
import torch as th
from game_env import make_env
from lux.config import EnvConfig
from nn import load_policy
from stable_baselines3.ppo import PPO
from train_nn import CustomNet
from wrappers import ControllerWrapper, ObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model.zip"


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        
        _, _, observation_wrapper = make_env(
            "LuxAI_S2-v0", 0, max_episode_steps=100,
            returns_controller_observation=True
        )()
        self.controller = ControllerWrapper(self.env_cfg)

        directory = osp.dirname(__file__)
        model_path = osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH)
        self.init_sb3_model(model_path)
        # load our RL policy
        # self.policy = load_policy(
        #     model_path,
        #     observation_wrapper=observation_wrapper,
        #     controller_wrapper=self.controller,
        # )
        # self.policy.eval()
        
    def init_sb3_model(self, model_path: str) -> None:
        env_id = "LuxAI_S2-v0"
        env, controller_wrapper, observation_wrapper = make_env(
            env_id, 0, max_episode_steps=100,
            returns_controller_observation=True
        )()
        
        env.reset()
        policy_kwargs = dict(
            features_extractor_class=CustomNet,
            features_extractor_kwargs=dict(
                # observation_space=observation_wrapper.observation_space, # do not need to specify observation_space since PPO automatically adds observation_space into CustomNet
                action_space=controller_wrapper.action_space,
                features_dim=128,    
            ),
        )
        
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        self.model.load(model_path)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False
        # if player == "player_1":
        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return dict(spawn=pos, metal=metal, water=metal)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = ObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        # with th.no_grad():
        #     # NOTE: we set deterministic to False here, which is only recommended for RL agents
        #     # that create too many invalid actions (less of an issue if you train with invalid action masking)

        #     # to improve performance, we have a rule based action mask generator for the controller used
        #     # which will force the agent to generate actions that are valid only.
        #     action_mask = (
        #         th.from_numpy(self.controller.action_masks(self.player, raw_obs))
        #         .unsqueeze(0)  # we unsqueeze/add an extra batch dimension =
        #         .bool()
        #     )
        #     actions = (
        #         self.policy.act(
        #             obs.unsqueeze(0), deterministic=False, action_masks=action_mask
        #         )
        #         .cpu()
        #         .numpy()
        #     )

        action, _states = self.model.predict(obs, deterministic=True)
        
        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, action
        )

        # commented code below adds watering lichen which can easily improve your agent
        # shared_obs = raw_obs[self.player]
        # factories = shared_obs["factories"][self.player]
        # for unit_id in factories.keys():
        #     factory = factories[unit_id]
        #     if 1000 - step < 50 and factory["cargo"]["water"] > 100:
        #         lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
