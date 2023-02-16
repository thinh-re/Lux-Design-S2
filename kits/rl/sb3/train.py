"""
Implementation of RL agent. 
Note that luxai_s2 and stable_baselines3 are packages 
not available during the competition running (ATM)
"""


import copy
import os.path as osp
from typing import Dict, Tuple

import gym
import numpy as np
import torch as th
import torch.nn as nn
from argparser import TrainArgumentParser
from gym import spaces
from gym.wrappers import TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecVideoRecorder)
from stable_baselines3.ppo import PPO
from train_nn import CustomNet
from wrappers import (CustomEnvWrapper, SimpleUnitDiscreteController,
                      SimpleUnitObservationWrapper)

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import SB3Wrapper


def make_env(
    env_id: str, rank: int, 
    seed: int = 0, max_episode_steps=100,
):
    def _init() -> LuxAI_S2:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 2 for simplification and keeping returns consistent 
        # as we survive longer if there are more initial resources
        env: LuxAI_S2 = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.
        env = SB3Wrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg),
        )
        env = SimpleUnitObservationWrapper(
            env
        )  # changes observation to include a few simple features
        env = CustomEnvWrapper(env)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def save_model_state_dict(save_path: str, model: BaseAlgorithm) -> None:
    # save the policy state dict for kaggle competition submission
    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)


def evaluate(
    args: TrainArgumentParser, 
    env_id: str, 
    model: PPO,
) -> None:
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)

def evaluate_one_process(
    args: TrainArgumentParser, 
    env_id: str, 
    model: PPO,
) -> None:
    '''Same as evaluate but use only one process. For debug only!'''
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = make_env(env_id, i, max_episode_steps=1000)()
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def train(
    args: TrainArgumentParser, 
    env_id: str, 
    model: BaseAlgorithm,
) -> None:
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))

def train_one_process(
    args: TrainArgumentParser, 
    env_id: str, 
    model: BaseAlgorithm,
) -> None:
    '''Same as train but use only one process. For debug only!'''
    eval_env = make_env(env_id, 0, max_episode_steps=1000)()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))


def main(args: TrainArgumentParser):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    
    # Creates a multiprocess vectorized wrapper for multiple environments, 
    # distributing each environment to its own process, 
    # allowing significant speed up when the environment is computationally complex.
    env = SubprocVecEnv([
        make_env(env_id, i, max_episode_steps=args.max_episode_steps)
        for i in range(args.n_envs)
    ])
    
    env.reset()
    rollout_steps = 4000
    policy_kwargs = dict(
        features_extractor_class=CustomNet,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # PPO from SB3
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=800,
        learning_rate=3e-4,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
        policy_kwargs=policy_kwargs,
    )
    
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model)
        
def main_single_process(args: TrainArgumentParser):
    '''Same as main but use only one process. For debug only!'''
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    
    # Creates a multiprocess vectorized wrapper for multiple environments, 
    # distributing each environment to its own process, 
    # allowing significant speed up when the environment is computationally complex.
    env = make_env(env_id, 0, max_episode_steps=args.max_episode_steps)()
    
    env.reset()
    rollout_steps = 4000
    policy_kwargs = dict(
        features_extractor_class=CustomNet,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # PPO from SB3
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=800,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )
    
    if args.eval:
        evaluate_one_process(args, env_id, model)
    else:
        train_one_process(args, env_id, model)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    args = TrainArgumentParser().parse_args()
    
    if args.n_envs == 1:
        main_single_process(args)
    else:
        main(args)
