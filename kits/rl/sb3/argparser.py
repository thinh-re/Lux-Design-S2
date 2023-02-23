from typing import Optional

from tap import Tap


class TrainArgumentParser(Tap):
    seed: Optional[int] = 24  # seed
    n_envs: Optional[
        int
    ] = 8  # number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value
    max_episode_steps: Optional[
        int
    ] = 1000  # Max steps per episode before truncating them
    total_timesteps: Optional[
        int
    ] = 10_000_000  # 3_000_000 # Total timesteps for training
    eval: Optional[
        bool
    ] = False  # If set, will only evaluate a given policy. Otherwise enters training mode
    model_path: Optional[
        str
    ] = "logs/models/latest_model.zip"  # Path to SB3 model weights to use for evaluation
    log_path: Optional[str] = "logs"  # Logging path
    learning_rate: Optional[float] = 3e-4  # The learning rate
    batch_size: Optional[int] = 800  # minibatchsize


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a heavy unit to dig ice and transfer it back to a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args
