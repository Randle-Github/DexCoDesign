# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import inspect
import os
import random
import time

import gymnasium as gym
import skrl
import torch
import yaml
from omegaconf import OmegaConf
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def restore_saved_env_scalars(env_cfg, path: str) -> set[str]:
    """Safely restore scalar fields from an Isaac Lab environment YAML.

    ``dump_yaml`` includes Python-specific tags for objects such as NumPy
    dtypes and Gym spaces. Constructing the whole document would require an
    unsafe YAML loader. Compose its node tree instead and restore only scalar
    values whose types match the live configuration object.
    """

    with open(path, encoding="utf-8") as stream:
        root = yaml.compose(stream)

    restored_fields: set[str] = set()

    def decode_scalar(node: yaml.ScalarNode):
        if node.tag == "tag:yaml.org,2002:str":
            return node.value
        if node.tag in {
            "tag:yaml.org,2002:null",
            "tag:yaml.org,2002:bool",
            "tag:yaml.org,2002:int",
            "tag:yaml.org,2002:float",
        }:
            return yaml.safe_load(node.value)
        return None

    def restore(target, node, namespace: str = "") -> None:
        if not isinstance(node, yaml.MappingNode):
            return
        for key_node, value_node in node.value:
            key = key_node.value
            field_path = f"{namespace}.{key}" if namespace else key
            if isinstance(target, dict):
                if key not in target:
                    continue
                current = target[key]
            else:
                if not hasattr(target, key):
                    continue
                current = getattr(target, key)

            if isinstance(value_node, yaml.MappingNode):
                restore(current, value_node, field_path)
                continue
            if not isinstance(value_node, yaml.ScalarNode):
                continue

            value = decode_scalar(value_node)
            compatible = (
                (current is None and value is None)
                or (isinstance(current, bool) and isinstance(value, bool))
                or (
                    isinstance(current, int)
                    and not isinstance(current, bool)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                )
                or (
                    isinstance(current, float)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
                or (isinstance(current, str) and isinstance(value, str))
            )
            if not compatible:
                continue
            if isinstance(current, float):
                value = float(value)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)
            restored_fields.add(field_path)

    restore(env_cfg, root)
    return restored_fields


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Preserve behavior that is specific to the registered Play task. The
    # training environment YAML is restored below, but playback should still
    # run the requested number of environments and show the complete reference
    # instead of terminating at the training failure thresholds.
    play_num_envs = env_cfg.scene.num_envs
    play_only_settings = {
        name: getattr(env_cfg, name)
        for name in (
            "object_failure_distance",
            "object_failure_orientation",
            "randomize_start_phase",
            "log_rollout_diagnostics",
        )
        if hasattr(env_cfg, name)
    }

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # Restore the environment configuration saved alongside the checkpoint.
    # Reapply Play-only behavior and explicit CLI/Hydra overrides afterwards so
    # those choices take precedence over the training configuration.
    saved_env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
    if os.path.isfile(saved_env_cfg_path):
        restored_env_fields = restore_saved_env_scalars(env_cfg, saved_env_cfg_path)
        if (
            hasattr(env_cfg, "disable_hand_support_collisions")
            and "disable_hand_support_collisions" not in restored_env_fields
        ):
            # Runs created before this option was introduced have no recorded
            # collision choice. Preserve their historical intended behavior.
            env_cfg.disable_hand_support_collisions = True
            print(
                "[INFO] Saved environment predates the hand-support collision "
                "setting; defaulting disable_hand_support_collisions=True."
            )
        env_cfg.scene.num_envs = play_num_envs
        for name, value in play_only_settings.items():
            setattr(env_cfg, name, value)

        env_override_args = [
            override.removeprefix("env.")
            for override in hydra_args
            if override.startswith("env.")
        ]
        if env_override_args:
            explicit_env_cfg = OmegaConf.to_container(
                OmegaConf.from_dotlist(env_override_args), resolve=True
            )
            env_cfg.from_dict(explicit_env_cfg)
        print(
            "[INFO] Loaded saved environment configuration "
            f"({len(restored_env_fields)} scalar fields) from: {saved_env_cfg_path}"
        )
    else:
        print(
            "[WARNING] No saved environment configuration found at "
            f"{saved_env_cfg_path}. Using the registered default configuration."
        )

    # Non-Hydra command-line options have the final precedence.
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # Restore the agent configuration saved alongside the checkpoint. The
    # checkpoint contains parameters and preprocessor state, but the Runner
    # still needs the matching model architecture to be constructed first.
    saved_agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.isfile(saved_agent_cfg_path):
        saved_experiment_cfg = load_yaml(saved_agent_cfg_path)

        # Reapply explicit agent-side Hydra overrides on top of the saved
        # configuration. Hydra's combined config prefixes these with
        # ``agent.``, while Runner receives the inner agent configuration.
        agent_override_args = [
            override.removeprefix("agent.")
            for override in hydra_args
            if override.startswith("agent.")
        ]
        if agent_override_args:
            saved_experiment_cfg = OmegaConf.to_container(
                OmegaConf.merge(
                    OmegaConf.create(saved_experiment_cfg),
                    OmegaConf.from_dotlist(agent_override_args),
                ),
                resolve=True,
            )
        experiment_cfg = saved_experiment_cfg
        print(f"[INFO] Loading saved agent configuration from: {saved_agent_cfg_path}")
    else:
        print(
            "[WARNING] No saved agent configuration found at "
            f"{saved_agent_cfg_path}. Using the registered default configuration."
        )

    # Set the agent and environment seed after restoring the run configuration.
    # Command-line --seed takes precedence over the saved training seed.
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    experiment_cfg["agent"]["experiment"]["wandb"] = False  # don't create a tracking run during playback
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # Set the agent to evaluation mode. SKRL 2.1 removed set_running_mode(),
    # so retain compatibility with both the old and new APIs.
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    obs, _ = env.reset()
    agent_act_requires_states = "states" in inspect.signature(runner.agent.act).parameters
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if agent_act_requires_states:
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
