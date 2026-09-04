# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""HO-Cap MANO residual-policy task."""

import gymnasium as gym

from . import agents


gym.register(
    id="DexCoDesign-MANO-Residual-Direct-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexCoDesign-MANO-Residual-Direct-Play-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexCoDesign-MANO-Residual-Direct-Eval-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexCoDesign-Hand-Residual-Direct-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexCoDesign-Hand-Residual-Direct-Play-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexCoDesign-Hand-Residual-Direct-Eval-v0",
    entry_point=f"{__name__}.mano_residual_env:ManoResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mano_residual_env:ManoResidualEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
