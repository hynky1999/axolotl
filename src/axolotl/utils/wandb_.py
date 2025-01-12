"""Module for wandb utilities"""

import os
from pdb import run
import uuid


def setup_wandb_env_vars(cfg):
    if cfg.wandb_mode and cfg.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = cfg.wandb_mode
    elif cfg.wandb_project and len(cfg.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        cfg.use_wandb = True
        if cfg.wandb_entity and len(cfg.wandb_entity) > 0:
            os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        if cfg.wandb_watch and len(cfg.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = cfg.wandb_watch
        if cfg.wandb_log_model and len(cfg.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = cfg.wandb_log_model
        if cfg.wandb_run_id and len(cfg.wandb_run_id) > 0:
            run_id = cfg.wandb_run_id
        else:
            run_id = uuid.uuid4().hex

        os.environ["WANDB_RUN_ID"] = run_id
        # Setup output dir if not already set
        if not cfg.output_dir:
            cfg.output_dir = os.path.join(
                "models",run_id
            )
    else:
        os.environ["WANDB_DISABLED"] = "true"
