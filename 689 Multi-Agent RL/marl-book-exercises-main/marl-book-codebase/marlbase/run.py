import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('C:\\Users\\nick2\\Desktop\\School Stuff\\stats\\AI-Portfolio\\Fall25\\689 Multi-Agent RL\\marl-book-exercises-main\\marl-book-codebase\\marlbase')))

from omegaconf import OmegaConf, DictConfig

import hydra
import numpy as np
import torch


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    env = hydra.utils.call(cfg.env, seed=cfg.seed)

    # Use singular env for evaluation/ recording
    if "parallel_envs" in cfg.env:
        del cfg.env.parallel_envs
    eval_env = hydra.utils.call(
        cfg.env,
        enable_video=True if cfg.algorithm.video_interval else False,
        seed=cfg.seed,
    )

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    assert cfg.env.time_limit is not None, "Time limit must be set."
    hydra.utils.call(
        cfg.algorithm,
        env,
        eval_env,
        logger,
        time_limit=cfg.env.time_limit,
        _recursive_=False,
    )

    return logger.get_state()


if __name__ == "__main__":
    main()
