import hydra
from omegaconf import DictConfig, OmegaConf

import gnn.plotting as plotting


@hydra.main(version_base=None, config_path="conf", config_name="default")
def call_animation_with_config(cfg: DictConfig) -> None:
    plotting.animate_rollout(cfg)
    return


if __name__ == "__main__":
    call_animation_with_config()
