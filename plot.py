import hydra
from omegaconf import DictConfig, OmegaConf

import gnn.plotting as plotting


@hydra.main(version_base=None, config_path="conf", config_name="default")
def call_plotting_with_config(cfg: DictConfig) -> None:
    plotting.plot_train_loss(cfg)
    plotting.plot_rollout_error(cfg)
    return


if __name__ == "__main__":
    call_plotting_with_config()
