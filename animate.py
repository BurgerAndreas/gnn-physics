import hydra
from omegaconf import DictConfig, OmegaConf

import gnn.plotting as plotting


@hydra.main(version_base=None, config_path="conf", config_name="default")
def call_animation_with_config(cfg: DictConfig) -> None:
    plotting.make_animation(cfg, single_step=True, start_step=0, num_steps=500, use_test_traj=True)
    plotting.make_animation(cfg, single_step=True, start_step=0, num_steps=500, use_test_traj=False)

    plotting.make_animation(cfg, single_step=True, start_step=50, num_steps=500, use_test_traj=True)
    plotting.make_animation(cfg, single_step=True, start_step=50, num_steps=500, use_test_traj=False)
    return


if __name__ == "__main__":
    call_animation_with_config()
