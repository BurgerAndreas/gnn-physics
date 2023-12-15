import torch
import os
import sys
import pathlib
import copy
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import pickle
import math

from matplotlib import tri as mtri
from matplotlib import animation
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from gnn.plotting import get_pred_to_plot, name_from_config
from gnn.dirs import CHECKPOINT_DIR, DATASET_DIR, PLOTS_EXTRA_DIR, ANIM_DIR, DELTA_T

import hydra
from omegaconf import OmegaConf


START_STEP = 50


def plot_rollout_error_noise(num_steps=100, use_test_traj=False):
    """Calc rollout prediction and plot RMSE."""

    f = plt.figure()
    plt.title("Rollout - Velocity RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")

    plots_dir = PLOTS_EXTRA_DIR

    config = ["+datasize=medium"]

    for override in [config + ["+noise=paper"], config]:
        # load config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(version_base=None, config_path="conf", job_name="plot"):
            cfg = hydra.compose(
                config_name="default_init", overrides=override, return_hydra_config=True
            )

        # get name
        model_name = name_from_config(cfg)

        # get data to plot
        data_pred, data_true, data_error = get_pred_to_plot(
            cfg,
            (START_STEP, START_STEP + num_steps),
            single_step=False,
            use_test_traj=use_test_traj,
        )
        # see run_gnn.test()
        velocity_rmse = []
        for t in range(len(data_true)):
            normal = torch.tensor(0)
            outflow = torch.tensor(5)
            # 2: is node_type
            # :2 is x and y velocity
            loss_mask = torch.logical_or(
                (torch.argmax(data_true[t].x[:, 2:], dim=1) == normal),
                (torch.argmax(data_true[t].x[:, 2:], dim=1) == outflow),
            )
            # abs error L1 -> MSE L2, x and y -> float
            error = torch.sum((data_error[t].x[:, :2]) ** 2, axis=1)
            # mean over all points
            velocity_rmse.append(torch.sqrt(torch.mean(error[loss_mask])))
        plt.plot(velocity_rmse, label=model_name)

    plt.legend()
    # plt.show()
    path_fig = os.path.join(
        plots_dir,
        model_name + f"_{START_STEP}_{num_steps+START_STEP}_rollout_noise.png",
    )
    f.savefig(path_fig, bbox_inches="tight")
    print("Saved rollout error plot to", path_fig)
    return f


if __name__ == "__main__":
    plot_rollout_error_noise()
