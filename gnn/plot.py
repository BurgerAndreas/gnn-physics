# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

import torch
import os
import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader

import numpy as np
import time
import torch.optim as optim
import tqdm
import pandas as pd
import copy
import matplotlib.pyplot as plt
import h5py
import tensorflow.compat.v1 as tf
import functools
import json
from torch_geometric.data import Data
import enum

from matplotlib import tri as mtri
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gnn.dirs import PLOTS_DIR


def name_from_config(cfg):
    return (
        "model_nl"
        + str(cfg.model.num_layers)
        + "_bs"
        + str(cfg.training.batch_size)
        + "_hd"
        + str(cfg.model.hidden_dim)
        + "_ep"
        + str(cfg.training.epochs)
        + "_wd"
        + str(cfg.training.weight_decay)
        + "_lr"
        + str(cfg.training.lr)
        + "_sh"
        + str(cfg.data.shuffle)
        + "_tr"
        + str(cfg.training.train_size)
        + "_te"
        + str(cfg.training.test_size)
    )


def save_plots(cfg, losses, test_losses, velocity_val_losses):
    model_name = name_from_config(cfg)

    if not os.path.isdir(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    path_fig = os.path.join(PLOTS_DIR, model_name + ".pdf")

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(losses, label="training loss")
    plt.plot(test_losses, label="test loss")
    # if (cfg.save_velo_val):
    #    plt.plot(velocity_val_losses, label="velocity loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.show()
    f.savefig(path_fig, bbox_inches="tight")
    return f


def make_animation(
    gs, pred, evl, path, name, skip=2, save_anim=True, plot_variables=False
):
    """
    input gs is a dataloader and each entry contains attributes of many timesteps.

    """
    print("Generating velocity fields...")
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    num_steps = len(gs)  # for a single trajectory
    num_frames = num_steps // skip
    print(num_steps)

    def animate(num):
        step = (num * skip) % num_steps
        traj = 0

        bb_min = gs[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max = (
            gs[0].x[:, 0:2].max()
        )  # use max and min velocity of gs dataset at the first step for both
        # gs and prediction plots
        bb_min_evl = evl[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max_evl = (
            evl[0].x[:, 0:2].max()
        )  # use max and min velocity of gs dataset at the first step for both
        # gs and prediction plots
        count = 0

        for ax in axes:
            ax.cla()
            ax.set_aspect("equal")
            ax.set_axis_off()

            pos = gs[step].mesh_pos
            faces = gs[step].cells
            if count == 0:
                # ground truth
                velocity = gs[step].x[:, 0:2]
                title = "Ground truth:"
            elif count == 1:
                velocity = pred[step].x[:, 0:2]
                title = "Prediction:"
            else:
                velocity = evl[step].x[:, 0:2]
                title = "Error: (Prediction - Ground truth)"

            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            if count <= 1:
                # absolute values

                mesh_plot = ax.tripcolor(
                    triang, velocity[:, 0], vmin=bb_min, vmax=bb_max, shading="flat"
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
            else:
                # error: (pred - gs)/gs
                mesh_plot = ax.tripcolor(
                    triang,
                    velocity[:, 0],
                    vmin=bb_min_evl,
                    vmax=bb_max_evl,
                    shading="flat",
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
                # ax.triplot(triang, lw=0.5, color='0.5')

            ax.set_title(
                "{} Trajectory {} Step {}".format(title, traj, step), fontsize="20"
            )
            # ax.color

            # if (count == 0):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=20)

            clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
            count += 1
        return (fig,)

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)

    if save_anim:
        gs_anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=1000  # delay between frames in ms
        )
        writergif = animation.PillowWriter(fps=10)
        anim_path = os.path.join(path, "{}_anim.gif".format(name))
        gs_anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        anim_path = ""
    return anim_path


def visualize(
    loader,
    best_model,
    file_dir,
    cfg,
    gif_name,
    stats_list,
    delta_t=0.01,
    skip=1,
    device="cpu",
):
    best_model.eval()
    viz_data = {}
    gs_data = {}
    eval_data = {}
    viz_data_loader = copy.deepcopy(loader)
    gs_data_loader = copy.deepcopy(loader)
    eval_data_loader = copy.deepcopy(loader)
    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )

    for data, viz_data, gs_data, eval_data in zip(
        loader, viz_data_loader, gs_data_loader, eval_data_loader
    ):
        data = data.to(device)
        viz_data = data.to(device)
        with torch.no_grad():
            pred = best_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            # pred gives the learnt accelaration between two timsteps
            # next_vel = curr_vel + pred * delta_t
            viz_data.x[:, 0:2] = data.x[:, 0:2] + pred[:] * delta_t
            gs_data.x[:, 0:2] = data.x[:, 0:2] + data.y * delta_t
            # gs_data - viz_data = error_data
            eval_data.x[:, 0:2] = viz_data.x[:, 0:2] - gs_data.x[:, 0:2]

    # print(viz_data_loader)
    anim_path = make_animation(
        gs_data_loader,
        viz_data_loader,
        eval_data_loader,
        file_dir,
        gif_name,
        skip,
        True,
        False,
    )

    return eval_data_loader, anim_path
