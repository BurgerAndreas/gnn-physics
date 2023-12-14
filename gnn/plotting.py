# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

import torch
import os
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

import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

import gnn.utils as stats
import gnn.gnn as models
from gnn.dirs import CHECKPOINT_DIR, DATASET_DIR, PLOTS_DIR, ANIM_DIR, DELTA_T


def name_from_config(cfg):
    if cfg.override_dirname:
        override = cfg.override_dirname
        override = override.replace(",", "-").replace("+", "")
        override = override.replace("=", "_").replace(".", "")
        # ignore this
        override = override.replace("-resume_checkpoint_False", "")
        return override
    else:
        return cfg.config_name
    return model_name


def save_plots(cfg):
    # find data
    model_name = name_from_config(cfg)
    path_df = os.path.join(CHECKPOINT_DIR, model_name + "_losses.pkl")

    # load data
    # pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "velocity_val_loss"])
    df = pd.read_pickle(path_df)
    train_loss = df["train_loss"]
    test_loss = df["test_loss"]

    if not os.path.isdir(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    path_fig = os.path.join(PLOTS_DIR, model_name + ".pdf")

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(train_loss, label="training loss")
    plt.plot(test_loss, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    # plt.show()
    f.savefig(path_fig, bbox_inches="tight")
    print("Saved loss plot to", path_fig)
    return f



def animate_rollout(cfg, num_steps=50):
    # Set the random seeds for all random number generators
    torch.manual_seed(cfg.rseed)  # Torch
    random.seed(cfg.rseed)  # Python
    np.random.seed(cfg.rseed)  # NumPy

    # animation function cannot work with data on GPU
    device = torch.device("cpu")

    # get the first trajectory from the dataset
    # this will be partially seen / unseen timesteps
    # depending on cfg.data.shuffle and cfg.data.one_traj
    file_path = os.path.join(DATASET_DIR, cfg.data.datapath)
    dataset = torch.load(file_path)[:num_steps]

    # look for model checkpoint
    model_name = name_from_config(cfg)
    path_model_checkpoint = os.path.join(CHECKPOINT_DIR, model_name + "_model.pt")
    path_infos = os.path.join(CHECKPOINT_DIR, model_name + "_infos.pkl")

    # get infos
    with open(path_infos, "rb") as f:
        infos = pickle.load(f)
    (num_node_features, num_edge_features, num_classes, stats_list) = infos

    # recompute statistics, since we use a different set of data than during training
    stats_list = stats.get_stats(dataset)

    # load and instantiate model
    model = models.MeshGraphNet(
        num_node_features, num_edge_features, cfg.model.hidden_dim, num_classes, cfg
    ).to(device)
    model.load_state_dict(torch.load(path_model_checkpoint, map_location=device))
    model.eval()

    # animate predicted velocities

    # get data statistics we need for normalization in eval
    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list
    # move tensors to the CPU
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )

    # get the predictions for the animation
    data_pred, data_true, data_error = get_single_step_pred(
        model, dataset, stats=[mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge]
    )

    print("Generating velocity fields...")
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))


    num_steps = len(data_true)  # for a single trajectory
    fps = 10
    if num_steps <= 50:
        # 5 seconds
        skip = 1
    else:
        skip = int(math.ceil(num_steps / 50))
    num_frames = num_steps // skip
    print(f'Generating gif with {num_frames} frames')

    def animate(num):
        step = (num * skip) % num_steps
        traj = 0

        # compute the bounding box (axis range of the plot)
        # use max and min velocity of data_true dataset at the first step for both
        bb_min = data_true[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max = data_true[0].x[:, 0:2].max()
        # data_true and prediction plots
        bb_min_error = data_error[0].x[:, 0:2].min()
        bb_max_error = data_error[0].x[:, 0:2].max()

        # data_true and prediction plots
        count = 0
        for ax in axes:
            ax.cla()
            ax.set_aspect("equal")
            ax.set_axis_off()

            pos = data_true[step].mesh_pos
            faces = data_true[step].cells
            if count == 0:
                # ground truth
                velocity = data_true[step].x[:, 0:2]
                title = "Ground truth:"
            elif count == 1:
                velocity = data_pred[step].x[:, 0:2]
                title = "Prediction:"
            else:
                velocity = data_error[step].x[:, 0:2]
                title = "Error: (Prediction - Ground truth)"

            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            if count <= 1:
                # absolute values
                mesh_plot = ax.tripcolor(
                    triang, velocity[:, 0], vmin=bb_min, vmax=bb_max, shading="flat"
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
            else:
                # error: (data_pred - data_true)/data_true
                mesh_plot = ax.tripcolor(
                    triang,
                    velocity[:, 0],
                    vmin=bb_min_error,
                    vmax=bb_max_error,
                    shading="flat",
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
                # ax.triplot(triang, lw=0.5, color='0.5')

            ax.set_title(
                "{} Trajectory {} Step {}".format(title, traj, step), fontsize="20"
            )

            # format every axes (timestep)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=20)
            clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
            count += 1
        return (fig,)

    # Save animation for visualization
    true_anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=1000  # delay between frames in ms
    )
    writergif = animation.PillowWriter(fps=fps)
    gif_name = "x_velocity_" + model_name.replace("model_", "")
    anim_path = os.path.join(ANIM_DIR, "{}_anim.gif".format(gif_name))
    true_anim.save(anim_path, writer=writergif)
    # plt.show(block=True)
    print('Saved animation of x_velocity to', anim_path)
    return


def get_single_step_pred(model, dataset, stats: list, device='cpu'):
    """Single step predictions for all timesteps in the dataset.

    Predictions are single step,
    because we use the ground truth as inputs to the model
    instead of feeding back previous predictions into the model as input.
    There are no compounding errors.
    We should not see any benefits of using noise during training,
    because the noise imitates using the imperfect predictions as input.

    Assuming that the dataset in the correct order (not shuffled)
    data_pred, data_true, data_error are one timestep ahead of dataset
    i.e. indices are shifted by one
    because instead of x(t+1) = x(t) + dt
    we are setting x(t) = x(t) + dt

    stats = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge]
    """
    # step_pred = {}
    # step_true = {}
    # step_error = {}
    model.eval()
    # create placeholders that we will overwrite now
    data_pred = copy.deepcopy(dataset)
    data_true = copy.deepcopy(dataset)
    data_error = copy.deepcopy(dataset)
    # loop over timesteps (datapoints) in the dataset
    for step_data, step_pred, step_true, step_error in zip(
        dataset, data_pred, data_true, data_error
    ):
        step_data = step_data.to(device)
        step_pred = step_data.to(device)  # TODO bug?
        with torch.no_grad():
            # get the predicted acceleration
            pred = model(step_data, *stats)
            # pred gives the learnt accelaration (y) between two timsteps
            # next_vel = curr_vel + pred * DELTA_T
            step_pred.x[:, 0:2] = step_data.x[:, 0:2] + pred[:] * DELTA_T
            # true values
            step_true.x[:, 0:2] = step_data.x[:, 0:2] + step_data.y * DELTA_T
            # step_true - step_pred = step_error
            step_error.x[:, 0:2] = step_pred.x[:, 0:2] - step_true.x[:, 0:2]

    # model.eval()
    # # create placeholders that we will overwrite now
    # data_pred = copy.deepcopy(dataset)
    # data_true = copy.deepcopy(dataset)
    # data_error = copy.deepcopy(dataset)
    # # loop over timesteps (datapoints) in the dataset
    # for t in range(len(dataset)):
    #     dataset[t] = dataset[t].to(device)
    #     data_pred[t] = data_pred[t].to(device) 
    #     with torch.no_grad():
    #         # get the predicted acceleration
    #         pred = model(dataset[t], *stats)
    #         # pred gives the learnt accelaration (y) between two timsteps
    #         # next_vel = curr_vel + pred * DELTA_T
    #         data_pred[t].x[:, 0:2] = dataset[t].x[:, 0:2] + pred[:] * DELTA_T
    #         # true values
    #         data_true[t].x[:, 0:2] = dataset[t].x[:, 0:2] + dataset[t].y * DELTA_T
    #         # data_true[t] - data_pred[t] = data_error[t]
    #         data_error[t].x[:, 0:2] = data_pred[t].x[:, 0:2] - data_true[t].x[:, 0:2]

    return data_pred, data_true, data_error

def get_rollout_pred(model, dataset, stats: list, device='cpu'):
    model.eval()
    # create placeholders that we will overwrite now
    data_pred = copy.deepcopy(dataset)
    data_true = copy.deepcopy(dataset)
    data_error = copy.deepcopy(dataset)
    # use prior predictions as input
    last_x = copy.deepcopy(data_pred[0].x[:, 0:2])
    last_x = last_x.to(device) 
    # loop over timesteps (datapoints) in the dataset
    for t in range(len(dataset)):
        dataset[t] = dataset[t].to(device)
        data_pred[t] = data_pred[t].to(device) 
        with torch.no_grad():
            # get the predicted acceleration
            pred = model(data_pred[t], *stats)
            # pred gives the learnt accelaration (y) between two timsteps
            # next_vel = curr_vel + pred * DELTA_T
            data_pred[t].x[:, 0:2] = last_x + pred[:] * DELTA_T
            last_x = copy.deepcopy(data_pred[0].x[:, 0:2])
            # true values
            data_true[t].x[:, 0:2] = dataset[t].x[:, 0:2] + dataset[t].y * DELTA_T
            # data_true[t] - data_pred[t] = data_error[t]
            data_error[t].x[:, 0:2] = data_pred[t].x[:, 0:2] - data_true[t].x[:, 0:2]
    return data_pred, data_true, data_error
