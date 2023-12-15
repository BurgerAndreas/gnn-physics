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


def plot_train_loss(cfg):
    """Load dataframe with loss over epochs during training and plot.
    Note that these are single step predictions.
    """
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
    path_fig = os.path.join(PLOTS_DIR, model_name + ".png")

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


def make_animation(cfg, start_step=0, num_steps=500, single_step=True, use_test_traj=False):
    """Animate predicted velocities

    single_step: if to use single step predictions or accumulate errors

    start_step:
    dirty cheating
    first few timesteps (~5-20) are very different than the bulk of steps
    since we train on (much) less data than a single trajectory
    early timesteps are out of distribution for training data

    use_test_traj: animate an unseen trajectory 
    """

    # get true values, predictions, and the difference
    print("Generating velocity fields...")
    data_pred, data_true, data_error = get_pred_to_plot(
        cfg,
        (start_step, start_step + num_steps),
        single_step,
        use_test_traj=use_test_traj,
    )

    # how many frames
    num_steps = len(data_true)
    fps = 10
    # keep gif to less than 5 seconds
    if num_steps <= 50:
        skip = 1
    elif num_steps == 500:
        skip = 10
    else:
        skip = int(math.ceil(num_steps / 50))
    num_frames = num_steps // skip
    print(f"Generating gif with {num_steps} timesteps over {num_frames} frames")

    # compute the bounding box (axis range of the plot)
    step_middle = start_step + int(num_steps / 2)
    # use max and min velocity of data_true dataset at the first step for both
    # data_true and prediction plots
    bb_min = data_true[step_middle].x[:, 0:2].min()  # first two columns are velocity
    bb_max = data_true[step_middle].x[:, 0:2].max()
    # for the error only use the nodes we computed the error over during training
    normal = torch.tensor(0)
    outflow = torch.tensor(5)
    loss_mask = torch.logical_or(
        (torch.argmax(data_true[0].x[:, 2:], dim=1) == normal),
        (torch.argmax(data_true[0].x[:, 2:], dim=1) == outflow),
    )
    # (num_nodes,) -> (num_nodes, 9), 9 = 2 + 7 = v_x, v_y, node_type
    # loss_mask = loss_mask.unsqueeze(1).repeat(1, 11)
    # error in the first 5 timesteps is usually 100x of error later
    masked_err = data_error[step_middle].x[loss_mask]
    bb_min_error = masked_err[:, 0:2].min().item() / 2
    bb_max_error = masked_err[:, 0:2].max().item() / 2

    # center scale
    center_scale = False
    if center_scale:
        bb_max = max(abs(bb_min), bb_max)
        bb_min = -1 * bb_max
        bb_max_error = max(abs(bb_min_error), bb_max_error)
        bb_min_error = -1 * bb_max_error

    # figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))

    def animate(num):
        """Generator for animation.FuncAnimation().
        Gets called for each frame.
        """
        step = (num * skip) % num_steps
        traj = 0

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

            ax.set_title("{} Step {}".format(title, step), fontsize="20")

            # format every axes (timestep)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=20)
            if count == 0:
                clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
            count += 1

        # reduce whitespace
        fig.subplots_adjust(
            left=0.001,  # the left side of the subplots of the figure
            right=0.95,  # the right side of the subplots of the figure
            bottom=0.01,  # the bottom of the subplots of the figure
            top=0.95,  # the top of the subplots of the figure
            wspace=0.2,  # the amount of width reserved for blank space between subplots
            hspace=0.2,  # the amount of height reserved for white space between subplots
        )
        # this will make the layout change frame to frame
        # fig.tight_layout()
        return (fig,)

    # Save animation for visualization
    true_anim = animation.FuncAnimation(
        fig,
        func=animate,
        frames=num_frames,
        interval=1000,  # delay between frames in ms
    )
    writergif = animation.PillowWriter(fps=fps)
    # save location
    model_name = name_from_config(cfg)
    gif_name = "x_velocity_"
    if not single_step:
        gif_name += "rollout_"
    if use_test_traj:
        gif_name += "testtraj_"
    gif_name += f"{start_step}_{num_steps+start_step}_"
    gif_name += model_name
    anim_path = os.path.join(ANIM_DIR, f"{gif_name}_anim.gif")
    # default dpi: 25 MB
    # dpi=100 25 MB
    # dpi=10 500 KB
    true_anim.save(anim_path, writer=writergif, dpi=100)
    # plt.show(block=True)
    print("Saved animation of x_velocity to", anim_path)
    return


def get_pred_to_plot(cfg, steps=[50, 550], single_step=True, use_test_traj=False):
    """Load data, load model, get predictions.
    Return predicted, true, and L1 difference of data.x (velocity)
    """
    # Set the random seeds for all random number generators
    torch.manual_seed(cfg.rseed)  # Torch
    random.seed(cfg.rseed)  # Python
    np.random.seed(cfg.rseed)  # NumPy

    # animation function cannot work with data on GPU
    device = torch.device("cpu")

    # get the first trajectory from the dataset
    # this will be partially seen / unseen timesteps
    # depending on cfg.data.shuffle and cfg.data.one_traj
    datapath = cfg.data.datapath
    if use_test_traj:
        datapath = datapath.replace("train", "test")
    print("datapath", datapath)
    path_data = os.path.join(DATASET_DIR, datapath)
    dataset = torch.load(path_data)[steps[0] : steps[1]]

    # look for model checkpoint
    model_name = name_from_config(cfg)
    path_model_checkpoint = os.path.join(CHECKPOINT_DIR, model_name + "_model.pt")

    # get infos
    path_infos = os.path.join(CHECKPOINT_DIR, model_name + "_infos.pkl")
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
    if single_step:
        data_pred, data_true, data_error = get_single_step_pred(
            model, dataset, stats=[mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge]
        )
    else:
        data_pred, data_true, data_error = get_rollout_pred(
            model, dataset, stats=[mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge]
        )

    return data_pred, data_true, data_error


def get_single_step_pred(model, dataset, stats: list, device="cpu"):
    """Single step predictions of the velocity for all timesteps in the dataset.

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

    in: stats = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge]

    return predicted, true, and L1 difference data.x (velocity)
    """
    model.eval()
    # create placeholders that we will overwrite now
    data_pred = copy.deepcopy(dataset)
    data_true = copy.deepcopy(dataset)
    data_error = copy.deepcopy(dataset)

    # loop over timesteps (datapoints) in the dataset
    for t in range(len(dataset)):
        dataset[t] = dataset[t].to(device)
        data_pred[t] = data_pred[t].to(device)
        with torch.no_grad():
            # get the predicted acceleration
            pred = model(dataset[t], *stats)
            # pred gives the learnt accelaration (y) between two timsteps
            # next_vel = curr_vel + pred * DELTA_T
            data_pred[t].x[:, 0:2] = dataset[t].x[:, 0:2] + pred[:] * DELTA_T
            # true values
            data_true[t].x[:, 0:2] = dataset[t].x[:, 0:2] + dataset[t].y * DELTA_T
            # data_true[t] - data_pred[t] = data_error[t]
            data_error[t].x[:, 0:2] = data_pred[t].x[:, 0:2] - data_true[t].x[:, 0:2]
    return data_pred, data_true, data_error


def get_rollout_pred(model, dataset, stats: list, device="cpu"):
    """Predictions of velocity over a whole rollout.
    Make sure to only pass in one trajectory,
    as code does not know where one trajectory starts and ends.
    and that timesteps are in correct sequence (not shuffled).

    return predicted, true, and L1 difference data.x (velocity)
    """
    if len(dataset) > 599:
        print(
            "Data passed to get_rollout_pred is more than one trajectory. This will lead to nonsensical results past 599 steps."
        )
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
        # dataset[t] = dataset[t].to(device)
        # data_pred[t] = data_pred[t].to(device)
        with torch.no_grad():
            # get the predicted acceleration
            data_pred[t].x[:, 0:2] = last_x
            pred = model(data_pred[t], *stats)
            # pred gives the learnt accelaration (y) between two timsteps
            # next_vel = curr_vel + pred * DELTA_T
            data_pred[t].x[:, 0:2] = data_pred[t].x[:, 0:2] + pred[:] * DELTA_T
            last_x = copy.deepcopy(data_pred[t].x[:, 0:2])
            # true values
            data_true[t].x[:, 0:2] = dataset[t].x[:, 0:2] + dataset[t].y * DELTA_T
            # data_true[t] - data_pred[t] = data_error[t]
            data_error[t].x[:, 0:2] = data_pred[t].x[:, 0:2] - data_true[t].x[:, 0:2]

    return data_pred, data_true, data_error


def plot_rollout_error(cfg, start_step=50, num_steps=100, use_test_traj=False):
    """Calc rollout prediction and plot RMSE."""
    # get data to plot
    data_pred, data_true, data_error = get_pred_to_plot(
        cfg,
        (start_step, start_step + num_steps),
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

    model_name = name_from_config(cfg)
    path_fig = os.path.join(
        PLOTS_DIR, model_name + f"_{start_step}_{num_steps+start_step}_rollout.png"
    )

    f = plt.figure()
    plt.title("Rollout - Velocity RMSE")
    plt.plot(velocity_rmse, label=model_name)
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")

    plt.legend()
    # plt.show()
    f.savefig(path_fig, bbox_inches="tight")
    print("Saved rollout error plot to", path_fig)
    return f
