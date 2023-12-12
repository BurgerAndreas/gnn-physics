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
from tqdm import trange
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

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, "data/datasets/cylinder_flow_pytorch")
checkpoint_dir = os.path.join(root_dir, "data/checkpoints")
postprocess_dir = os.path.join(root_dir, "data/animations")


# Utility functions, provided in the release of the code from the original MeshGraphNets study:
# https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets


def triangles_to_edges(faces):
    """Computes mesh edges from triangles.
    Note that this triangles_to_edges method was provided as part of the
    code release for the MeshGraphNets paper by DeepMind, available here:
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # collect edges from triangles
    edges = tf.concat(
        [faces[:, 0:2], faces[:, 1:3], tf.stack([faces[:, 2], faces[:, 0]], axis=1)],
        axis=0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (
        tf.concat([senders, receivers], axis=0),
        tf.concat([receivers, senders], axis=0),
    )


class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study:
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """

    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


file_path = os.path.join(dataset_dir, "meshgraphnets_miniset5traj_vis.pt")
dataset_full_timesteps = torch.load(file_path)
dataset = torch.load(file_path)[:1]

print("dataset:\n", dataset)


def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize - mean_vec) / std_vec


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def get_stats(data_list):
    """
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    """

    # mean and std of the node features are calculated
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    # mean and std of the edge features are calculated
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    # mean and std of the output parameters are calculated
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    # Define the maximum number of accumulations to perform such that we do
    # not encounter memory issues
    max_accumulations = 10**6

    # Define a very small value for normalizing to
    eps = torch.tensor(1e-8)

    # Define counters used in normalization
    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    # Iterate through the data in the list to accumulate statistics
    for dp in data_list:
        # Add to the
        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x**2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr**2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y, dim=0)
        std_vec_y += torch.sum(dp.y**2, dim=0)
        num_accs_y += dp.y.shape[0]

        if (
            num_accs_x > max_accumulations
            or num_accs_edge > max_accumulations
            or num_accs_y > max_accumulations
        ):
            break

    mean_vec_x = mean_vec_x / num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accs_x - mean_vec_x**2), eps)

    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(
        torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge**2), eps
    )

    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ]

    return mean_std_list


class MeshGraphNet(torch.nn.Module):
    def __init__(
        self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False
    ):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)
        """
        self.num_layers = args.num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of message passing layers is not >=1"

        processor_layer = self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # decoder: only for node embeddings
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, output_dim)
        )

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, pressure = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.p,
        )

        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x)  # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(
            edge_attr
        )  # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs, mean_vec_y, std_vec_y):
        # Define the node types that we calculate loss for
        normal = torch.tensor(0)
        outflow = torch.tensor(5)

        # Get the loss mask for the nodes of the types we calculate loss for
        loss_mask = torch.logical_or(
            (torch.argmax(inputs.x[:, 2:], dim=1) == torch.tensor(0)),
            (torch.argmax(inputs.x[:, 2:], dim=1) == torch.tensor(5)),
        )

        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        # Find sum of square errors
        error = torch.sum((labels - pred) ** 2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """
        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )  # out has the shape of [E, out_channels]
        updated_nodes = torch.cat(
            [x, out], dim=1
        )  # Complete the aggregation through self-aggregation
        updated_nodes = x + self.node_mlp(updated_nodes)  # residual connection
        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """
        updated_edges = torch.cat(
            [x_i, x_j, edge_attr], dim=1
        )  # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """
        # The axis along which to index number of nodes.
        node_dim = 0
        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce="sum"
        )
        return out, updated_edges


def train(dataset, device, stats_list, args):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    """

    df = pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "velo_val_loss"])

    # Define the model name for saving
    model_name = (
        "model_nl"
        + str(args.num_layers)
        + "_bs"
        + str(args.batch_size)
        + "_hd"
        + str(args.hidden_dim)
        + "_ep"
        + str(args.epochs)
        + "_wd"
        + str(args.weight_decay)
        + "_lr"
        + str(args.lr)
        + "_shuff_"
        + str(args.shuffle)
        + "_tr"
        + str(args.train_size)
        + "_te"
        + str(args.test_size)
    )

    # torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(
        dataset[: args.train_size], batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset[args.train_size :], batch_size=args.batch_size, shuffle=False
    )

    # The statistics of the data are decomposed
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

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2  # the dynamic variables have the shape of 2 (velocity)

    model = MeshGraphNet(
        num_node_features, num_edge_features, args.hidden_dim, num_classes, args
    ).to(device)
    # Paper used Adam optimizer with no learning rate schedule.
    filter_fn = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    # train
    losses = []
    test_losses = []
    velo_val_losses = []
    best_test_loss = np.inf
    best_model = None
    path_model_checkpoint = ""
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops = 0
        for batch in loader:
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_loops += 1
        total_loss /= num_loops
        losses.append(total_loss)

        # Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 10 == 0:
            if args.save_velo_val:
                # save velocity evaluation
                test_loss, velo_val_rmse = test(
                    test_loader,
                    device,
                    model,
                    mean_vec_x,
                    std_vec_x,
                    mean_vec_edge,
                    std_vec_edge,
                    mean_vec_y,
                    std_vec_y,
                    args.save_velo_val,
                )
                velo_val_losses.append(velo_val_rmse.item())
            else:
                test_loss, _ = test(
                    test_loader,
                    device,
                    model,
                    mean_vec_x,
                    std_vec_x,
                    mean_vec_edge,
                    std_vec_edge,
                    mean_vec_y,
                    std_vec_y,
                    args.save_velo_val,
                )

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)

            path_model_checkpoint = os.path.join(
                args.checkpoint_dir, model_name + ".csv"
            )
            df.to_csv(path_model_checkpoint, index=False)

            # save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            # If not the tenth epoch, append the previously calculated loss to the
            # list in order to be able to plot it on the same plot as the training losses
            if args.save_velo_val:
                test_losses.append(test_losses[-1])
                velo_val_losses.append(velo_val_losses[-1])

        if args.save_velo_val:
            df = pd.concat(
                [
                    df,
                    pd.Series(
                        {
                            "epoch": epoch,
                            "train_loss": losses[-1],
                            "test_loss": test_losses[-1],
                            "velo_val_loss": velo_val_losses[-1],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            df = pd.concat(
                [
                    df,
                    # pd.DataFrame.from_dict
                    pd.Series(
                        {
                            "epoch": epoch,
                            "train_loss": losses[-1],
                            "test_loss": test_losses[-1],
                            "test_loss": test_losses[-1],
                        }
                    ),
                ],
                ignore_index=True,
            )

        if epoch % 100 == 0:
            if args.save_velo_val:
                print(
                    "train loss",
                    str(round(total_loss, 2)),
                    "test loss",
                    str(round(test_loss.item(), 2)),
                    "velo loss",
                    str(round(velo_val_rmse.item(), 5)),
                )
            else:
                print(
                    "train loss",
                    str(round(total_loss, 2)),
                    "test loss",
                    str(round(test_loss.item(), 2)),
                )

            if args.save_best_model:
                path_model_checkpoint = os.path.join(
                    args.checkpoint_dir, model_name + ".pt"
                )
                torch.save(best_model.state_dict(), path_model_checkpoint)

    print("Finished!")
    if (best_model is not None) and args.save_best_model:
        path_model_checkpoint = os.path.join(args.checkpoint_dir, model_name + ".pt")
        torch.save(best_model.state_dict(), path_model_checkpoint)
        # torch.save(stats_list, os.path.join(args.checkpoint_dir, model_name + 'statslist.pt'))
        print("Saving best model to", str(path_model_checkpoint))

    return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader


def test(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    is_validation,
    delta_t=0.01,
    save_model_preds=False,
    model_type=None,
):
    """
    Calculates test set losses and validation set errors.
    """

    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # calculate validation error if asked to
            if is_validation:
                # Like for the MeshGraphNets model, calculate the mask over which we calculate
                # flow loss and add this calculated RMSE value to our val error
                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                loss_mask = torch.logical_or(
                    (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                    (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)),
                )

                eval_velo = (
                    data.x[:, 0:2]
                    + unnormalize(pred[:], mean_vec_y, std_vec_y) * delta_t
                )
                gs_velo = data.x[:, 0:2] + data.y[:] * delta_t

                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops += 1
        # if velocity is evaluated, return velo_rmse as 0
    return loss / num_loops, velo_rmse / num_loops


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def save_plots(args, losses, test_losses, velo_val_losses):
    model_name = (
        "model_nl"
        + str(args.num_layers)
        + "_bs"
        + str(args.batch_size)
        + "_hd"
        + str(args.hidden_dim)
        + "_ep"
        + str(args.epochs)
        + "_wd"
        + str(args.weight_decay)
        + "_lr"
        + str(args.lr)
        + "_shuff_"
        + str(args.shuffle)
        + "_tr"
        + str(args.train_size)
        + "_te"
        + str(args.test_size)
    )

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir, model_name + ".pdf")

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)
    # if (args.save_velo_val):
    #    plt.plot(velo_val_losses, label="velocity loss" + " - " + args.model_type)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches="tight")


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
            fig, animate, frames=num_frames, interval=1000
        )
        writergif = animation.PillowWriter(fps=10)
        anim_path = os.path.join(path, "{}_anim.gif".format(name))
        gs_anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass


def visualize(
    loader, best_model, file_dir, args, gif_name, stats_list, delta_t=0.01, skip=1
):
    best_model.eval()
    device = args.device
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
        data = data.to(args.device)
        viz_data = data.to(args.device)
        with torch.no_grad():
            pred = best_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            # pred gives the learnt accelaration between two timsteps
            # next_vel = curr_vel + pred * delta_t
            viz_data.x[:, 0:2] = data.x[:, 0:2] + pred[:] * delta_t
            gs_data.x[:, 0:2] = data.x[:, 0:2] + data.y * delta_t
            # gs_data - viz_data = error_data
            eval_data.x[:, 0:2] = viz_data.x[:, 0:2] - gs_data.x[:, 0:2]

    # print(viz_data_loader)
    make_animation(
        gs_data_loader,
        viz_data_loader,
        eval_data_loader,
        file_dir,
        gif_name,
        skip,
        True,
        False,
    )

    return eval_data_loader


def load_train_plot(cfg):
    # To ensure reproducibility the best we can, here we control the sources of
    # randomness by seeding the various random number generators used in this Colab
    # For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(5)  # Torch
    random.seed(5)  # Python
    np.random.seed(5)  # NumPy

    dataset = torch.load(file_path)[: (cfg.train_size + cfg.test_size)]

    if cfg.shuffle:
        random.shuffle(dataset)

    stats_list = get_stats(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(device)

    (
        test_losses,
        losses,
        velo_val_losses,
        best_model,
        best_test_loss,
        test_loader,
    ) = train(dataset, device, stats_list, args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))
    if cfg.save_velo_val:
        print("Minimum velocity validation loss: {0}".format(min(velo_val_losses)))

    save_plots(args, losses, test_losses, velo_val_losses)


if __name__ == "__main__":
    for args in [
        {
            "model_type": "meshgraphnet",
            "num_layers": 10,
            "batch_size": 16,
            "hidden_dim": 10,
            "epochs": 5000,
            "weight_decay": 5e-4,
            "lr": 0.001,
            "train_size": 45,
            "test_size": 10,
            "device": "cuda",
            "shuffle": True,
            "save_velo_val": True,
            "save_best_model": True,
            "checkpoint_dir": "./data/checkpoints/",
            "postprocess_dir": "./data/2d_loss_plots/",
        },
    ]:
        args = objectview(args)

    load_train_plot(args)

    # larger training set
    args.train_size = 85
    args.test_size = 15
    load_train_plot(args)

    # # load model.
    # args.device = torch.device('cpu') # animation function cannot work with data on GPU
    # num_node_features = dataset[0].x.shape[1]
    # num_edge_features = dataset[0].edge_attr.shape[1]
    # num_classes = 2 # the dynamic variables have the shape of 2 (velocity)
    # PATH = os.path.join( checkpoint_dir, 'model_nl10_bs16_hd10_ep5000_wd0.0005_lr0.001_shuff_True_tr85_te15_new.pt')
    # model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
    #                             args).to(args.device)

    # model.load_state_dict(torch.load(PATH, map_location=args.device))

    # # visualize predicted velocities
    # animation_name = 'x_velocity'
    # args.postprocess_dir = postprocess_dir

    # eval_data_loader = visualize(dataset, model, args.postprocess_dir, args, animation_name, stats_list,
    #         delta_t = 0.01, skip = 1)
