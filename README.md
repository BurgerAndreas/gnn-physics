# gnn-physics

Simple re-implementation of [Learning Mesh-Based Simulation with Graph Networks](https://sites.google.com/view/meshgraphnets) for `cylinder_flow` in PyTorch.

The [original codebase](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) contains for the `cylinder_flow` and `flag_simple` domains.
It does not contain the prediction of the sizing field and the corresponding remesher.
Out of time constraints we do not implement the sizing field and remesher either, but we leave some pseudo-code and ideas how to build both in `/sizing_field`.


## Installation

Get the code
```bash
git clone git@github.com:BurgerAndreas/gnn-physics.git
# git clone https://github.com/BurgerAndreas/gnn-physics.git
```
Setup Conda environment
```bash
conda create -n mlp11 python=3.11 -y
conda activate mlp11
conda install -c conda-forge nbformat jupyter plotly matplotlib mediapy pip tqdm gdown -y

pip3 install torch torchvision torchaudio torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric torchdata black
pip3 install tensorflow tensorrt protobuf==3.20.3
```

### Delete later
---
conda activate /home/andreasburger/miniconda3/envs/mlp11

nvcc --version # cuda 12.3
conda update -n base -c conda-forge conda
conda install conda=23.11.0

pip3 install torch torchvision torchaudio torch_scatter torch_sparse torch_cluster torch_spline_conv

conda install -c conda-forge gdown

conda remove --force protobuf
pip3 install protobuf==3.20.3

pip install black
black stanford_simple.py

wandb.init(mode="disabled") or by setting WANDB_MODE=disable
---

### Dataset
The `cylinder_flow` dataset contains 1,200 trajectories with 600 timesteps each.
The data is in `.tfrecord` format. `.tfrecord` is highly optimized, but only works with Tensorflow and can be hard to handle.

We simplify the dataset to 3 trajectories.
We save the data as numpy arrays in a `.hdf5` file.
We provide the 3 trajectories in this github repo, you do not need to do anything.

If you want to download the original `.tfrecord` dataset (16 GB)
```bash
chmod +x ./data/datasets/download_dataset.sh
bash ./data/datasets/download_dataset.sh cylinder_flow ./data/datasets
```
If you want to convert the `.tfrecord` dataset to numpy in `.hdf5`
```bash
conda activate meshgnn
# -num_traj -1 means convert all trajectories
python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/train' -out 'data/datasets/cylinder_flow_hdf5/train' --num_traj 3 
```


## Ressources

- Original [Paper](https://arxiv.org/abs/2010.03409)
|
[Website](https://sites.google.com/view/meshgraphnets)
|
[Code (Tensorflow)](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- [Helpful explanation](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d)

Follow-up papers
- [MultiScale MeshGraphNets, 2022](https://arxiv.org/abs/2210.00612)
- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
- [Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a.html)
- [Learned Coarse Models for Efficient Turbulence Simulation, 2022](https://arxiv.org/abs/2112.15275)