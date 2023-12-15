# Meshed-based fluid simulation with GNNs

Re-implementation of [Learning Mesh-Based Simulation with Graph Networks](https://sites.google.com/view/meshgraphnets) for `cylinder_flow` in PyTorch based on [this blog post](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d).

Look at results.md for a summary!

## Installation

Get the code
```bash
git clone git@github.com:BurgerAndreas/gnn-physics.git
# git clone https://github.com/BurgerAndreas/gnn-physics.git
```
Setup Conda environment
(Tested on Ubuntu 22.04, RTX 3060, Cuda 12.3)
```bash
conda create -n meshgnn python=3.11 -y
conda activate meshgnn
pip3 install -r requirements.txt

# or try by hand
conda install -c conda-forge nbformat jupyter plotly matplotlib mediapy pip tqdm gdown -y
pip3 install torch torchvision torchaudio torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric torchdata 
pip3 install black hydra-core
pip3 install tensorflow tensorrt protobuf==3.20.3
```

## Run
Download the small dataset (1GB) from google drive
```bash
cd data/datasets/cylinder_flow_pyg/
# https://drive.google.com/file/d/1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt" -O data_pt.tgz && rm -rf /tmp/cookies.txt

# Unzip the data
tar -zxvf data_pt.tgz
```

Run the code
```bash
cd ../../..
# Run with default settings
python run_gnn.py
# Try additional configs like this:
python run_gnn.py +noise=paper +datasize=small
# Plot training loss & predictions from loaded checkpoints like this:
python plot.py +datasize=medium
python animate.py +datasize=medium
```


## Dataset
The original `cylinder_flow` dataset contains 1,200 trajectories with 600 timesteps each.
The data is in the `.tfrecord` format. `.tfrecord` is highly optimized, but only works with Tensorflow and can be hard to handle.

I simplified the dataset to 4 trajectories (3 train, 1 test) saved as numpy arrays in a `.hdf5` file.
The 4 trajectories are provided via the google drive link avove.

#### Optional: get more data
If you want to download the full original `.tfrecord` dataset for `cylinder_flow` (16 GB)
```bash
chmod +x ./data/datasets/download_dataset.sh
bash ./data/datasets/download_dataset.sh cylinder_flow ./data/datasets
```
If you want to convert the `.tfrecord` dataset to numpy in `.hdf5`
```bash
conda activate meshgnn
# -num_traj -1 means convert all trajectories
python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/train' -out 'data/datasets/cylinder_flow_hdf5/train' --num_traj 3 
python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/test' -out 'data/datasets/cylinder_flow_hdf5/test' --num_traj 1
```
If you want to convert the `.hdf5` dataset to PyTorch graphs `.pt`
```bash
conda activate meshgnn
python ./data/datasets/hdf5_to_pyg.py -in 'data/datasets/cylinder_flow_hdf5/train.hdf5' -out 'data/datasets/cylinder_flow_pyg/train.pt'
python ./data/datasets/hdf5_to_pyg.py -in 'data/datasets/cylinder_flow_hdf5/test.hdf5' -out 'data/datasets/cylinder_flow_pyg/test.pt'
```

#### Optional: get prior blog data
I basde my code on this [blog post](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d) 
which ships with some data in `.pt` format.
Sadly they did not include the code they used to transform the data.
My code still works on their data. 
In practice their data performs worse than my data conversion, for unknown reasons.

If you want to download their data:
```bash
python ./data/datasets/download_pyg_stanford_data.py
```


## Future Work

The [original codebase of the paper](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) implements the `cylinder_flow` and `flag_simple` domains.
- [ ] Change `./data/datasets/hdf5_to_pyg.py` to work with all datasets with different features

The original codebase also does not contain the prediction of the sizing field and the corresponding remesher.
Out of time constraints we do not implement the sizing field prediction + remesher either.
To implement sizing field prediction:
- [ ] Build prediction head and combine with existing GNN 
- [ ] Build sizing-based remesher (pseudo-code can be found in [this paper](http://graphics.berkeley.edu/papers/Narain-AAR-2012-11/Narain-AAR-2012-11.pdf) and [A3 of the original paper](https://arxiv.org/abs/2010.03409))
- [ ] Adapt training loop to learn sizing field prediction on `flag_dynamic_sizing` 
(Only the `flag_dynamic_sizing` (36 GB) and `sphere_dynamic_sizing` datasets include the necessary data to learn the sizing field)

## Ressources

- Original [Paper](https://arxiv.org/abs/2010.03409)
|
[Website](https://sites.google.com/view/meshgraphnets)
|
[Code (Tensorflow)](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- My code is based on this [blog](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d)
|
[Code (PyTorch)](https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing)

Follow-up papers
- [MultiScale MeshGraphNets, 2022](https://arxiv.org/abs/2210.00612)
- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
- [Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a.html)
- [Learned Coarse Models for Efficient Turbulence Simulation, 2022](https://arxiv.org/abs/2112.15275)
