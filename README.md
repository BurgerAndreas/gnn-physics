# gnn-physics

Simple re-implementation of [Learning Mesh-Based Simulation with Graph Networks](https://sites.google.com/view/meshgraphnets) for `cylinder_flow` in PyTorch.

The [original codebase](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) contains for the `cylinder_flow` and `flag_simple` domains.
It does not contain the prediction of the sizing field and the corresponding remesher.
Out of time constraints we do not implement the sizing field and remesher either.


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

pip3 install torch torchvision torchaudio torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric torchdata 

pip3 install black hydra-core
pip3 install tensorflow tensorrt protobuf==3.20.3
```

## Run

Unzip the data
```bash
cd data/datasets/cylinder_flow_pyg/
# tar -zcvf data_pt.tgz train.pt test.pt
tar -zxvf data_pt.tgz
```
Run the code
```bash
cd $ROOT_DIR
python run_gnn.py
# Try additional configs like this:
python run_gnn.py +noise=paper +datasize=small
```


## Dataset
The `cylinder_flow` dataset contains 1,200 trajectories with 600 timesteps each.
The data is in `.tfrecord` format. `.tfrecord` is highly optimized, but only works with Tensorflow and can be hard to handle.

I simplified the dataset to 4 trajectories (3 train, 1 test).
I save the data as numpy arrays in a `.hdf5` file.
I provide the 4 trajectories in this github repo.

#### Optional: more data
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

#### Optional: prior blog data
I base my code on this [blog post](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d) 
which ships with some transformed data in `.pt` format.
Sadly they do not include the code used to transform the data.
My code still works on their data. 
In practice their data performs worse than my data conversion, for unknown reasons.

If you want to download their data:
```bash
python ./data/datasets/download_pyg_stanford_data.py
```


## Future Work
[] Change `./data/datasets/hdf5_to_pyg.py` to work with any kind of dataset and their features
Implement sizing field prediction
[] Build prediction head and combine with existing GNN 
[] Build sizing-based remesher (pseudo-code can be found in [this paper](http://graphics.berkeley.edu/papers/Narain-AAR-2012-11/Narain-AAR-2012-11.pdf) and [A3 of the original paper](https://arxiv.org/abs/2010.03409))
[] Adapt training loop to learn sizing field prediction on `flag_dynamic_sizing` (Only the `flag_dynamic_sizing` (36 GB) and `sphere_dynamic_sizing` datasets include the necessary data to learn the sizing field)

## Ressources

- Original [Paper](https://arxiv.org/abs/2010.03409)
|
[Website](https://sites.google.com/view/meshgraphnets)
|
[Code (Tensorflow)](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- My code is heavily based on this [blog](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d)
|
[Code (PyTorch)](https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing)

Follow-up papers
- [MultiScale MeshGraphNets, 2022](https://arxiv.org/abs/2210.00612)
- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
- [Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a.html)
- [Learned Coarse Models for Efficient Turbulence Simulation, 2022](https://arxiv.org/abs/2112.15275)