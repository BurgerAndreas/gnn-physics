# gnn-physics
Learning Mesh-Based Simulation with Graph Networks

https://arxiv.org/abs/2010.03409

Website
https://sites.google.com/view/meshgraphnets

Code and dataset
https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

Helpful
https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d

## Get started

```
conda create -n mlp11 python=3.11 -y
conda activate mlp11
conda install -c pytorch pytorch torchvision torchaudio -y
conda install -c conda-forge nbformat jupyter plotly matplotlib mediapy pip tqdm -y
```

Download dataset
```
chmod +x ./data/download_dataset.sh
bash ./data/download_dataset.sh cylinder_flow ./data/
```


