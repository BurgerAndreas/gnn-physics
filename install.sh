#!/bin/bash

# Check if in right directory
if [ ! -d "../gnn-physics/" ]; then
    echo "Please run this file from .../gnn-physics"
    exit 1
fi

# if [ $PWD/ != **/gnn-physics/ ]; then
#     echo "Please run this file from .../gnn-physics"
#     exit 1
# fi

# Download conda
echo "------------------------"
printf 'Do you want to install conda (Y/N)? Default: N'
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then 
    echo "Downloading conda..."
    mkdir -p miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O 
    miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
else
    echo "Continuing without installing conda."
fi


# Install conda environment
echo "------------------------"
printf 'Do you want to setup a new conda environment (Y/N)? Default: N'
read answer
if [ "$answer" != "${answer#[Yy]}" ]; then 
    echo "------------------------"
    echo "Setting up conda environment..."
    conda create -n mlp11 python=3.11 -y
    conda activate mlp11
    conda install -c pytorch pytorch torchvision torchaudio -y
    conda install -c conda-forge nbformat jupyter plotly matplotlib mediapy pip tqdm -y 
else
    echo "Continuing without installing conda."
fi


# Download dataset
echo "------------------------"
DOWNLOADS=./data/cylinder_flow
if [ -d "$DOWNLOADS" ]; then
    echo "Dataset already exists :)"
else 
    echo "Downloading dataset..."
    chmod +x ./data/download_dataset.sh
    bash ./data/download_dataset.sh cylinder_flow ./data/
fi


# End
echo "------------------------"
echo "Setup complete :)"
echo "------------------------"

