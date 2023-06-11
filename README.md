# Install_conda_enviroment\

#Install tensorflow:

conda create --name tf python=3.9

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow

conda create --name tf_gpu python=3.6

conda install -c anaconda tensorflow-gpu


conda install -c anaconda pillow

conda install -c menpo imageio

conda install -c anaconda scikit-image

conda install -c conda-forge opencv

conda install -c conda-forge ffmpeg

conda install -c anaconda ipython

conda install -c conda-forge matplotlib

pip install Keras==2.3.1

conda install -c anaconda scikit-learn

# Install Pytorch 

conda create -n torch_gpu python=3.7

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly


conda install -c conda-forge matplotlib

conda install -c conda-forge opencv

conda install -c anaconda pillow

conda install -c conda-forge vim

conda install -c anaconda h5py

conda install -c conda-forge imageio 

conda install -c conda-forge tqdm

conda install -c anaconda scikit-image 

conda install -c conda-forge timm

conda install -c anaconda scikit-learn


# Check pytorch

python

import torch

torch.cuda.is_available()

python3 -c 'import torch; print(torch.cuda.is_available())'


# Install tf-nightly for RXT3090, A100. (2021)


conda create --name tf_gpu python=3.8

pip install tf-nightly-gpu==2.6.0-dev20210625 

pip install tensorflow-gpu==2.6 # 2022

pip install scikit-learn

pip install matplotlib

pip install ipython

pip install opencv-python

pip install scikit-image

pip install imageio

sudo apt install ffmpeg

python3 -c 'import tensorflow as tf; print(tf.__version__)'



# Note to install inplace-abn
 conda create --name multilabel2 -y
conda activate  multilabel2

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 export CUDA_HOME=$CONDA_PREFI
sudo apt-get --yes install build-essential
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
sudo apt-get --purge remove gcc
 sudo apt-get install --reinstall gcc
sudo apt-get update
 sudo apt-get install g++
pip install inplace-abn
