# Install_conda_enviroment\

#Install tensorflow ver 2.1:
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
pip install pandas
pip install matplotlib
conda install -c conda-forge opencv
conda install -c anaconda pillow
conda install -c conda-forge vim
conda install -c anaconda h5py
conda install -c conda-forge imageio 
conda install -c conda-forge tqdm
conda install -c anaconda scikit-image 
# Check pytorch
python
import torch
torch.cuda.is_available()
