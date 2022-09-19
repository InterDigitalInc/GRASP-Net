#!/usr/bin/env bash
# GRASP-Net installation example
# Run "echo y | conda create -n grasp python=3.8 && conda activate grasp && ./install_torch-1.8.1+cu-11.2.sh"

# 1. Basic installation for PccAI
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==2.8.0
pip install plyfile==0.7.4
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric==2.0.3

# 2. Additional packages for GRASP-Net

# PCGCv2 for some basic utilities
cd third_party
pip install h5py==3.6.0
pip install torchac==0.9.3
pip install ninja==1.10.2.3
git clone https://github.com/NJUVISION/PCGCv2.git
cd PCGCv2
ln -s ../tmc3 ./tmc3

# nndistance for computing Chamfer Distance
cd ../nndistance
export PATH="/usr/local/cuda-11.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"
python build.py install
cd ..

# FAISS for fast nearest-neighbor search
echo y | conda install -c pytorch faiss-gpu

# MinkowskiEngine for sparse convolution
echo y | conda install openblas-devel==0.3.10 -c anaconda
export CXX=g++-7
export CUDA_HOME=/usr/local/cuda-11.2
git clone https://github.com/NVIDIA/MinkowskiEngine.git MinkowskiEngine_Py38
cd MinkowskiEngine_Py38
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ../..
