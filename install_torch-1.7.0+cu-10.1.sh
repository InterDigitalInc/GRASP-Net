#!/usr/bin/env bash
# GRASP-Net installation example
# Run "echo y | conda create -n grasp python=3.6 && conda activate grasp && ./install_torch-1.7.0+cu-10.1.sh"

# 1. Basic installation for PccAI
echo y | conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install tensorboard==2.9.0
pip install plyfile==0.7.4
pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric==2.0.3

# 2. Additional packages for GRASP-Net

# PCGCv2 for some basic utilities
cd third_party
pip install h5py==3.1.0
pip install torchac==0.9.3
git clone https://github.com/NJUVISION/PCGCv2.git
cd PCGCv2
ln -s ../tmc3 ./tmc3

# nndistance for computing Chamfer Distance
cd ../nndistance
export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
python build.py install
cd ../..

# FAISS for fast nearest-neighbor search
echo y | conda install -c pytorch faiss-gpu

# MinkowskiEngine for sparse convolution
export CXX=g++-7
echo y | conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
