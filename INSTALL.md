## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. It **will not** work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0


### Environment Set Up

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name DRG python=3.6
conda activate DRG
conda install -c psi4 gcc-5

conda install ipython pip
pip install ninja yacs cython matplotlib tqdm opencv-python
conda install pytorch-nightly=1.0.0 torchvision=0.2.2 cudatoolkit=10.0 -c pytorch
pip install tensorboardX tensorboard scipy requests Pillow==6.1 sklearn pandas scikit-image

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
# Need to use this version
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
# For GPU usage
python setup.py install --cuda_ext --cpp_ext
# For CPU usage
python setup.py install --cpp_ext

# install DRG
cd $INSTALL_DIR
git clone https://github.com/vt-vl-lab/DRG.git
cd DRG && make
```
