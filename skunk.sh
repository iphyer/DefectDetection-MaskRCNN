#!/usr/bin/env bash
# Ron the short-list GPU queue

#SBATCH -p sbel_cmg
#SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 3-2:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
source /srv/home/yliu/.bashrc
#conda create -n mxnet2 python=3.6
#conda activate mxnet2
#source activate mxnet2
#which python
## Load CUDA into your environment

# this installs the right pip and dependencies for the fresh python
#conda install ipython
#
## maskrcnn_benchmark and coco api dependencies
#pip install ninja yacs cython matplotlib tqdm
#pip install -U scikit-image
#pip install -U cython
#pip install opencv-python
#
#pip install numpy
#pip install --upgrade mxnet-cu100 gluoncv
#pip install --pre --upgrade mxnet-cu100
#pwd

#cd ~/mxnet/gluon
#pwd 
#python setup.py install --user
#cd ~
##export INSTALL_DIR=$PWD
## install pycocotools
##cd $INSTALL_DIR
#git clone https://github.com/cocodataset/cocoapi.git
#cd cocoapi/PythonAPI
#python setup.py build_ext install

#python /srv/home/yliu/mxnet/train_mask_rcnn.py --gpus 0 --save-prefix "~/mxnet/data/"

###################################################################
# This should be un-comment when first created virtual environment
###################################################################
conda create --name maskrcnn_benchmark001 python=3.6
conda activate maskrcnn_benchmark001

conda install -c anaconda -n maskrcnn_benchmark001 ncurses
#source activate maskrcnn_benchmark2
module load cuda/10.0
module load gcc/4.9.2
conda install -c conda-forge yacs
pip install ninja cython matplotlib tqdm opencv-python
pip install Pillow
conda install pytorch=1.0.0 cuda100 -c pytorch 
conda install -c  pytorch-nightly=1.0.0 torchvision=0.2.2 cudatoolkit=10.0
pip install torchvision==0.2.2
#conda install torchvision
#conda install pytorch=1.0.0 -c pytorch cudatoolkit=10.0
#conda install -c pytorch pytorch-nightly torchvision cudatoolkit=10.0
#conda install pytorch cudatoolkit=10.0 -c pytorch-nightly

# this installs the right pip and dependencies for the fresh python
#conda install ipython
#module load gcc/4.9.2
#module load cuda/10.0
# maskrcnn_benchmark and coco api dependencies
#pip install ninja yacs cython matplotlib tqdm opencv-python
#conda install pytorch cudatoolkit=10.0 -c pytorch-nightly
# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
#conda install -c pytorch=1.0.0 pytorch-nightly=1.0.0 torchvision=0.2.2 cudatoolkit=10.0
#conda install -c pytorch pytorch-nightly=1.0.0 torchvision cudatoolkit=10.0
#conda install -c pytorch pytorch-nightly torchvision cudatoolkit=10.0
#pip install torchvision==0.2.2
#export INSTALL_DIR=$PWD

# install pycocotools
#cd $INSTALL_DIR
cd /srv/home/yliu/maskrcnn-benchmark001/cocoapi/PythonAPI
python setup.py build_ext install

# install apex
#cd /srv/home/yliu/maskrcnn-benchmark
#git clone https://github.com/NVIDIA/apex.git
cd /srv/home/yliu/maskrcnn-benchmark001/apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
#cd $INSTALL_DIR
#git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd /srv/home/yliu/maskrcnn-benchmark001/

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python /srv/home/yliu/maskrcnn-benchmark001/setup.py build develop


#unset INSTALL_DIR
python /srv/home/yliu/maskrcnn-benchmark001/tools/train_net.py --config-file "/srv/home/yliu/maskrcnn-benchmark001/configs/defect_detection.yaml" SOLVER.IMS_PER_BATCH 1  SOLVER.BASE_LR 0.0005 SOLVER.MAX_ITER 200000 SOLVER.STEPS "(30000, 40000)"  
