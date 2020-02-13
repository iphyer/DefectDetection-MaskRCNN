#!/usr/bin/env bash
# Ron the short-list GPU queue

#SBATCH -p batch_default
#SBATCH --account=skunkworks --qos=priority

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
module load anaconda/wml
bootstrap_conda
#conda create --name maskrcnn_benchmark10 python=3.6
conda activate maskrcnn_benchmark55
#source activate maskrcnn_benchmark2
module load cuda/10.0
module load gcc/5.1.0

#unset INSTALL_DIR
python /srv/home/yliu/maskrcnn-benchmark55/tools/train_net.py --config-file "/srv/home/yliu/maskrcnn-benchmark55/configs/defect_detection.yaml" SOLVER.IMS_PER_BATCH 1  SOLVER.BASE_LR 0.0005 SOLVER.MAX_ITER 200000 SOLVER.STEPS "(30000, 40000)"  
