#!/bin/bash
#$ -l cluster=andrena
#$ -l h_rt=20:00:0
#$ -l h_vmem=7.5G
#$ -pe smp 8
#$ -j y
#$ -N fold2
#$ -l gpu=1
#$ -wd /data/home/mpx588/logs

module load miniforge
conda activate dl
# unset CUDA_VISIBLE_DEVICES
echo "Checking for /dev/nvidia* devices:"
ls -l /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found"
python /data/EECS-LITQ/fran_storage/code/fran/fran/run/train.py \
  --project nodes \
  --plan-num 7 \
  --fold 2 \
  --bs 2 \
  --epochs 600 \
  --compiled false\
  --profiler false \
  --neptune true \
  --cache-rate 0.0
                       
