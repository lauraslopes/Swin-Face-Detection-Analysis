#!/usr/bin/env bash

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py configs/config_train
