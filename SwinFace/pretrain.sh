#!/usr/bin/env bash
export SWINFACE_WATCHDOG_TIMEOUT_MINUTES=30
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 pretrain.py configs/config_pretrain
