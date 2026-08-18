
from timm.data import create_transform

import torch
import numpy as np
import torch.distributed as dist

from typing import Iterable
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info

from .datasets import ExpressionDataset, HeadPoseDataset, LandmarkDataset
from .samplers import SubsetRandomSampler


def get_analysis_train_dataloader(data_choose, config) -> Iterable:

    if data_choose == "landmarks":
        batch_size = config.landmark_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = LandmarkDataset(config=config, choose="train", transform=transform)

    elif data_choose == "expression":
        batch_size = config.expression_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = ExpressionDataset(config=config, choose="train", transform=transform)

    elif data_choose == "headpose":
        batch_size = config.head_pose_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = HeadPoseDataset(config=config, choose="train", transform=transform)

    rank, world_size = get_dist_info()
    sampler_train = DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=config.train_num_workers,
        pin_memory=config.train_pin_memory,
        drop_last=True,
    )
    return data_loader_train


def get_analysis_val_dataloader(data_choose, config):

    transform = create_transform(
            input_size=config.img_size,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    
    if data_choose == "landmarks":
        dataset_val = LandmarkDataset(config=config, choose="test", transform=transform)
    elif data_choose == "expression":
        dataset_val = ExpressionDataset(config=config, choose="test", transform=transform)
    elif data_choose == "headpose":
        dataset_val = HeadPoseDataset(config=config, choose="test", transform=transform)

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=config.val_pin_memory,
        drop_last=True
    )

    return data_loader_val

