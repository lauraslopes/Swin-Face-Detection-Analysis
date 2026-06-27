_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/wider_face.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=4))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)})))

optimizer_config = dict(grad_clip=None)
