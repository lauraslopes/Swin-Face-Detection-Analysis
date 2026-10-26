import argparse
import logging
import os

from SwinFace.analysis.verification import LandmarkVerification, ExpressionVerification, HeadPoseVerification
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from lr_scheduler import build_scheduler
from analysis import *

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed
from utils.utils_watchdog import TrainingWatchdog, get_watchdog_timeout_minutes

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

class VerificationModel(torch.nn.Module):
    """
    Joins the shared backbone with the relevant Subnet(s) into a single
    nn.Module exposing forward(img) -> Tensor (or, for mode="attribute",
    a list of 40 Tensors), which is exactly the `model(img)` interface
    analysis/verification.py's *Verification classes call internally.
 
    This plays the same role that ModelBox.set_output_type(...) plays in
    train.py's full MLCA model, adapted to pretrain.py's separate
    backbone + independent Subnet architecture.
 
    mode="Landmarks"        -> FGNetVerification / LAPVerification: returns the age
                          regression tensor (age_subnet's 1st head), squeezed
                          to shape (B,) to match the target tensors.
    mode="Expression" -> RAFVerification: returns the expression logits
                          tensor (expression_subnet's 1st head), shape (B,7).
    mode="Pose"  -> CelebAVerification: returns a list of 40 tensors, in
                          the SAME order pretrain.py's training loop already
                          uses for analysis_outputs[1:41] (Table II grouping:
                          Attractive, Blurry, ..., Wearing Necktie). This
                          assumes the CelebA label file/dataset is prepared in
                          that same group order, matching what the training
                          loop already assumes when it does
                          `analysis_labels[5] = gender_label` (i.e. index 4
                          of the raw CelebA label = the Male/Gender attribute,
                          not CelebA's official column order).
    """
 
    def __init__(self, backbone, mode, landmarks_subnet=None, expression_subnet=None,
                 pose_subnet=None):
        super().__init__()
        assert mode in ("Landmarks", "Expression", "Pose")
        self.backbone = backbone
        self.mode = mode
        self.landmarks_subnet = landmarks_subnet
        self.expression_subnet = expression_subnet
        self.pose_subnet = pose_subnet
 
    def forward(self, x):
        _, global_features, _ = self.backbone(x)
 
        if self.mode == "Landmarks":
            landmarks = self.landmarks_subnet(global_features)
            return landmarks
 
        if self.mode == "Expression":
            expression = self.expression_subnet(global_features)
            return expression
 
        if self.mode == "Pose":
            pose = self.pose_subnet(global_features)
            return pose

 
try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    watchdog = TrainingWatchdog(get_watchdog_timeout_minutes(), rank)
    watchdog.start()

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    landmarks_train_loader = get_analysis_train_dataloader("landmarks", cfg)
    expression_train_loader = get_analysis_train_dataloader("expression", cfg)
    headpose_train_loader = get_analysis_train_dataloader("headpose", cfg)

    # Backbone
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, use_checkpoint=True).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16)
    backbone.train()
    backbone._set_static_graph()

    # Analysis subnets
    landmarks_subnet = Subnet(in_chans=768, num_features=512, out_num=1, out_features=10).cuda()
    headpose_subnet = Subnet(in_chans=768, num_features=512, out_num=1, out_features=3).cuda()
    expression_subnet = Subnet(in_chans=768, num_features=512, out_num=1, out_features=7).cuda()

    landmarks_subnet = torch.nn.parallel.DistributedDataParallel(
        module=landmarks_subnet, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16)
    landmarks_subnet.train()
    #landmarks_subnet._set_static_graph()

    headpose_subnet = torch.nn.parallel.DistributedDataParallel(
        module=headpose_subnet, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16)
    headpose_subnet.train()
    #headpose_subnet._set_static_graph()

    expression_subnet = torch.nn.parallel.DistributedDataParallel(
        module=expression_subnet, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16)
    expression_subnet.train()
    #expression_subnet._set_static_graph()

    # Single-callable adapters for analysis/verification.py's *Verification
    # classes, which expect one `model(img)` -> Tensor (or list) callable.
    landmarks_verification_model = VerificationModel(backbone, mode="Landmarks", landmarks_subnet=landmarks_subnet)
    headpose_verification_model = VerificationModel(backbone, mode="Pose", pose_subnet=headpose_subnet)
    expression_verification_model = VerificationModel(backbone, mode="Expression", expression_subnet=expression_subnet)

    cfg.total_batch_size = world_size * cfg.batch_size
    cfg.num_image = max(len(landmarks_train_loader.dataset), len(expression_train_loader.dataset), len(headpose_train_loader.dataset))
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step  = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    
    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Analysis task losses for the 3-task training run (expression + landmarks + head pose)
    criteria = [
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
        torch.nn.MSELoss(), #Mean Squared Error
    ]
    
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": expression_subnet.parameters()},
                    {"params": headpose_subnet.parameters()},
                    {"params": landmarks_subnet.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": expression_subnet.parameters()},
                    {"params": headpose_subnet.parameters()},
                    {"params": landmarks_subnet.parameters()},
                    ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"pretrain_start_{rank}.pt"))
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])  # only load backbone!
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"pretrain_checkpoint_epoch_{cfg.resume_epoch}_gpu_0.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        landmarks_subnet.module.load_state_dict(dict_checkpoint["state_dict_landmarks_subnet"])
        headpose_subnet.module.load_state_dict(dict_checkpoint["state_dict_headpose_subnet"])
        expression_subnet.module.load_state_dict(dict_checkpoint["state_dict_expression_subnet"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    #callback_verification = CallBackVerification(
    #    val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    #)

    landmarks_test_loader = get_analysis_val_dataloader(data_choose="landmarks", config=cfg)
    expression_test_loader = get_analysis_val_dataloader(data_choose="expression", config=cfg)
    headpose_test_loader = get_analysis_val_dataloader(data_choose="headpose", config=cfg)

    expression_verification = ExpressionVerification(data_loader=expression_test_loader, summary_writer=summary_writer)
    headpose_verification = HeadPoseVerification(data_loader=headpose_test_loader, summary_writer=summary_writer)
    landmarks_verification = LandmarkVerification(data_loader=landmarks_test_loader, summary_writer=summary_writer)

    task_names = ["Expression", "Pose", "Landmarks"]
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
        analysis_task_names=task_names
    )

    loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for _ in range(3)]

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(expression_train_loader, DataLoader):
            expression_train_loader.sampler.set_epoch(epoch)
        if isinstance(headpose_train_loader, DataLoader):
            headpose_train_loader.sampler.set_epoch(epoch)
        if isinstance(landmarks_train_loader, DataLoader):
            landmarks_train_loader.sampler.set_epoch(epoch)

        for _, data in enumerate(zip(expression_train_loader, headpose_train_loader, landmarks_train_loader)):
            watchdog.beat(f"received training batch epoch={epoch}, step={global_step}")

            logging.info(str(global_step))

            RAF = data[0]
            AFLW = data[1]
            CelebA = data[2]

            expression_img, expression_label = RAF
            head_pose_img, head_pose_label = AFLW
            landmarks_img, landmarks_label = CelebA

            expression_label = expression_label.cuda(non_blocking=True)
            head_pose_label = head_pose_label.cuda(non_blocking=True)
            landmarks_label = landmarks_label.cuda(non_blocking=True)

            img = torch.cat([expression_img, head_pose_img, landmarks_img], dim=0).cuda(non_blocking=True)

            #with torch.cuda.amp.autocast(cfg.fp16):
            # Concat images from different dataloaders
            local_features, global_features, x = backbone.module.forward_features(img)
            
            expression_features = global_features[:cfg.expression_bz]
            headpose_features = global_features[cfg.expression_bz: cfg.expression_bz + cfg.head_pose_bz]
            landmarks_features = global_features[cfg.expression_bz + cfg.head_pose_bz:cfg.expression_bz + cfg.head_pose_bz + cfg.landmark_bz]

            expression = expression_subnet(expression_features)
            headpose = headpose_subnet(headpose_features)
            landmarks = landmarks_subnet(landmarks_features)

            analysis_losses = [
                                criteria[0](expression, expression_label),
                                criteria[1](headpose, head_pose_label),
                                criteria[2](landmarks, landmarks_label),
                        ]

            loss = torch.tensor(0.0, device=img.device)
            loss_weights = cfg.analysis_loss_weights[:3]
            for j in range(3):
                loss += analysis_losses[j] * loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                for j in range(3):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                callback_logging(global_step, loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                                 lr_scheduler.get_update_values(global_step)[0], amp, None)

                if (global_step+1) % cfg.verbose == 0:
                    #with torch.no_grad():
                    #callback_verification(global_step, backbone)
                    watchdog.beat(f"starting validation epoch={epoch}, step={global_step}")
                    expression_verification(global_step, expression_verification_model)
                    headpose_verification(global_step, headpose_verification_model)
                    landmarks_verification(global_step, landmarks_verification_model)
                    watchdog.beat(f"validation complete epoch={epoch}, step={global_step}")

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if cfg.save_all_states:
            watchdog.beat(f"saving checkpoint epoch={epoch}, step={global_step}")
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_expression_subnet": expression_subnet.module.state_dict(),
                "state_dict_headpose_subnet": headpose_subnet.module.state_dict(),
                "state_dict_landmarks_subnet": landmarks_subnet.module.state_dict(),

                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"pretrain_checkpoint_epoch_{epoch}_gpu_{rank}.pt"))
            watchdog.beat(f"checkpoint complete epoch={epoch}, step={global_step}")


        #if rank == 0:
        #    path_module = os.path.join(cfg.output, "model.pt")
        #    torch.save(backbone.module.state_dict(), path_module)

    with torch.no_grad():
        #callback_verification(global_step, backbone)
        watchdog.beat("starting final validation")
        expression_verification(global_step, expression_verification_model)
        headpose_verification(global_step, headpose_verification_model)
        landmarks_verification(global_step, landmarks_verification_model)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    watchdog.beat("destroying process group")
    distributed.destroy_process_group()
    watchdog.stop()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
