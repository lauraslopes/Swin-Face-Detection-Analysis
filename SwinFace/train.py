import argparse
import logging
import os
from itertools import cycle
import math

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import build_scheduler
from partial_fc import PartialFC, PartialFCAdamW
from analysis import *
from analysis import subnets
from model import build_model

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

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
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # Recognition dataloader (commented out for head-only training)
    '''
    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )
    '''
    # train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)
    # Analysis dataloaders (only keep expression and head_pose active)
    # age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    # CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)
    head_pose_train_loader = get_analysis_train_dataloader("head_pose", cfg, args.local_rank)

    model = build_model(cfg).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()

    # For head-only training, compute total batch using only expression and head_pose
    # (keep original line commented for reference)
    # cfg.total_batch_size = world_size * (cfg.recognition_bz + cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz + cfg.head_pose_bz)
    cfg.total_batch_size = world_size * (cfg.expression_bz + cfg.head_pose_bz)
    cfg.epoch_step = len(expression_train_loader)

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    #cfg.total_recognition_bz = cfg.recognition_bz * world_size
    #cfg.warmup_step = cfg.num_image // cfg.total_recognition_bz * cfg.warmup_epoch
    #cfg.total_step = cfg.num_image // cfg.total_recognition_bz * cfg.num_epoch

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Recognition loss (commented out: training focuses on head tasks only)
    # margin_loss = CombinedMarginLoss(
    #     64,
    #     cfg.margin_list[0],
    #     cfg.margin_list[1],
    #     cfg.margin_list[2],
    #     cfg.interclass_filtering_threshold
    # )

    # Analysis task losses for the 3-task training run (expression + landmarks + head pose)
    criteria = [
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
        torch.nn.MSELoss(),
    ]

    if cfg.optimizer == "sgd":
        # PartialFC and recognition branch commented out (not training recognition)
        # module_partial_fc = PartialFC(
        #     margin_loss, cfg.embedding_size, cfg.num_classes,
        #     cfg.sample_rate, cfg.fp16)
        # module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        # PartialFC and recognition branch commented out (not training recognition)
        # module_partial_fc = PartialFCAdamW(
        #     margin_loss, cfg.embedding_size, cfg.num_classes,
        #     cfg.sample_rate, cfg.fp16)
        # module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
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
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"],
                                              strict=False)  # only load backbone!
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])                                  
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch = start_epoch+1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])  # commented: not using recognition
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    #FGNet_loader = get_analysis_val_dataloader(data_choose="FGNet", config=cfg)
    #LAP_loader = get_analysis_val_dataloader(data_choose="LAP", config=cfg)
    #CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    expression_test_loader = get_analysis_val_dataloader(data_choose="expression", config=cfg)
    head_pose_test_loader = get_analysis_val_dataloader(data_choose="head_pose", config=cfg)

    #FGNet_verification = FGNetVerification(data_loader=FGNet_loader, summary_writer=summary_writer)
    #LAP_verification = LAPVerification(data_loader=LAP_loader, summary_writer=summary_writer)
    #CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer)
    expression_verification = RAFVerification(data_loader=expression_test_loader, summary_writer=summary_writer)
    head_pose_verification = AFLWVerification(data_loader=head_pose_test_loader, summary_writer=summary_writer)

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
        analysis_task_names=["Expression", "Landmarks", "Pose"]
    )

    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for _ in range(3)]

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    '''
    with torch.no_grad():
        model.module.set_output_type("Recognition")
        callback_verification(global_step, model)
        model.module.set_output_type("Age")
        FGNet_verification(global_step, model)
        LAP_verification(global_step, model)
        model.module.set_output_type("Attribute")
        CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        RAF_verification(global_step, model)
    '''

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(expression_train_loader, DataLoader):
            expression_train_loader.sampler.set_epoch(epoch)
        if isinstance(head_pose_train_loader, DataLoader):
            head_pose_train_loader.sampler.set_epoch(epoch)

        # iterate only over expression and head_pose dataloaders
        for idx, data in enumerate(zip(expression_train_loader, head_pose_train_loader)):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            RAF = data[0]
            AFLW = data[1]

            expression_img, expression_label = RAF
            head_pose_img, head_pose_label = AFLW

            expression_label = expression_label.cuda(non_blocking=True)
            head_pose_label = head_pose_label.cuda(non_blocking=True)

            analysis_labels = [
                expression_label,
                head_pose_label[:, :42],
                head_pose_label[:, 42:]
            ]

            img = torch.cat([expression_img, head_pose_img], dim=0).cuda(non_blocking=True)

            model.module.set_output_type("List")
            outputs = model(img)

            # Recognition branch commented out (not training recognition)
            #recognition_loss = torch.tensor(0.0, device=img.device)

            analysis_losses = []

            expr_count = expression_img.size(0)
            aflwf_count = head_pose_img.size(0)
            aflwf_start = expr_count
            aflwf_end = expr_count + aflwf_count

            # Expression classification on expression batch
            expr_output = outputs[0][:expr_count]
            analysis_losses.append(criteria[0](expr_output, analysis_labels[0]))

            # Landmark regression on AFLW batch only
            landmarks_output = outputs[1][aflwf_start:aflwf_end]
            analysis_losses.append(criteria[1](landmarks_output, analysis_labels[1]))

            # Pose regression on AFLW batch only
            pose_output = outputs[2][aflwf_start:aflwf_end]
            analysis_losses.append(criteria[2](pose_output, analysis_labels[2]))

            loss = torch.tensor(0.0, device=img.device)
            loss_weights = cfg.analysis_loss_weights[:3] if len(cfg.analysis_loss_weights) >= 3 else [1.0, 1.0, 1.0]
            for j in range(3):
                loss += analysis_losses[j] * loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                #recognition_loss_am.update(recognition_loss.item(), 1)
                for j in range(3):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                callback_logging(global_step, loss_am, recognition_loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step)[0], amp)

                if (global_step+1) % cfg.verbose == 0:
                    # Recognition and age/attribute verifications commented for head-only training
                    # model.module.set_output_type("Recognition")
                    # callback_verification(global_step, model)
                    # model.module.set_output_type("Age")
                    # FGNet_verification(global_step, model)
                    # LAP_verification(global_step, model)
                    # model.module.set_output_type("Attribute")
                    # CelebA_verification(global_step, model)
                    model.module.set_output_type("Expression")
                    expression_verification(global_step, model)
                    model.module.set_output_type("List")
                    head_pose_verification(global_step, model)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    # "state_dict_softmax_fc": module_partial_fc.state_dict(),  # commented: not using recognition
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            # train_loader.reset()  # commented out for head-only training (recognition loader disabled)
            pass

    with torch.no_grad():
        # Recognition and age/attribute verification disabled for head-only training
        # model.module.set_output_type("Recognition")
        # callback_verification(global_step, model)
        # model.module.set_output_type("Age")
        # FGNet_verification(global_step, model)
        # LAP_verification(global_step, model)
        # model.module.set_output_type("Attribute")
        # CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        expression_verification(global_step, model)
        model.module.set_output_type("List")
        head_pose_verification(global_step, model)

    # if rank == 0:
    # path_module = os.path.join(cfg.output, "model.pt")
    # torch.save(backbone.module.state_dict(), path_module)

    # from torch2onnx import convert_onnx
    # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
