from timm.utils import accuracy, AverageMeter

import logging
import time
import torch
import torch.distributed as dist

from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class LimitedAvgMeter(object):

    def __init__(self, max_num=10, best_mode="max"):
        self.avg = 0.0
        self.num_list = []
        self.max_num = max_num
        self.best_mode = best_mode
        self.best = 0.0 if best_mode == "max" else 100.0

    def append(self, x):
        self.num_list.append(x)
        len_list = len(self.num_list)
        if len_list > 0:
            if len_list < self.max_num:
                self.avg = sum(self.num_list) / len_list
            else:
                self.avg = sum(self.num_list[(len_list - self.max_num):len_list]) / self.max_num

        if self.best_mode == "max":
            if self.avg > self.best:
                self.best = self.avg
        elif self.best_mode == "min":
            if self.avg < self.best:
                self.best = self.avg

class LandmarkVerification(object):

    def __init__(self, data_loader, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.best_nme = float("inf")

        self.data_loader = data_loader
        self.summary_writer = summary_writer

        self.limited_meter = LimitedAvgMeter(best_mode="min")

    def _compute_nme(self, predictions, targets):
        """
        Compute Normalized Mean Error (NME) for landmark points.
        
        Args:
            predictions: (B, 10) tensor of predicted landmark coordinates (x1, y1, ..., x5, y5)
            targets: (B, 10) tensor of target landmark coordinates
            
        Returns:
            nme: scalar tensor representing the normalized mean error
        """
        # Reshape to (B, 5, 2) - 5 landmarks with (x, y) coordinates
        predictions_reshaped = predictions.reshape(-1, 5, 2)
        targets_reshaped = targets.reshape(-1, 5, 2)
        
        # Compute Euclidean distance for each landmark point
        # (B, 5)
        distances = torch.sqrt(torch.sum((predictions_reshaped - targets_reshaped) ** 2, dim=2))
        
        # Compute interocular distance (distance between left eye and right eye)
        # Assuming landmarks are: [left_eye, right_eye, nose, left_mouth, right_mouth]
        # Index 0: left eye, Index 1: right eye
        interocular_dist = torch.sqrt(
            torch.sum((targets_reshaped[:, 0, :] - targets_reshaped[:, 1, :]) ** 2, dim=1)
        )  # (B,)
        
        # Normalize by interocular distance and compute mean
        # Avoid division by zero
        interocular_dist = torch.clamp(interocular_dist, min=1e-8)
        normalized_distances = distances / interocular_dist.unsqueeze(1)  # (B, 5)
        
        # Compute mean NME across all landmarks and samples
        nme = torch.mean(normalized_distances)
        
        return nme

    def ver_test(self, model, global_step):

        logging.info("Val on CelebA landmarks:")

        nme_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, targets) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)

            if isinstance(targets, (list, tuple)):
                target = torch.stack([t.float() for t in targets], dim=0).cuda(non_blocking=True)
            else:
                target = targets.float().cuda(non_blocking=True)

            analysis_outputs = model(img)

            # Compute NME
            nme = self._compute_nme(analysis_outputs, target)
            nme = reduce_tensor(nme)

            nme_meter.update(nme.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Landmark NME {nme_meter.val:.4f} ({nme_meter.avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB')

        if nme_meter.avg < self.best_nme:
            self.best_nme = nme_meter.avg

        self.limited_meter.append(nme_meter.avg)

        if self.rank == 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar("CelebA Landmark NME", nme_meter.avg, global_step)
            self.summary_writer.add_scalar("CelebA Landmark NME-Best", self.best_nme, global_step)
            self.summary_writer.add_scalar("CelebA Landmark 10 Times Avg NME", self.limited_meter.avg, global_step)

            logging.info('[%d]CelebA Landmark NME: %1.5f' % (global_step, nme_meter.avg))
            logging.info('[%d]CelebA Landmark NME-Best: %1.5f' % (global_step, self.best_nme))
            logging.info('[%d]10 Times Landmark NME: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Landmark NME-Best: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        model.eval()
        self.ver_test(model, num_update)
        model.train()


class HeadPoseVerification(object):

    def __init__(self, data_loader, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.best_pose_mae: float = 100.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="min")

    def ver_test(self, model, global_step):

        logging.info("Val on AFLW:")

        pose_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            pose_target = target.float().cuda(non_blocking=True)

            pose_pred = model(img)

            pose_error = torch.mean(torch.abs(pose_pred - pose_target))
            pose_error = reduce_tensor(pose_error) #calculate the mean of the three euler angles
            pose_meter.update(pose_error.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Pose MAE {pose_meter.val:.3f} ({pose_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        if pose_meter.avg < self.best_pose_mae:
            self.best_pose_mae = pose_meter.avg

        self.limited_meter.append(pose_meter.avg / 2.0)

        if self.rank == 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="AFLW Pose MAE", scalar_value=pose_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="AFLW Pose MAE-Best", scalar_value=self.best_pose_mae, global_step=global_step)
            self.summary_writer.add_scalar(tag="AFLW 10 Times Avg MAE", scalar_value=self.limited_meter.avg, global_step=global_step)
            logging.info('[%d]Pose MAE: %1.5f' % (global_step, pose_meter.avg))
            logging.info('[%d]Pose MAE-Best: %1.5f' % (global_step, self.best_pose_mae))
            logging.info('[%d]10 Times Avg MAE: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Avg MAE-Best: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        model.eval()
        self.ver_test(model, num_update)
        model.train()


class ExpressionVerification(object):

    def __init__(self, data_loader, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc1: float = 0.0
        self.highest_acc3: float = 0.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="max")

    def ver_test(self, model, global_step):

        logging.info("Val on RAF:")

        criterion = torch.nn.CrossEntropyLoss()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc3_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            expression_output = model(img)

            loss = criterion(expression_output, target)
            acc1, acc3 = accuracy(expression_output, target, topk=(1, 3))

            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc3 = reduce_tensor(acc3)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc3_meter.update(acc3.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@3 {acc3_meter.val:.3f} ({acc3_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        if acc1_meter.avg > self.highest_acc1:
            self.highest_acc1 = acc1_meter.avg
        if acc3_meter.avg > self.highest_acc3:
            self.highest_acc3 = acc3_meter.avg

        self.limited_meter.append(acc1_meter.avg)
        
        if self.rank == 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="expression loss", scalar_value=loss_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc1", scalar_value=acc1_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc3", scalar_value=acc3_meter.avg, global_step=global_step)
    
            logging.info('[%d]Expression Loss: %1.5f' % (global_step, loss_meter.avg))
            logging.info('[%d]Expression Acc@1: %1.5f' % (global_step, acc1_meter.avg))
            logging.info('[%d]Expression Acc@1-Highest: %1.5f' % (global_step, self.highest_acc1))
            logging.info('[%d]Expression Acc@3: %1.5f' % (global_step, acc3_meter.avg))
            logging.info('[%d]Expression Acc@3-Highest: %1.5f' % (global_step, self.highest_acc3))
            logging.info('[%d]10 Times Expression Acc@1: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Expression Acc@1-Highest: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        model.eval()
        self.ver_test(model, num_update)
        model.train()
