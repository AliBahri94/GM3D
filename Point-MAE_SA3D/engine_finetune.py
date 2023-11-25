# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import builtins

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from pointnet2_ops import pointnet2_utils
import numpy as np
from torchvision import transforms
from datasets import data_transforms


train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def mixup_point_cloud(data, alpha, labels, device):
    """
    Apply mixup augmentation to a batch of point clouds.

    Args:
    - data (torch.Tensor): Point cloud data of shape (batch_size, num_points, num_features).
    - alpha (float): Mixup interpolation coefficient.
    - labels (torch.Tensor): Labels associated with the point cloud data.

    Returns:
    - Mixed point cloud data.
    - Mixed labels.
    """

    # Compute mixup ratio
    lam = np.random.beta(alpha, alpha)

    # Randomly shuffle data and labels
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    mixed_data = lam * data + (1 - lam) * shuffled_data

    # Convert labels to one-hot encoding
    num_classes = 15
    labels_onehot = torch.zeros(batch_size, num_classes).to(device).scatter_(1, labels.view(-1, 1), 1).to(device)
    shuffled_labels_onehot = torch.zeros(batch_size, num_classes).to(device).scatter_(1, shuffled_labels.view(-1, 1), 1).to(device)

    # Mix the labels
    mixed_labels_onehot = lam * labels_onehot + (1 - lam) * shuffled_labels_onehot

    return mixed_data, mixed_labels_onehot


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, npoints = 0):
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (taxonomy_ids, model_ids, data) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = data[0].cuda()
        targets = data[1].cuda()
        #samples = samples.to(device, non_blocking=True)
        #targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        ####################
        
        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)

        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
        # import pdb; pdb.set_trace()
        points = train_transforms(points)
        #points, targets = mixup_point_cloud(points, 0.2, targets, device)
        ####################

        with torch.cuda.amp.autocast():
            outputs = model(points)
            targets = targets.long()                  ### just for modelnet
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        #loss.backward()
        #optimizer.step()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}