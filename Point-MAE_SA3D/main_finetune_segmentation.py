# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from util import utils
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

import yaml
from utils.logger import *
from utils import miscc, dist_utils
from sklearn.svm import SVC
from tools import builder

from engine_finetune_segmentation import train_one_epoch, evaluate
import pickle
import timm.optim.optim_factory as optim_factory
from segmentation.dataset import PartNormalDataset
import importlib


class DotDict(dict):
    """
    A dictionary that supports dot notation as well as bracket notation for access.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)


def get_args_parser():
    parser = argparse.ArgumentParser('Hard Patches Mining for Masked Image Modeling', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #parser.add_argument('--blr', type=float, default=0.00004, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='./experimets/2_PMAE_MSE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_MAE_as_Dino/mae_vit_base_patch16_dec512d8b_hpm_relative_in1k_ep200_temp_last.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--teacher', action='store_true',
                        help='Load EMA teacher for fine-tuning')
    parser.set_defaults(teacher=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--dataloader_type', type=str, default='nori',
                        help="""dataloader type, folder, nori, dpflow..""")
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./experimets/2_PMAE_MSE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_MAE_as_Dino/Segmentation_hard_patches/finetune_ShapeNetPart_new_opti_with_last_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./experimets/2_PMAE_MSE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_MAE_as_Dino/Segmentation_hard_patches/finetune_ShapeNetPart_new_opti_with_last_model',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name (for log)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #dataset_train = build_dataset(is_train=True, args=args)
    #dataset_val = build_dataset(is_train=False, args=args)

    ################ dataset for point cloud
    with open('./args_finetune_scan_hardest.pkl', 'rb') as f:
            loaded_args = pickle.load(f)

    with open("config_finetune_scan_hardest.yaml", 'r') as stream:
    #with open("config_finetune_scan_objonly.yaml", 'r') as stream:
    #with open("config_finetune_scan_objbg.yaml", 'r') as stream:
    #with open("finetune_modelnet.yaml", 'r') as stream:
        loaded_config = yaml.safe_load(stream)

    loaded_config = DotDict(loaded_config)
    """(train_sampler, data_loader_train), (_, data_loader_val),= builder.dataset_builder(loaded_args, loaded_config.dataset.train), \
                                                            builder.dataset_builder(loaded_args, loaded_config.dataset.val)
    npoints = loaded_config.npoints
    val_writer = SummaryWriter(os.path.join(args.output_dir, 'test'))
    logger = get_logger("pretrain")"""

    root = "/export/livia/home/vision/Abahri/projects/Point-MAE/Point-MAE-org/Point-MAE/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/"
    TRAIN_DATASET = PartNormalDataset(root= root, npoints= 2048, split='trainval', normal_channel= False)
    data_loader_train = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root= root, npoints= 2048, split='test', normal_channel= False)
    data_loader_val = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50
    ########################################

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        #sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        pass

    global_rank = 0  
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    """data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )"""

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    """model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )"""

    #model = builder.model_builder(loaded_config.model)
    ############### load model
    MODEL = importlib.import_module("pt")
    #shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0
    ##########################

    if args.finetune:
        # load pretrained model
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        classifier.load_model_from_ckpt(args.ckpts)
        """checkpoint = torch.load(args.finetune, map_location='cpu')

        if args.teacher:
            key = 'ema_state_dict'
        else:
            key = 'state_dict' if 'state_dict' in checkpoint else 'model'
        print("Load checkpoint[{}] for fine-tuning".format(key))
        checkpoint_model = checkpoint[key]

        state_dict = model.state_dict()
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}"""

        """for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]"""

        # interpolate position embedding
        """interpolate_pos_embed(model, checkpoint_model)"""

        # load pre-trained model
        """msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)"""

        """if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}"""

        # manually initialize fc layer
        #trunc_normal_(model.head.weight, std=2e-5)

    classifier.to(device)

    model_without_ddp = classifier
    n_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=[{'pos_embed', 'cls_token'}],
        layer_decay=args.layer_decay
    )

    #param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    """if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()"""
    #criterion = SoftTargetCrossEntropy()
    #criterion = torch.nn.CrossEntropyLoss()
    #print("criterion = %s" % str(criterion))

    # resume model
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best_86.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler)

    """if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)"""

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_metrics = Acc_Metric(0.)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            classifier, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args, npoints= npoints
        )


        if epoch % loaded_args.val_freq == 0:

            metrics, acc_ = validate(model, data_loader_val, epoch, val_writer, args, loaded_config, logger=logger, device= device)
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                #builder.save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                # "ema_state_dict": model_ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model": args.model,
                }
                save_dict['loss_scaler'] = loss_scaler.state_dict()
                ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best.pth")
                utils.save_on_master(save_dict, ckpt_path)
                print(f"model_path: {ckpt_path}")
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            """if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)"""

        #builder.save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger) 


        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }

        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp.pth")
        utils.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir,
                                     "{}_{}_{:04d}.pth".format(args.model, args.experiment,
                                                               epoch))
            utils.save_on_master(save_dict, ckpt_path)

        """test_stats = evaluate(data_loader_val, model, device)
        print(f"Pretrained from: {args.finetune}")
        print(f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')"""

        #if log_writer is not None:
            #log_writer.add_scalar('perf/test_acc', test_stats['acc1'], epoch)
            #log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            #log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        #**{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters, 
                        "val_acc": acc_}

        log_stats_list = {k: tensor_to_list(v) for k, v in log_stats.items()}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(
                    args.output_dir,
                    "{}_{}_log.txt".format(
                        args.model,
                        args.experiment
                    )
            ), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_list) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, device = 0):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].to(device)
            label = data[1].to(device)

            points = miscc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc), acc

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


if __name__ == '__main__':
    # if not misc.is_main_process():
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass
        
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
