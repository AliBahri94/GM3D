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
import builtins

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2"  
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma

from util import utils
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import ImageListFolder

from engine_pretrain_Classifier_SVM import train_one_epoch, train_one_epoch_seperated
from mask_transform import MaskTransform

import models_mae
#import models_mae_learn_loss_Classifier_SVM
import models_mae_learn_feature_loss

import pickle
from utils import registry
from tools import builder
from types import SimpleNamespace
import yaml
from utils.logger import *
from utils import miscc, dist_utils
from sklearn.svm import SVC

from models.Point_MAE import Classifier

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#torch.autograd.set_detect_anomaly(True)


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
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--bf16', action='store_true', help='whether to use bf16')

    # Model parameters
    #parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
    parser.add_argument('--model', default='mae_vit_base_patch16_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--token_size', default=int(224 / 16), type=int,
                        help='number of patch (in one dimension), usually input_size//16')  # for mask generator
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    # Mask parameters (by UM-MAE)
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_regular', action='store_true',
                        help='Uniform sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_regular=False)
    parser.add_argument('--mask_block', action='store_true',
                        help='Block sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_block=False)
    parser.add_argument('--vis_mask_ratio', default=0.0, type=float,
                        help='Secondary masking ratio (mask percentage of visible patches, secondary masking phase).')

    # HPM parameters
    parser.add_argument('--learning_loss', action='store_true', help='Learn to predict loss for each patch.')
    parser.set_defaults(learning_loss=True)
    #parser.add_argument('--learn_feature_loss', default='dino', type=str,
    #parser.add_argument('--learn_feature_loss', default='none', type=str,
    #                    help='Use MSE loss for features as target.')
    parser.add_argument('--byol', default='none', type=str,
                        help='Use MSE loss for features as target.')
    parser.add_argument('--relative', action='store_true', help='Use relative learning loss or not.')
    parser.set_defaults(relative=True)
    #parser.add_argument('--dino_path', default='./HPM/pretrain_PMAE.pth', type=str,
    #parser.add_argument('--dino_path', default='none', type=str,
    #                    help='Pre-trained DINO for feature distillation (ViT-B/16).')
    parser.add_argument('--clip_path', default='none', type=str,
                        help='Pre-trained CLIP for feature distillation (ViT-B/16).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
    #parser.add_argument('--weight_decay', type=float, default=0.15,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, 40 for MAE and 10 for SimMIM')

    # Dataset parameters
    parser.add_argument('--data_path', default='./HPM/data/', type=str,
                        help='dataset path')



    parser.add_argument('--output_dir', default='./alakiii/2_PMAE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_chamfer_multiply_1000_shared_learnable_tokens_shared_optimizer_for_loss_prediction_and_reconstruction_4_4_Decoders_Original_MLP_for_test', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./alakiii/2_PMAE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_chamfer_multiply_1000_shared_learnable_tokens_shared_optimizer_for_loss_prediction_and_reconstruction_4_4_Decoders_Original_MLP_for_test', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--shared_opt', default=True, type=bool)
    parser.add_argument('--after_200_epoch', default=False, type=bool)
    parser.add_argument('--classification', default=False, type=bool)
    parser.add_argument('--loss_multiply_by', default=[13.889 , 1000], type=list)    #13.889 , 1000
    parser.add_argument('--after_epoch', default= 15, type=int, help='feature or usual')
    parser.add_argument('--dino_path', default='./HPM/pretrain_PMAE.pth', type=str,
    #parser.add_argument('--dino_path', default='none', type=str,
                        help='Pre-trained DINO for feature distillation (ViT-B/16).')
    parser.add_argument('--learn_feature_loss', default='dino', type=str, help='Use MSE loss for features as target.')
    #parser.add_argument('--learn_feature_loss', default='none', type=str, help='Use MSE loss for features as target.')
    parser.add_argument('--mode', default='feature', type=str, help='feature or usual')
    parser.add_argument('--decoder_part_ablattion', default=False, type=bool)
    parser.add_argument('--shared_learnable_tokens', default=False, type=bool)


    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--load_from', default='')
    parser.add_argument('--experiment', default='hpm_relative_in1k_ep200', type=str, help='experiment name (for log)')
    parser.add_argument('--name_config', default='./finetune_modelnet.yaml',         #config_finetune_scan_hardest
                        help='path where to save, empty for no saving')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser
 


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

    # simple augmentation
    transform_train = MaskTransform(args)

    # build dataset
    """dataset_train = ImageListFolder(os.path.join(args.data_path, 'Data/CLS-LOC/train'), transform=transform_train,
                                    ann_file=os.path.join(args.data_path, 'ImageSets/CLS-LOC/train_cls_modified.txt'))
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )"""

    ################ dataset for point cloud
    """# Save the dictionary to a json file
    with open("config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)"""

    """# Assuming args is your namespace object
    with open("args.pkl", "wb") as pkl_file:
        pickle.dump(args, pkl_file)"""

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    with open('./args.pkl', 'rb') as f:
        loaded_args = pickle.load(f)

    """with open("config.yaml", 'r') as stream:
        try:
            loaded_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)"""
    with open(args.name_config, 'r') as stream:  
        loaded_config = yaml.safe_load(stream)    

    with open("config_m.yaml", 'r') as stream:
        try:
            loaded_config_m = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)    

    loaded_config = DotDict(loaded_config)
    loaded_config_m = DotDict(loaded_config_m)
    sampler_train, data_loader_train = builder.dataset_builder(loaded_args, loaded_config.dataset.train)

    loaded_config.dataset.extra_train_svm.others.bs = loaded_config.total_bs * 2
    loaded_config.dataset.extra_test_svm.others.bs = loaded_config.total_bs * 2
    #loaded_config.dataset.extra_train_svm.others.bs = int(loaded_config.total_bs / 4)
    #loaded_config.dataset.extra_test_svm.others.bs = int(loaded_config.total_bs / 4)
    (_, extra_train_dataloader), (_, extra_test_dataloader),  = builder.dataset_builder(args, loaded_config.dataset.extra_train_svm), \
                                                            builder.dataset_builder(args, loaded_config.dataset.extra_test_svm)
    
    """(_, extra_train_dataloader), (_, extra_test_dataloader),  = builder.dataset_builder(args, loaded_config.dataset.extra_train_svm), \
                                                            builder.dataset_builder(args, loaded_config.dataset.extra_test_svm)"""
    
    val_writer = SummaryWriter(os.path.join(args.output_dir, 'test'))
    logger = get_logger("pretrain")
    ########################################

    if (args.mode == "usual"):
        import models_mae_learn_loss_Classifier_SVM as models_mae_learn_loss_Classifier_SVM
    else:
        import models_mae_learn_loss_Classifier_SVM_feature_besed as models_mae_learn_loss_Classifier_SVM

    if global_rank == 0 and args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, f"{args.model}_{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    model_teacher = None
    # define the model
    if args.learning_loss:
        if args.learn_feature_loss != 'none':
            assert args.learn_feature_loss in ['clip', 'dino', 'ema']

            """model = models_mae_learn_feature_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                                       vis_mask_ratio=args.vis_mask_ratio)"""
            """model = models_mae_learn_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                               vis_mask_ratio=args.vis_mask_ratio)"""

            model = models_mae_learn_loss_Classifier_SVM.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                               vis_mask_ratio=args.vis_mask_ratio)                                                   

            if args.learn_feature_loss == 'dino':
                #model_teacher = timm.models.vit_base_patch16_224()
                #model_teacher.load_state_dict(torch.load(args.dino_path), strict=False)

                model_teacher = builder.model_builder(loaded_config_m.model)
                base_ckpt = torch.load(args.dino_path)
                base_ckpt = {k.replace("module.", ""): v for k, v in base_ckpt['base_model'].items()}
                model_teacher.load_state_dict(base_ckpt, strict=True)
            else:
                from models_clip import build_model
                state_dict = torch.load(args.clip_path, map_location='cpu')
                model_clip = build_model(state_dict)
                model_clip.load_state_dict(state_dict, strict=False)
                model_clip.float()

                model_teacher = model_clip.visual

            """############### freez parameters
            for param in model_teacher.parameters():
                param.requires_grad = False

            for param in model_teacher.increase_dim.parameters():
                param.requires_grad = True
            ################################"""

            model_teacher.to(device)
            model_teacher.eval()

            if (args.classification):
                classifier = Classifier()
        else:
            model = models_mae_learn_loss_Classifier_SVM.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                               vis_mask_ratio=args.vis_mask_ratio)
            if (args.classification):
                classifier = Classifier()

    else:
        if args.learn_feature_loss != 'none':
            assert args.learn_feature_loss in ['clip', 'dino', 'ema']

            model = models_mae_learn_feature_loss.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                                       vis_mask_ratio=args.vis_mask_ratio,
                                                                       learning_loss=False)

            if args.learn_feature_loss == 'dino':
                model_teacher = timm.models.vit_base_patch16_224()
                model_teacher.load_state_dict(torch.load(args.dino_path), strict=False)
            else:
                from models_clip import build_model
                state_dict = torch.load(args.clip_path, map_location='cpu')
                model_clip = build_model(state_dict)
                model_clip.load_state_dict(state_dict, strict=False)
                model_clip.float()

                model_teacher = model_clip.visual

            model_teacher.to(device)
            model_teacher.eval()
        else:
            model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                    vis_mask_ratio=args.vis_mask_ratio)

    model.to(device)
    if (args.classification):
        classifier.to(device)
    else:
        classifier = None    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # define ema model
    model_ema = None
    if args.byol or args.learning_loss or args.learn_feature_loss == 'ema':
        # use momentum encoder for BYOL
        model_ema = ModelEma(model, decay=0.999, device=args.device, resume='')

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    ##########################################Shared Optimizer
    if (args.shared_opt == True):
        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        #param_groups_2 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

        if (args.classification):
            ############### Classifier
            param_groups_c = optim_factory.add_weight_decay(classifier, args.weight_decay)
            #param_groups_2 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
            optimizer_cls = torch.optim.AdamW(param_groups_c, lr=args.lr, betas=(0.9, 0.95))
            criterion_cls = torch.nn.CrossEntropyLoss()
        else:
            optimizer_cls = None
            criterion_cls = None
            ##########################################################

    else:    

        ######################################### Seperate Optimizer
        # following timm: set wd as 0 for bias and norm layers
        param_groups_encoder = optim_factory.add_weight_decay(model_without_ddp.MAE_encoder, args.weight_decay)
        param_groups_decoder = optim_factory.add_weight_decay(model_without_ddp.MAE_decoder, args.weight_decay)
        param_groups_MLP = optim_factory.add_weight_decay(model_without_ddp.increase_dim_just_network_without_feature, args.weight_decay)
        param_groups = param_groups_encoder + param_groups_decoder + param_groups_MLP
        #param_groups_2 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

        if (args.classification):
            ############### Classifier
            param_groups_c = optim_factory.add_weight_decay(classifier, args.weight_decay)
            #param_groups_2 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
            optimizer_cls = torch.optim.AdamW(param_groups_c, lr=args.lr, betas=(0.9, 0.95))
            criterion_cls = torch.nn.CrossEntropyLoss()
            ##########################
        else:
            optimizer_cls = None
            criterion_cls = None    

        ############### Loss Prediction
        param_groups_decoder_loss_pred = optim_factory.add_weight_decay(model_without_ddp.MAE_decoder_loss_pred, args.weight_decay)
        param_groups_MLP_loss_pred = optim_factory.add_weight_decay(model_without_ddp.increase_dim_2, args.weight_decay)
        param_groups_loss_pred = param_groups_decoder_loss_pred + param_groups_MLP_loss_pred
        #param_groups_2 = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer_loss_pred = torch.optim.AdamW(param_groups_loss_pred, lr=args.lr, betas=(0.9, 0.95))
        ###############################################################

    print(optimizer)
    loss_scaler = NativeScaler()

    ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_bestteeeee.pth")
    #ckpt_path = "./Final_CVPR2024/2_PMAE_Chamfer_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_chamfer_multiply_1000_shared_learnable_tokens_shared_optimizer_for_loss_prediction_and_reconstruction_4_4_Decoders_Original_MLP/mae_vit_base_patch16_dec512d8b_hpm_relative_in1k_ep200_temp_last.pth"
    #ckpt_path = "./new_experiments_Feature_Based/2_PMAE_Chamfer_MSE_Feauture_based_lr_for_Hard_Patches_keep_ratio_0_5_G_64_N_32_Mask_ratio_6_2_decoders_chamfer_multiply_1000_seperate_learnable_tokens_shared_optimizer_for_loss_prediction_and_reconstruction_MODIFIED_2/mae_vit_base_patch16_dec512d8b_hpm_relative_in1k_ep200_temp_last.pth"

    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, model_ema=model_ema.ema)

    print(f"Start training for {args.epochs} epochs")
    best_metrics = Acc_Metric(0.)
    start_time = time.time()
    #args.start_epoch = 398
    for epoch in range(args.start_epoch, args.epochs):
        #data_loader_train.sampler.set_epoch(epoch)                 It is important I think. So, pay attention

        ######## shared optimizer
        if (args.shared_opt == True):
            train_stats = train_one_epoch(
                model, classifier, data_loader_train, extra_train_dataloader, criterion_cls, 
                optimizer, optimizer_cls, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                model_ema=model_ema,
                model_teacher=model_teacher,
                scheduler= None,
                optimizer_learn_loss = None,
                after_200_epoch = args.after_200_epoch,
                classification = args.classification,
                loss_multiply_by = args.loss_multiply_by,
                after_epoch = args.after_epoch,
                shared_learnable_tokens = args.shared_learnable_tokens
            )

        else:
            ######## seperated optimizer
            train_stats = train_one_epoch_seperated(
                model, classifier, data_loader_train, extra_train_dataloader, criterion_cls, 
                optimizer, optimizer_cls, optimizer_loss_pred, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                model_ema=model_ema,
                model_teacher=model_teacher,
                scheduler= None,
                optimizer_learn_loss = None,
                after_200_epoch = args.after_200_epoch,
                classification = args.classification,
                loss_multiply_by = args.loss_multiply_by,
                after_epoch = args.after_epoch
            )

        if epoch % args.val_freq == 0:
            metrics, svm_acc, metrics_cls_acc, cls_acc = validate(model, classifier, extra_train_dataloader, extra_test_dataloader, epoch, val_writer, args, loaded_config, device, args.classification, logger=logger)
            sign_saved = " "
            ############################ ADD with Moslem (after 200 epochs)
            if (args.after_200_epoch):
                if (epoch >= int(args.epochs/2)):

                    # Save ckeckpoints
                    #if metrics.better_than(best_metrics): 
                    if (args.classification):
                        if metrics_cls_acc.better_than(best_metrics):
                            best_metrics = metrics_cls_acc
                            save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            # "ema_state_dict": model_ema.ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model": args.model,
                            }
                            if model_ema is not None:
                                save_dict['ema_state_dict'] = model_ema.ema.state_dict()
                            if loss_scaler is not None:
                                save_dict['loss_scaler'] = loss_scaler.state_dict()

                            ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best.pth")
                            utils.save_on_master(save_dict, ckpt_path)
                            print(f"model_path: {ckpt_path}")
                            print("###############################################################################################") 
                    else:
                        if metrics.better_than(best_metrics):
                            best_metrics = metrics
                            save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            # "ema_state_dict": model_ema.ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model": args.model,
                            }
                            if model_ema is not None:
                                save_dict['ema_state_dict'] = model_ema.ema.state_dict()
                            if loss_scaler is not None:
                                save_dict['loss_scaler'] = loss_scaler.state_dict()

                            ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best.pth")
                            utils.save_on_master(save_dict, ckpt_path)
                            print(f"model_path: {ckpt_path}")
                            print("###############################################################################################") 
                            sign_saved = "###################################################"      
                            ############################################

            else:
                # Save ckeckpoints
                if (args.classification):
                    if metrics_cls_acc.better_than(best_metrics):
                        best_metrics = metrics_cls_acc
                        save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        # "ema_state_dict": model_ema.ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model": args.model,
                        }
                        if model_ema is not None:
                            save_dict['ema_state_dict'] = model_ema.ema.state_dict()
                        if loss_scaler is not None:
                            save_dict['loss_scaler'] = loss_scaler.state_dict()

                        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best.pth")
                        utils.save_on_master(save_dict, ckpt_path)
                        print(f"model_path: {ckpt_path}")
                        print("###############################################################################################") 
                        sign_saved = "###################################################"

                else:    
                    if metrics.better_than(best_metrics):
                        best_metrics = metrics
                        save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        # "ema_state_dict": model_ema.ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model": args.model,
                        }
                        if model_ema is not None:
                            save_dict['ema_state_dict'] = model_ema.ema.state_dict()
                        if loss_scaler is not None:
                            save_dict['loss_scaler'] = loss_scaler.state_dict()

                        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_best.pth")
                        utils.save_on_master(save_dict, ckpt_path)
                        print(f"model_path: {ckpt_path}")
                        print("###############################################################################################") 
                        sign_saved = "###################################################"

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            # "ema_state_dict": model_ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if model_ema is not None:
            save_dict['ema_state_dict'] = model_ema.ema.state_dict()
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp_last.pth")
        utils.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir,
                                     "{}_{}_{:04d}.pth".format(args.model, args.experiment,
                                                                     epoch))
            utils.save_on_master(save_dict, ckpt_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, "val_svm_acc": svm_acc, "val_cls_acc": cls_acc, "sign_saved": sign_saved}

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
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def validate(model, classifier, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, device, classification, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    model.eval()  # set model to eval mode
    if (classification):
        classifier.eval()

    test_features = []
    test_label = []

    train_features = []
    train_label = []

    test_features_cls = []

    npoints = config.dataset.train.others.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].to(device)
            label = data[1].to(device)
            points = miscc.fps(points, npoints)
            assert points.size(1) == npoints
            mask = torch.zeros((points.shape[0], 64)).to(torch.bool).to(device)
            feature = model(points, mask, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].to(device)
            label = data[1].to(device)
            points = miscc.fps(points, npoints)
            assert points.size(1) == npoints
            mask = torch.zeros((points.shape[0], 64)).to(torch.bool).to(device)
            feature = model(points, mask, noaug=True)
            target = label.view(-1)
            if (classification):
                out_cls= classifier(feature)
                test_features_cls.append(out_cls.detach())

            test_features.append(feature.detach())
            test_label.append(target.detach())

            
        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        if (classification):
            test_features_cls = torch.cat(test_features_cls, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)    

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())
        if (classification):
            cls_acc = np.sum(test_label.data.cpu().numpy() == np.argmax(test_features_cls.data.cpu().numpy(), -1)) * 1. / test_features_cls.shape[0]
            print_log('[Validation] EPOCH: %d  cls_acc = %.4f' % (epoch,cls_acc), logger=logger)
        print_log('[Validation] EPOCH: %d  svm_acc = %.4f' % (epoch,svm_acc), logger=logger)


        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    if (classification):
        return Acc_Metric(svm_acc), svm_acc, Acc_Metric(cls_acc), cls_acc
    else:
        return Acc_Metric(svm_acc), svm_acc, None, None    

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    # clf = LinearSVC()
    clf = SVC(C=0.01, kernel='linear')
    train_features = train_features.mean(1) + train_features.max(1)
    clf.fit(train_features, train_labels)
    test_features = test_features.mean(1) + test_features.max(1)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


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
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
