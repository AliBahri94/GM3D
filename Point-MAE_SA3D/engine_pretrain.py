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
from typing import Iterable
import builtins

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision import transforms
from datasets import data_transforms

#torch.autograd.set_detect_anomaly(True)


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

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None, model_ema=None, model_teacher=None, scheduler= None, optimizer_learn_loss = None):
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

    if args.learning_loss:
        assert model_ema is not None
        if epoch < 100:
            model_ema.decay = 0.999 + epoch / 100 * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

    optimizer.zero_grad()
    #optimizer_learn_loss.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    u = 0
    for data_iter_step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        u += 1
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = batch

        samples = points.to(device, non_blocking=True)
        samples = train_transforms(samples)
        bool_masked_pos = torch.zeros(samples.shape[0], 64).to(device, non_blocking=True).to(torch.bool)
        #bool_masked_pos = torch.zeros(samples.shape[0], 128).to(device, non_blocking=True).to(torch.bool)
        #bool_masked_pos = torch.zeros(samples.shape[0], 256).to(device, non_blocking=True).to(torch.bool)
        #bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)   # (N, L)
        visible_mask = torch.zeros_like(bool_masked_pos).to(device, non_blocking=True).to(torch.bool)

        with torch.cuda.amp.autocast():
            if model_ema is not None:
                with torch.no_grad():
                    outs_ema = model_ema.ema(samples, mask=visible_mask)

            if args.learning_loss:
                # generate mask by predicted loss
                mask = model_ema.ema.generate_mask(outs_ema['loss_pred'], mask_ratio=args.mask_ratio,
                                                   guide=True, epoch=epoch, total_epoch=args.epochs)
                bool_masked_pos = mask.to(device, non_blocking=True).flatten(1).to(torch.bool)

            outs = model(samples, mask=bool_masked_pos)

            if args.learn_feature_loss != 'none':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if args.learn_feature_loss in ['clip', 'dino']:
                            #feature_target = forward_features(model_teacher, samples, args.learn_feature_loss)
                            #feature_target = forward_features(model_teacher, outs_ema["neighborhood"], outs_ema["center"], args.learn_feature_loss)
                            feature_target, point_target, point_reconstructed= forward_features_Decoder(model_teacher, outs_ema["neighborhood"], outs_ema["center"], args.learn_feature_loss, outs["pix_pred"][:, -outs['mask_num']:], outs["mask"])
                            
                        elif args.learn_feature_loss == 'ema':
                            feature_target = outs_ema['features'][:, 1:, :]

                #loss_outs = model.module.forward_loss(
                """loss_outs = model.forward_loss(
                    outs['pix_pred'][:, -outs['mask_num']:],
                    feature_target.detach(),
                    outs['mask'],
                )"""
                """loss_outs = model.forward_loss(
                    outs['pix_pred'][:, -outs['mask_num']:],
                    feature_target.detach(),
                    outs['mask'],
                    #point_target,
                    outs["neighborhood"],
                    point_reconstructed,
                )"""
                ############ MLP IN Model
                loss_outs = model.forward_loss(
                    outs['pix_pred'][:, -outs['mask_num']:],
                    feature_target.detach(),
                    outs['mask'],
                    #point_target,
                    outs["neighborhood"],
                    #point_reconstructed,
                    outs['point_pred'][:, -outs['mask_num']:],
                )

            else:
                #loss_outs = model.module.forward_loss(
                loss_outs = model.forward_loss(
                    #samples,
                    outs['pix_pred'][:, -outs['mask_num']:],
                    outs["neighborhood"],
                    #outs['pix_pred'],
                    outs['mask'],
                )

            if isinstance(loss_outs, dict):
                #loss = loss_outs['mean']                    
                loss_mse = loss_outs['MSE_mean']               #*********change for loss
                loss_chfr = loss_outs['Chamfer_mean']          #*********change for loss
                #if (epoch < 15):
                if (epoch < -1):
                    loss = loss_outs['MSE_mean'] + loss_outs['Chamfer_mean']
                else:
                    #loss = 13.889*(loss_outs['MSE_mean']) + 1000.*(loss_outs['Chamfer_mean'])
                    loss = 13.889*(loss_outs['MSE_mean']) + 1.*(loss_outs['Chamfer_mean'])
            #### original
            """if isinstance(loss_outs, dict):
                loss = loss_outs['mean']                    
                #loss = loss_outs['mean']               #*********change for loss
            else:
                loss = loss_outs"""

            if args.learning_loss:
                loss_target = loss_outs['matrix']

                #loss_learn = model.module.forward_learning_loss(
                loss_learn = model.forward_learning_loss(
                    outs['loss_pred'][:,  -outs['mask_num']:],
                    #outs['loss_pred'],
                    bool_masked_pos,
                    loss_target.detach(),
                    relative=args.relative,
                )
                loss_learn_value = loss_learn.item()
                if not math.isfinite(loss_learn_value):
                    print("Loss learning is {}, skip".format(loss_learn_value))
                    sys.exit(1)

        #loss_value = loss.item()                    #### original
        #if (epoch < 15):
        if (epoch < -1):
            loss_value = loss_mse.item() + loss_chfr.item()                 #*********change for loss
        else:
            #loss_value = 13.889*(loss_mse.item()) + 1000.*(loss_chfr.item())                 #*********change for loss  
            loss_value = 13.889*(loss_mse.item()) + 1.*(loss_chfr.item())

        if not math.isfinite(loss_value):
            print("Loss is {}, skip".format(loss_value))
            sys.exit(1)

        if args.learning_loss:
            loss += loss_learn
            #loss = (1.5 * loss) + (loss_learn)
            #loss += (1.5 * loss_learn)
            #pass

        loss = loss / accum_iter
        #loss_learn = loss_learn / accum_iter 
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)        #### yadet bashe reatain=True gozashte budi ti MSE ha, retain_graph=False
        
        ############## Added by ******
        #loss.backward()
        #optimizer.step()
        #grad_norm = 0
        #################################
        #grad_norm_learn_loss = loss_scaler(loss_learn, optimizer_learn_loss, parameters=model.increase_dim_2.parameters(),
                                #update_grad=(data_iter_step + 1) % accum_iter == 0, retain_graph=True)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            #optimizer_learn_loss.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.learning_loss:
            metric_logger.update(loss_learn=loss_learn_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(grad_norm=grad_norm)
        #if (epoch < 15):
        if (epoch < -1):
            metric_logger.update(loss_mse=loss_mse.item())                                          #*********change for loss
            metric_logger.update(loss_chfr=loss_chfr.item())                                        #*********change for loss
            #metric_logger.update(grad_norm_learn_loss=grad_norm_learn_loss)
        else:
            metric_logger.update(loss_mse=loss_mse.item()*13.889)                                          #*********change for loss
            #metric_logger.update(loss_chfr=loss_chfr.item()*1000.)   
            metric_logger.update(loss_chfr=loss_chfr.item()*1.)  

        #### original
        #metric_logger.update(loss_mse=loss.item())                                          #*********change for loss

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_reduce_mse = misc.all_reduce_mean(loss_mse.item()*13.889)                         #*********change for loss
        loss_value_reduce_chfr = misc.all_reduce_mean(loss_chfr.item()*1000.)                    #*********change for loss

        #### original
        #loss_value_reduce_mse = misc.all_reduce_mean(loss.item())

        if args.learning_loss:
            loss_learn_value_reduce = misc.all_reduce_mean(loss_learn_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('train_loss', loss_value_reduce, it)
            log_writer.add_scalar('train_loss_MSE', loss_value_reduce_mse, it)                              #*********change for loss
            log_writer.add_scalar('train_loss_Chfr', loss_value_reduce_chfr, it)                              #*********change for loss
            log_writer.add_scalar('lr', lr, it)
            log_writer.add_scalar('grad_norm', grad_norm, it)
            #log_writer.add_scalar('grad_norm_learn_loss', grad_norm_learn_loss, it)

            if args.learning_loss:
                log_writer.add_scalar('train_loss_learn', loss_learn_value_reduce, it)

        if (data_iter_step + 1) >= len(data_loader):
            break

        """if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)"""

        #if (u == 2):
            #break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def forward_features(model, x, center, model_type):
    assert model_type in ['dino', 'clip']
    if model_type == 'dino':
        return forward_features_dino(model, x, center)
    else:
        return forward_features_clip(model, x)
    
def forward_features_Decoder(model, x, center, model_type, features, mask_real):
    assert model_type in ['dino', 'clip']
    if model_type == 'dino':
        return forward_features_dino_decoder(model, x, center, features, mask_real)
    else:
        return forward_features_clip(model, x)    


def forward_features_dino(model, x, center):
    B = x.shape[0]

    x_vis, mask = model.MAE_encoder(x, center)

    return x_vis


"""def forward_features_dino_decoder(model, x, center, features, mask_real):
    B = x.shape[0]

    x_vis, mask = model.MAE_encoder(x, center)
    B, N, C = x_vis.shape

    ########## decoder part for original features
    pos_emd_vis = model.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
    x_rec = model.MAE_decoder(x_vis, pos_emd_vis, N)
    rebuild_points_org = model.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)
    #######################

    ########## decoder part for reconstructed features
    pos_emd_vis = model.decoder_pos_embed(center[mask_real]).reshape(B, -1, C)
    x_rec = model.MAE_decoder(features, pos_emd_vis, N)
    rebuild_points_reco = model.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)
    #######################

    return x_vis, rebuild_points_org, rebuild_points_reco"""

################# MSE Balance 
def forward_features_dino_decoder(model, x, center, features, mask_real):
    B = x.shape[0]

    x_vis, mask = model.MAE_encoder(x, center)
    B, N, C = x_vis.shape

    ########## decoder part for original features
    pos_emd_vis = model.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
    x_rec_org = model.MAE_decoder(x_vis, pos_emd_vis, N)
    rebuild_points_org = model.increase_dim(x_rec_org.transpose(1, 2)).transpose(1, 2)
    #######################

    ########## decoder part for reconstructed features
    pos_emd_vis = model.decoder_pos_embed(center[mask_real]).reshape(B, -1, C)
    x_rec = model.MAE_decoder(features, pos_emd_vis, N)
    rebuild_points_reco = model.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)
    #######################

    return x_rec_org, rebuild_points_org, rebuild_points_reco


def forward_features_clip(model, x):
    x = model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x = model.ln_post(x[:, 0, :])
    x = model.ln_post(x)

    if model.proj is not None:
        x = x @ model.proj

    return x[:, 1:, :]


######################################## save point cloud and its loss-pred in .ply file 
def tensors_to_ply(a, cof, filename="output.ply"):
    """
    Save a point cloud to a .ply file.
    a: tensor of shape (num_groups, num_points, 3)
    cof: tensor of shape (num_groups,)
    filename: desired output filename
    """
    num_groups, num_points, _ = a.shape
    
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_groups * num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write point cloud data
        for i in range(num_groups):
            for j in range(num_points):
                x, y, z = a[i, j]
                r, g, b = cof_to_color(cof[i])  # Convert cof value to RGB
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

def cof_to_color(cof_value):
    """
    Convert a cof value to an RGB color. 
    Since cof is between 0 and 1, we'll use it to interpolate between blue (0) and red (1).
    """
    r = int(255 * cof_value)
    g = 0
    b = int(255 * (1 - cof_value))
    return r, g, b

def min_max_normalize(tensor):
    """
    Normalize a tensor to have values between 0 and 1.

    Parameters:
    tensor (torch.Tensor): The input tensor to normalize.

    Returns:
    torch.Tensor: A new tensor with values normalized between 0 and 1.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor    

# Save to PLY
#tensors_to_ply(a, cof, "output.ply")    
######################################################################################
