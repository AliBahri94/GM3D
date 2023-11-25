# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

from functools import partial
import random

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from timm.models.vision_transformer import PatchEmbed, Block, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

###### Point_Cloud
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
###### Point_Cloud

### orginal
class MaskedAutoencoderViT__(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 asymmetric_decoder=False, mask_ratio=0.75, vis_mask_ratio=0.,
                 saliency=False):
        
        super().__init__()

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.saliency = saliency
        if saliency:
            self.saliency_model = BASNet(3, 1)
            ckpt_path = 'saliency_model/basnet.pth'
            self.saliency_model.load_state_dict(torch.load(ckpt_path))
            self.saliency_model.eval()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))    ############################################ avaz kardam baraye PMAE
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      #requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        #self.norm = norm_layer(embed_dim)                             ############################################ avaz kardam baraye PMAE
         # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics (reconstructor and loss predictor)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # reconstructor
        self.decoder_blocks = nn.ModuleList([
            #Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # loss predictor
        self.decoder_blocks_losspred = nn.ModuleList([
            #Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm_losspred = norm_layer(decoder_embed_dim)
        self.decoder_pred_losspred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        #self.initialize_weights()

        ######### Point_Cloud
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = 32
        self.decoder_depth = 4
        self.decoder_num_heads = 6
        self.num_group = 64

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
        embed_dim = self.trans_dim,
        depth = self.depth,
        drop_path_rate = dpr,
        num_heads = self.num_heads,
        )
        self.MAE_decoder = TransformerDecoder(
        embed_dim=self.trans_dim,
        depth=self.decoder_depth,
        drop_path_rate=dpr,
        num_heads=self.decoder_num_heads,
    )
        self.MAE_decoder_loss_pred = TransformerDecoder(
        embed_dim=self.trans_dim,
        depth=self.decoder_depth,
        drop_path_rate=dpr,
        num_heads=self.decoder_num_heads,
    )

        self.norm_p = nn.LayerNorm(self.trans_dim)
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.mask_token_loss_pred = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, 3*self.group_size, 1, bias=True)
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            nn.Conv1d(1024, self.trans_dim, 1, bias=True)
        )
        self.increase_dim_2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            nn.Conv1d(1024, self.trans_dim, 1, bias=True)
        )
        self.increase_dim_original= nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, 3*self.group_size, 1, bias=True)
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            nn.Conv1d(1024, 3*self.group_size, 1, bias=True)
        )

        ########## just network without feature
        self.increase_dim_just_network_without_feature = nn.Sequential(
            #nn.Conv1d(self.trans_dim, 1024, 1),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1, bias=True)
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            #nn.Conv1d(1024, self.trans_dim, 1, bias=True)
        )

        ############# predict_chamfer_and_MSE
        """self.predict_chamfer_and_MSE = nn.Sequential(
            #nn.Conv1d(self.trans_dim, 1024, 1),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, 3*self.group_size, 1, bias=True)
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1, bias=True)
            #nn.Conv1d(1024, self.trans_dim, 1, bias=True)
        )"""

        self.loss_func = ChamferDistanceL2().cuda()
        ######### Point_Cloud


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, 'vis_mask_token'):
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        # x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, _, D = x.shape
        x = x[~mask].reshape(N, -1, D)

        if self.vis_mask_ratio > 0:
            vis_mask_token = self.vis_mask_token + self.pos_embed[:, 1:, :]
            vis_mask_token = vis_mask_token.expand(N, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(N, -1, D)
            L = x.size(1)
            noise = torch.rand(N, L, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(L * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones([N, L], device=x.device)
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(vis_mask, dim=1, index=ids_restore).unsqueeze(-1)

            x = x * (1. - vis_mask) + vis_mask_token * vis_mask

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    

    def forward_encoder_point(self, neighborhood, center, mask):

        ####### Point_Cloud
        group_input_tokens = self.encoder(neighborhood)  #  B G C
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~mask].reshape(batch_size, -1, C)
        masked_center = center[~mask].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm_p(x_vis)
        ####### Point_Cloud

        return x_vis

    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        x_vis = x[:, 1:, :]
        N, _, D = x_vis.shape

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        pos_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_mask = expand_pos_embed[mask].reshape(N, -1, D)

        x_ = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)

        # add cls_token + decoder_pos_embed
        x = torch.cat([x[:, :1, :] + self.decoder_pos_embed[:, :1, :], x_], dim=1)
        loss_pred = x.clone()

        # apply reconstructor
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        # apply loss predictor
        for blk in self.decoder_blocks_losspred:
            loss_pred = blk(loss_pred)
        loss_pred = self.decoder_norm_losspred(loss_pred)
        loss_pred = self.decoder_pred_losspred(loss_pred)
        loss_pred = loss_pred[:, 1:, :]    # (N, L, 1)

        return x, pos_mask.shape[1], loss_pred.mean(dim=-1)

    """def forward_loss(self, imgs, pred, mask):
        
        #imgs: [N, 3, H, W]
        #pred: [N, mask, p*p*3]
        #mask: [N, L], 0 is keep, 1 is remove,
        
        target = self.patchify(imgs)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)
        pred = pred[mask].reshape(N, -1, D)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5  # (N, L, p*p*3)

        loss = (pred - target) ** 2

        return {'mean': loss.mean(), 'matrix': loss.mean(dim=-1)}"""
    
    ########## MSE
    """def forward_loss(self, imgs, pred, mask):
        
        #imgs: [N, 3, H, W]
        #pred: [N, mask, p*p*3]
        #mask: [N, L], 0 is keep, 1 is remove,
        
        target = imgs.reshape(imgs.shape[0], imgs.shape[1], -1)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)
        #pred = pred[mask].reshape(N, imgs.shape[1], -1)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5  # (N, L, p*p*3)

        loss = (pred - target) ** 2

        return {'mean': loss.mean(), 'matrix': loss.mean(dim=-1)}"""
    
    ########## Chamfer + MSE
    def forward_loss(self, pred, target, mask):

        N, t, n, D = target.shape
        ################### MSE
        ##target_MSE = target[mask].reshape(-1, n, D)
        ##pred = pred.reshape(-1, n, D)
        #N, PP, D = target.shape

        ##pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        ##target_MSE = torch.nn.functional.normalize(target_MSE, p=2, dim=-1)
        ##loss_mse = ((pred - target_MSE) ** 2).sum(dim=-1)
        ######################

        target = target[mask].reshape(-1, n, D)
        pred = pred.reshape(-1, n, D)

        pred = pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        ########### mean and var
        #mean = target.mean(dim=-1, keepdim=True)
        #var = target.var(dim=-1, keepdim=True)
        #target = (target - mean) / (var + 1.e-6) ** .5  # (N, L, p*p*3)
        ###########
        loss = self.loss_func(pred, target)

        loss = loss.reshape(N, -1, n)

        #return {'mean': loss.mean(), 'matrix': loss.mean(dim=-1)}
        return {'MSE_mean': (loss.mean()) * 0. , 'Chamfer_mean': loss.mean(), 'matrix': loss.mean(dim=-1)}
    
    ########## MSE_2 ()original
    """def forward_loss(self, pred, target, mask):

        target = target.reshape(target.shape[0], target.shape[1], -1)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        loss = ((pred - target) ** 2).sum(dim=-1)

        return {'mean': loss.mean(), 'matrix': loss}"""
    

    ########## Chamfer_2
    """def forward_loss(self, pred, target, mask):

        N, n, D = target.shape
        target = target[mask].reshape(N, -1, D)

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)

        pred = pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        pred = pred.reshape(-1, D)[:, :, None]
        pred = torch.cat((pred, pred, pred), -1)

        target = target.reshape(-1, D)[:, :, None]
        target = torch.cat((target, target, target), -1)

        loss = self.loss_func(pred, target) 

        #loss = loss[:, :, None]

        #return {'mean': loss.mean(), 'matrix': loss.mean(dim=-1)}
        return {'mean': loss.mean(), 'matrix': loss.mean(dim=-1).reshape(N, -1)}"""
    

    ########## MSE_2 & Chamfer
    """def forward_loss(self, pred, target, mask, point_target, point_reconstructed):

        ################### MSE
        N, P, D = target.shape
        target = target[mask].reshape(N, -1, D)
        N, PP, D = target.shape

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        loss_mse = ((pred - target) ** 2).sum(dim=-1)
        ######################

        ############## Chamfer
        point_target_ = point_target[mask]
        point_target_ = point_target_.reshape(N * PP, -1, 3)
        point_reconstructed_ = point_reconstructed.reshape(N * PP, -1, 3)

        point_reconstructed_ = point_reconstructed_.to(dtype=torch.float32)
        point_target_ = point_target_.to(dtype=torch.float32)

        loss_chamfer = self.loss_func(point_reconstructed_, point_target_) 
        loss_chamfer = loss_chamfer.reshape(N, PP, -1).mean(-1)
        ######################

        #loss_matrix = loss_mse + loss_chamfer 
        loss_matrix = loss_chamfer 

        return {'MSE_mean': loss_mse.mean(), 'Chamfer_mean': loss_chamfer.mean(), 'matrix': loss_matrix}"""



    ########## MSE_2 & Chamfer (MLP IN Model)
    """def forward_loss(self, pred, target, mask, point_target, point_reconstructed):

        ################### MSE
        N, P, D = target.shape
        target = target[mask].reshape(N, -1, D)
        N, PP, D = target.shape

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        loss_mse = ((pred - target) ** 2).sum(dim=-1)
        ######################

        ############## Chamfer
        point_target_ = point_target[mask]
        point_target_ = point_target_.reshape(N * PP, -1, 3)
        point_reconstructed_ = point_reconstructed.reshape(N * PP, -1, 3)

        point_reconstructed_ = point_reconstructed_.to(dtype=torch.float32)
        point_target_ = point_target_.to(dtype=torch.float32)

        loss_chamfer = self.loss_func(point_reconstructed_, point_target_) 
        loss_chamfer = loss_chamfer.reshape(N, PP, -1).mean(-1)
        ######################

        loss_matrix = loss_mse + loss_chamfer 
        #loss_matrix = loss_chamfer 

        return {'MSE_mean': loss_mse.mean(), 'Chamfer_mean': loss_chamfer.mean(), 'matrix': loss_matrix}"""


    """def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask)  # returned mask may change
        pred, mask_num, loss_pred = self.forward_decoder(latent, mask)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred[:, -mask_num:], mask)
        # return loss, pred, mask
        out = {
            'pix_pred': pred,
            'mask': mask,
            'mask_num': mask_num,
            'features': latent,
            'loss_pred': loss_pred,
        }
        return out"""
    
    """def forward(self, pts, mask, noaug = False):

        neighborhood, center, neighborhood_org= self.group_divider(pts)

        #latent = self.forward_encoder(imgs, mask)  # returned mask may change
        x_vis = self.forward_encoder_point(neighborhood, center, mask)  # returned mask may change

        B,_,C = x_vis.shape # B VIS C

        ################# ****
        if (noaug == True):
            return x_vis
        #################

        pos_emd_vis = self.pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        #loss_pred = pos_full.clone()
        loss_pred = x_full.clone()

        # apply reconstructor
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024
        #rebuild_points = self.increase_dim_original(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024

        # apply loss predictor
        loss_pred_ = self.MAE_decoder_loss_pred(loss_pred, pos_full, N)
        #loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024
        loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024

        #gt_points = neighborhood[mask].reshape(B*M,-1,3)

        out = {
            'pix_pred': rebuild_points,
            'mask': mask,
            'mask_num': pos_emd_mask.shape[1],
            'features': x_vis,
            'loss_pred': loss_pred_f.mean(dim=-1),
            #'gt_points': gt_points,
            'neighborhood': neighborhood,
            'neighborhood_org': neighborhood_org,
            'center': center,
        }

        return out"""
    
    ############ shabake ro freez kardam be joz increase_dim va albate increase_dim ham avaz kardam az 1 to 2
    """def forward(self, pts, mask, noaug = False):

        neighborhood, center, neighborhood_org= self.group_divider(pts)

        #latent = self.forward_encoder(imgs, mask)  # returned mask may change
        x_vis = self.forward_encoder_point(neighborhood, center, mask)  # returned mask may change

        B,_,C = x_vis.shape # B VIS C

        ################# ****
        if (noaug == True):
            return x_vis
        #################

        pos_emd_vis = self.pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        #loss_pred = pos_full.clone()
        loss_pred = x_full.clone()

        # apply reconstructor
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        ##rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024
        #rebuild_points = self.increase_dim_original(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024

        # apply loss predictor
        loss_pred_ = self.MAE_decoder_loss_pred(loss_pred, pos_full, N)
        #loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024
        loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024

        #gt_points = neighborhood[mask].reshape(B*M,-1,3)

        out = {
            'pix_pred': x_rec,
            'mask': mask,
            'mask_num': pos_emd_mask.shape[1],
            'features': x_vis,
            'loss_pred': loss_pred_f.mean(dim=-1),
            #'gt_points': gt_points,
            'neighborhood': neighborhood,
            'neighborhood_org': neighborhood_org,
            'center': center,
        }

        return out"""
    
    ################## just network without feature (shared optimizer)
    def forward(self, pts, mask, noaug = False):

        neighborhood, center, neighborhood_org= self.group_divider(pts)

        #latent = self.forward_encoder(imgs, mask)  # returned mask may change
        x_vis = self.forward_encoder_point(neighborhood, center, mask)  # returned mask may change

        B,_,C = x_vis.shape # B VIS C

        ################# ****
        if (noaug == True):
            return x_vis
        #################

        pos_emd_vis = self.pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1) 
        mask_token_loss_pred = self.mask_token_loss_pred.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        loss_pred = torch.cat([x_vis, mask_token_loss_pred], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        #loss_pred = pos_full.clone()
        #loss_pred = x_full.clone()

        # apply reconstructor
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim_just_network_without_feature(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024
        #rebuild_points = self.increase_dim_original(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024

        # apply loss predictor
        loss_pred_ = self.MAE_decoder_loss_pred(loss_pred, pos_full, N)
        #loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024
        loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024

        #gt_points = neighborhood[mask].reshape(B*M,-1,3)

        out = {
            'pix_pred': rebuild_points,
            'mask': mask,
            'mask_num': pos_emd_mask.shape[1],
            'features': x_vis,
            'loss_pred': loss_pred_f.mean(dim=-1),
            #'gt_points': gt_points,
            'neighborhood': neighborhood,
            'neighborhood_org': neighborhood_org,
            'center': center,
        }

        return out





    ############ shabake ro freez kardam be joz increase_dim va albate increase_dim ham avaz kardam az 1 to 2 (MLP IN Model)
    """def forward(self, pts, mask, noaug = False):

        neighborhood, center, neighborhood_org= self.group_divider(pts)

        #latent = self.forward_encoder(imgs, mask)  # returned mask may change
        x_vis = self.forward_encoder_point(neighborhood, center, mask)  # returned mask may change

        B,_,C = x_vis.shape # B VIS C

        ################# ****
        if (noaug == True):
            return x_vis
        #################

        pos_emd_vis = self.pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        #loss_pred = pos_full.clone()
        loss_pred = x_full.clone()

        # apply reconstructor
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = self.predict_chamfer_and_MSE(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024
        #rebuild_points = self.increase_dim_original(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024

        # apply loss predictor
        loss_pred_ = self.MAE_decoder_loss_pred(loss_pred, pos_full, N)
        #loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024
        loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024

        #gt_points = neighborhood[mask].reshape(B*M,-1,3)

        out = {
            'pix_pred': x_rec,
            'point_pred': rebuild_points,
            'mask': mask,
            'mask_num': pos_emd_mask.shape[1],
            'features': x_vis,
            'loss_pred': loss_pred_f.mean(dim=-1),
            #'gt_points': gt_points,
            'neighborhood': neighborhood,
            'neighborhood_org': neighborhood_org,
            'center': center,
        }

        return out"""



    
    @torch.no_grad()
    def generate_mask(self, loss_pred, mask_ratio=0.75, images=None, guide=True, epoch=0, total_epoch=200):
        N, L = loss_pred.shape
        len_keep = int(L * (1 - mask_ratio))

        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)    # (N, L)

        # keep `keep_ratio` loss and `1 - keep_ratio` random
        keep_ratio = 0.5
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()

        if guide:
            #keep_ratio = float((epoch + 1) / total_epoch) * 0.5
            keep_ratio = min(float((epoch + 1) / (total_epoch / 2)) * 0.5 , 0.5)
            #keep_ratio = float((epoch + 1) / total_epoch) * 0.9

        ## top 0 -> 0.5
        if int((L - len_keep) * keep_ratio) <= 0:
            # random
            noise = torch.randn(N, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                ## mask top `keep_ratio` loss and `1 - keep_ratio` random
                len_loss = int((L - len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=loss_pred.device)
        mask[:, :len_keep] = 0
        # unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_learning_loss(self, loss_pred, mask, loss_target, relative=False):
        
        #loss_pred: [N, L, 1]
        #mask: [N, L], 0 is keep, 1 is remove,
        #loss_target: [N, L]
        
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)

        if relative:
            # binary classification for LxL
            labels_positive = loss_target.unsqueeze(1) > loss_target.unsqueeze(2)
            labels_negative = loss_target.unsqueeze(1) < loss_target.unsqueeze(2)
            labels_valid = labels_positive + labels_negative

            loss_matrix = loss_pred.unsqueeze(1) - loss_pred.unsqueeze(2)
            loss = - labels_positive.int() * torch.log(torch.sigmoid(loss_matrix) + 1e-6) \
                   - labels_negative.int() * torch.log(1 - torch.sigmoid(loss_matrix) + 1e-6)

            return loss.sum() / labels_valid.sum()

        else:
            # normalize by each image
            mean = loss_target.mean(dim=1, keepdim=True)
            var = loss_target.var(dim=1, keepdim=True)
            loss_target = (loss_target - mean) / (var + 1.e-6) ** .5  # [N, L, 1]

            loss = (loss_pred - loss_target) ** 2
            loss = loss.mean()
            return loss
    
    """def forward_learning_loss(self, loss_pred, mask, loss_target, relative=False):
        
        #loss_pred: [N, L, 1]
        #mask: [N, L], 0 is keep, 1 is remove,
        #loss_target: [N, L]
        
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)

        # Compute the mean of gt and pr
        gt_mean = torch.mean(loss_target, dim=1, keepdim=True)
        pr_mean = torch.mean(loss_pred, dim=1, keepdim=True)

        # Compute the numerator and denominators for Pearson correlation
        numerator = torch.sum((loss_target - gt_mean) * (loss_pred - pr_mean), dim=1)
        denominator = torch.sqrt(torch.sum((loss_target - gt_mean) ** 2, dim=1) * torch.sum((loss_pred - pr_mean) ** 2, dim=1))

        # Pearson correlation for each item in the batch
        correlation_coefficient = numerator / denominator

        # Loss based on correlation: 1 - r. Averaged over batch.
        loss = 1 - torch.mean(correlation_coefficient)

        return loss"""


### modified_pose_embed and encoder as class
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 asymmetric_decoder=False, mask_ratio=0.75, vis_mask_ratio=0.,
                 saliency=False):
        
        super().__init__()

        #self.initialize_weights()

        ######### Point_Cloud
        self.encoder_dims = 384

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = 32
        self.decoder_depth = 4
        self.decoder_num_heads = 6
        self.num_group = 64

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_encoder = MaskTransformer()
        self.MAE_decoder = TransformerDecoder(
        embed_dim=self.trans_dim,
        depth=self.decoder_depth,
        drop_path_rate=dpr,
        num_heads=self.decoder_num_heads,
        )
        """self.MAE_decoder_loss_pred = TransformerDecoder(
        embed_dim=self.trans_dim,
        depth=self.decoder_depth,
        drop_path_rate=dpr,
        num_heads=self.decoder_num_heads,
        )"""
        ############### FOR modified_2
        dpr_MODIFIED_2 = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.MAE_decoder_loss_pred = TransformerDecoder(
        embed_dim=self.trans_dim,
        depth=self.depth,
        drop_path_rate=dpr_MODIFIED_2,
        num_heads=self.decoder_num_heads,
        )
        ############### FOR modified_2

        self.norm_p = nn.LayerNorm(self.trans_dim)
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.mask_token_loss_pred = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # prediction head
        self.increase_dim_2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            nn.Conv1d(1024, self.trans_dim, 1)
        )
        ########## just network without feature
        self.increase_dim_just_network_without_feature = nn.Sequential(
            #nn.Conv1d(self.trans_dim, 1024, 1),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            #nn.Conv1d(1024, self.trans_dim, 1, bias=True)
        )

        """self.increase_dim_features = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Conv1d(self.trans_dim, self.trans_dim, 1, bias=True)
            nn.Conv1d(1024, self.trans_dim, 1)
        )"""

        self.loss_func = ChamferDistanceL2().cuda()
        ######### Point_Cloud


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, 'vis_mask_token'):
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    ########## MSE_2 & Chamfer
    def forward_loss(self, pred, target, mask, point_target, point_reconstructed):

        ################### MSE
        N, P, D = target.shape
        target = target[mask].reshape(N, -1, D)
        N, PP, D = target.shape

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        loss_mse = ((pred - target) ** 2).sum(dim=-1)
        ######################

        ############## Chamfer
        point_target_ = point_target[mask]
        point_target_ = point_target_.reshape(N * PP, -1, 3)
        point_reconstructed_ = point_reconstructed.reshape(N * PP, -1, 3)

        point_reconstructed_ = point_reconstructed_.to(dtype=torch.float32)
        point_target_ = point_target_.to(dtype=torch.float32)

        loss_chamfer = self.loss_func(point_reconstructed_, point_target_) 
        loss_chamfer = loss_chamfer.reshape(N, PP, -1).mean(-1)
        ######################

        loss_matrix = loss_mse + loss_chamfer 
        #loss_matrix = loss_chamfer 

        return {'MSE_mean': loss_mse.mean(), 'Chamfer_mean': loss_chamfer.mean(), 'matrix': loss_matrix} 
    

    ################## just network without feature (shared optimizer)
    def forward(self, pts, mask, shared_learnable_tokens = False, noaug = False):

        neighborhood, center, neighborhood_org= self.group_divider(pts)

        #x_vis = self.forward_encoder_point(neighborhood, center, mask)  # returned mask may change
        x_vis = self.MAE_encoder(neighborhood, center, mask)

        B,_,C = x_vis.shape # B VIS C

        ################# ****
        if (noaug == True):
            return x_vis
        #################

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1) 
        x_full = torch.cat([x_vis, mask_token], dim=1)
        #loss_pred = torch.cat([x_vis, mask_token_loss_pred], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        if (shared_learnable_tokens == False):
            mask_token_loss_pred = self.mask_token_loss_pred.expand(B, N, -1)
            loss_pred = torch.cat([x_vis, mask_token_loss_pred], dim=1)
        else:    
            loss_pred = x_full.clone()
        #loss_pred = pos_full.clone()
        #loss_pred = x_full.clone()

        # apply reconstructor
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim_just_network_without_feature(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024
        #rebuild_points = self.increase_dim_features(x_rec.transpose(1, 2)).transpose(1, 2)  # B M 1024

        # apply loss predictor
        loss_pred_ = self.MAE_decoder_loss_pred(loss_pred, pos_full, N)
        loss_pred_f = self.increase_dim_2(loss_pred_.transpose(1, 2)).transpose(1, 2) # B M 1024

        out = {
            #'pix_pred': rebuild_points,
            'pix_pred': x_rec,
            'mask': mask,
            'mask_num': pos_emd_mask.shape[1],
            'features': x_vis,
            'loss_pred': loss_pred_f.mean(dim=-1),
            'neighborhood': neighborhood,
            'neighborhood_org': neighborhood_org,
            'center': center,
        }

        return out


    @torch.no_grad()
    def generate_mask(self, loss_pred, mask_ratio=0.75, images=None, guide=True, epoch=0, total_epoch=200, after_200_epoch = None):
        N, L = loss_pred.shape
        len_keep = int(L * (1 - mask_ratio))

        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)    # (N, L)

        # keep `keep_ratio` loss and `1 - keep_ratio` random
        keep_ratio = 0.5
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()

        if guide:
            if (after_200_epoch):
            #keep_ratio = float((epoch + 1) / total_epoch) * 0.5
                keep_ratio = min(float((epoch + 1) / (total_epoch / 2)) * 0.5 , 0.5)
            else:
                #keep_ratio = float((epoch + 1) / total_epoch) * 0.5  
                #keep_ratio = float((epoch + 1) / total_epoch) * 0.4 
                #keep_ratio = float((epoch + 1) / total_epoch) * 0.6 
                #keep_ratio = float((epoch + 1) / total_epoch) * 0.7   
                keep_ratio = float((epoch + 1) / total_epoch) * 0.8
            #keep_ratio = float((epoch + 1) / total_epoch) * 0.9

        ## top 0 -> 0.5
        if int((L - len_keep) * keep_ratio) <= 0:
            # random
            noise = torch.randn(N, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                ## mask top `keep_ratio` loss and `1 - keep_ratio` random
                len_loss = int((L - len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=loss_pred.device)
        mask[:, :len_keep] = 0
        # unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_learning_loss(self, loss_pred, mask, loss_target, relative=False):
        
        #loss_pred: [N, L, 1]
        #mask: [N, L], 0 is keep, 1 is remove,
        #loss_target: [N, L]
        
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)

        if relative:
            # binary classification for LxL
            labels_positive = loss_target.unsqueeze(1) > loss_target.unsqueeze(2)
            labels_negative = loss_target.unsqueeze(1) < loss_target.unsqueeze(2)
            labels_valid = labels_positive + labels_negative

            loss_matrix = loss_pred.unsqueeze(1) - loss_pred.unsqueeze(2)
            loss = - labels_positive.int() * torch.log(torch.sigmoid(loss_matrix) + 1e-6) \
                   - labels_negative.int() * torch.log(1 - torch.sigmoid(loss_matrix) + 1e-6)

            return loss.sum() / labels_valid.sum()

        else:
            # normalize by each image
            mean = loss_target.mean(dim=1, keepdim=True)
            var = loss_target.var(dim=1, keepdim=True)
            loss_target = (loss_target - mean) / (var + 1.e-6) ** .5  # [N, L, 1]

            loss = (loss_pred - loss_target) ** 2
            loss = loss.mean()
            return loss





def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


######## Point_Cloud
class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x    
    
class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def fps(self, data, number):
        '''
            data B N 3
            number int
        '''
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_data

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = self.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_org = neighborhood
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, neighborhood_org  

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        #x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        x = self.head(self.norm(x[:, :]))
        return x
    
class TransformerDecoder_2(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x)

        #x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        x = self.head(self.norm(x[:, :]))
        return x    


class MaskTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = 32
        self.decoder_depth = 4
        self.decoder_num_heads = 6
        self.num_group = 64

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
        embed_dim = self.trans_dim,
        depth = self.depth,
        drop_path_rate = dpr,
        num_heads = self.num_heads,
        )
        self.norm_p = nn.LayerNorm(self.trans_dim)

    def forward(self, neighborhood, center, mask):

        ####### Point_Cloud
        group_input_tokens = self.encoder(neighborhood)  #  B G C
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~mask].reshape(batch_size, -1, C)
        masked_center = center[~mask].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm_p(x_vis)
        ####### Point_Cloud

        return x_vis 
       



