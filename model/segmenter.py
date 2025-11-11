import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from .layers import HA # Restore Neck (HA)
from .bridger import Bridger_RN as Bridger_RL, Bridger_ViT as Bridger_VL
from .lar_decoder import LAR_Net_Decoder_V27 

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    if pred.dtype != target.dtype:
        target = target.type_as(pred)
    pred_sigmoid = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal_term = (1 - pt).pow(gamma)
    alpha_t = torch.full_like(target, 1 - alpha)
    alpha_t[target == 1] = alpha
    loss = (alpha_t * focal_term * bce).mean()
    return loss

def dice_loss(pred, target, smooth=1.):
    pred_sigmoid = torch.sigmoid(pred)
    iflat = pred_sigmoid.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_coeff

class ETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
            self.bridger = Bridger_RL(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage)
        else:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            self.bridger = Bridger_VL(d_model=cfg.ladder_dim, nhead=cfg.nhead)
        
        for param_name, param in self.backbone.named_parameters():
            if 'positional_embedding' not in param_name:
                param.requires_grad = False       

        self.neck = HA(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        
        self.decoder = LAR_Net_Decoder_V27(
            fpn_in_dims=cfg.fpn_in,
            fpn_out_dim=cfg.vis_dim,
            word_dim=cfg.word_dim,
            state_dim=cfg.word_dim,
            n_iter=getattr(cfg, 'n_iter', 2)
        )
        self.aux_weights = getattr(cfg, 'aux_weights', [0.2, 0.5])
        self.label_smoothing = getattr(cfg, 'label_smoothing', 0.1)
    def forward(self, img, word, mask=None):
        vis_outs, word_features, state_feature = self.bridger(img, word, self.backbone)
        
        fq_context = self.neck(vis_outs, state_feature)
        
        decoder_outputs = self.decoder(vis_outs, fq_context, word_features, state_feature)
        main_logit, aux_logits = decoder_outputs

        if self.training:
            target_mask = F.interpolate(mask, size=main_logit.shape[-2:], mode='nearest').detach()
            
            if self.label_smoothing > 0.0:
                smooth_target_mask = target_mask * (1.0 - self.label_smoothing) + self.label_smoothing / 2
            else:
                smooth_target_mask = target_mask
            
            main_loss = focal_loss(main_logit, smooth_target_mask) + dice_loss(main_logit, target_mask)
            
            total_loss = main_loss
            
            if len(aux_logits) > 0:
                for i, logit in enumerate(aux_logits):
                    if i < len(self.aux_weights):
                        aux_target_mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest').detach()
                        aux_loss = focal_loss(logit, aux_target_mask) + dice_loss(logit, aux_target_mask)
                        total_loss += self.aux_weights[i] * aux_loss
            
            return main_logit.detach(), target_mask, total_loss
        else:
            return main_logit.detach()
