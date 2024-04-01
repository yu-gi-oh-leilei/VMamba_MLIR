import torch
import torch.nn as nn
from .bceloss import BinaryCrossEntropyLossOptimized, BCELoss
from .dualcoop_loss import AsymmetricLoss_partial
from .aslloss import AsymmetricLoss, AsymmetricLossOptimized
from .kl_loss import DistillKL
from models.builder_network import SetCriterion

# class SetCriterion(nn.Module):

#     def __init__(self, weight_dict, losses):
#         super().__init__()
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.loss_dualcoop = None

#     def loss_cls(self, outputs, targets, **kwargs):
#         """Classification loss (Binary focal loss)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
        
#         loss_ce = self.loss_dualcoop(outputs, targets)
#         losses = {'loss_ce': loss_ce}

#         return losses

#     def get_loss(self, loss, outputs, targets, **kwargs):
#         loss_map = {
#             'cls': self.loss_cls,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, **kwargs)

#     def forward(self, outputs, targets, **kwargs):
        
#         # losses =None
#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             losses.update(self.get_loss(loss, outputs, targets, **kwargs))
        
#         return losses


def build_criterion(cfg, model):

    # weight_dict = {}
    # for k, v in cfg.LOSS.Coef.items():
    #     weight_dict.update({str(k): v})
    # loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict

    weight_dict = {'loss_cls': cfg.LOSS.Coef.cls_asl_coef, 'loss_kcr': cfg.LOSS.Coef.cls_kcr_coef}
    

    # weight_dict = {'distill': cfg.LOSS.Coef.distill, 'filtered': cfg.LOSS.Coef.filtered, 'bce': cfg.LOSS.Coef.bce}
    # weight_dict = {'distill': cfg.LOSS.Coef.distill, 'filtered': cfg.LOSS.Coef.filtered, \
    #                'distillKL': cfg.LOSS.Coef.distillKL,'bce': cfg.LOSS.Coef.bce}
    # print(weight_dict)

    # losses = ['self_distill', 'ctr_filtered']
    # losses = ['tech_distillKL', 'self_distill', 'ctr_filtered']
    # losses = ['cls', 'kcr']
    losses = ['cls']
    criterion = SetCriterion(weight_dict, losses, cfg)

    if cfg.LOSS.loss_mode == 'asl':
        criterion.cls_loss = AsymmetricLossOptimized(
            gamma_neg=cfg.LOSS.ASL.gamma_neg, 
            gamma_pos=cfg.LOSS.ASL.gamma_pos,
            clip=cfg.LOSS.ASL.loss_clip,
            disable_torch_grad_focal_loss=cfg.LOSS.ASL.dtgfl,
            eps=cfg.LOSS.ASL.eps)

    elif cfg.LOSS.loss_mode == 'bce':
        criterion.cls_loss = BCELoss(reduce=True, size_average=True)

    elif cfg.LOSS.loss_mode == 'multi_bce':
        criterion.cls_loss = nn.MultiLabelSoftMarginLoss()

    device = cal_gpu(model)

    criterion = criterion.to(device)
    
    return criterion

    # criterion = {}
    
    # if cfg.LOSS.loss_mode == 'dualcoop':
    #     criterion['dualcoop'] = AsymmetricLoss_partial(
    #         gamma_neg=cfg.LOSS.DUALCOOP.gamma_neg, 
    #         gamma_pos=cfg.LOSS.DUALCOOP.gamma_pos, 
    #         clip=cfg.LOSS.DUALCOOP.loss_clip, 
    #         eps=cfg.LOSS.DUALCOOP.eps, 
    #         disable_torch_grad_focal_loss=cfg.LOSS.DUALCOOP.dtgfl,
    #         thresh_pos = cfg.LOSS.DUALCOOP.thresh_pos,
    #         thresh_neg = cfg.LOSS.DUALCOOP.thresh_neg
    #         )

    # elif cfg.LOSS.loss_mode == 'bce':
    #     criterion['bce'] = BCELoss(reduce=True, size_average=True)

    # else:
    #     raise NotImplementedError("Unknown loss mode %s" % cfg.LOSS.loss_mode)
    
    # # criterion['consistency'] = Consistencyloss()
    # criterion['klloss'] = DistillKL(T=4)

    # device = cal_gpu(model)
    # if isinstance(criterion, dict):
    #     for k, v in criterion.items():
    #         criterion[k] = v.to(device)



def cal_gpu(module):
    if hasattr(module, 'module') or isinstance(module, torch.nn.DataParallel):
        for submodule in module.module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device
    else:
        for submodule in module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device