import os
import math
import torch

import torch.nn as nn
import os.path as osp
import torchvision
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict

import numpy as np
import torch.distributed as dist

from utils.misc import is_dist_avail_and_initialized, get_world_size
from utils.label_tool import AverageMeter, compute_pseudo_label_accuracy
from utils.util import cal_gpu

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

from .mlp import SimpleMLP
from .mamba import build_model


class VmambaMLIC(nn.Module):
    def __init__(self, cfg, classnames, pretrain_model, return_interm_layers=False):    
        super().__init__()

        # visual_encoder
        self.visual_encoder = pretrain_model        
        if 'vssm1_small' in cfg.MODEL.arch:
            self.visual_encoder.classifier.head = nn.Linear(768, cfg.MODEL.CLASSIFIER.num_class)
        elif 'vssm1_base' in cfg.MODEL.arch:
            self.visual_encoder.classifier.head = nn.Linear(1024, cfg.MODEL.CLASSIFIER.num_class)

        self.vis = False
        self.save_flag = True
        self.memorybank_flag = False
        self.is_vis_prototype_neg = True
        self.num_class = cfg.MODEL.CLASSIFIER.num_class
        self.if_pos = True
        self.group_loss_weight = 0.5

        print('=================================================================================================')
        print('==============  VMamba: Visual State Space Model for Multi-Label Image Recognition ==============')
        print('=================================================================================================')
        print('self.start_epoch: ', cfg.TRAIN.start_epoch)
        print('cfg.DATA.len_train_loader:', cfg.DATA.len_train_loader)

    def switch_mode_train(self):
        self.training = True
        self.train()

    def switch_mode_eval(self):
        self.training = False
        self.eval()


    # @torch.no_grad()
    def encode_image(self, x):
        dict_out = self.visual_encoder.forward(x)
        return dict_out
        # x_norm_clstoken = dict_out['x_norm_clstoken']
        # return x_norm_clstoken


    @torch.no_grad()
    def forward_encoder_and_val(self, image):
        logit = self.encode_image(image)
        # logit = self.classifier(x_norm_clstoken)
        return {'logit': logit}

    def forward_encoder_and_train(self, image):
        logit = self.encode_image(image)
        # logit = self.classifier(x_norm_clstoken)
        return {'logit': logit}


    def forward(self, x, is_val=True):
        if is_val:
            return self.forward_encoder_and_val(x)
        else:
            return self.forward_encoder_and_train(x)


def do_forward_and_criterion_train(cfg, data_full, model, criterion, is_val):

    image, label = data_full['image'], data_full['target']
    targets = {'label': label}

    outputs = model(x=image, is_val=False)

    loss_dict = criterion(outputs, targets, is_val=False)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return outputs['logit'], losses, loss_dict, weight_dict

@torch.no_grad()
def do_forward_and_criterion_test(cfg, data_full, model, criterion, is_val):

    image, label = data_full['image'], data_full['target']
    targets = {'label': label}

    outputs = model(x=image, is_val=True)

    loss_dict = criterion(outputs, targets, is_val=True)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return outputs['logit'], losses, loss_dict, weight_dict


def do_forward_and_criterion(cfg, data_full, model, criterion, is_val):
    if not is_val:
        return do_forward_and_criterion_train(cfg, data_full, model, criterion, is_val)
    else:
        return do_forward_and_criterion_test(cfg, data_full, model, criterion, is_val)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def build_NetWork(cfg):
    classnames = cfg.DATA.classnames

    vmanba_model = build_model(cfg, is_pretrain=True)
    checkpoint = torch.load(cfg.MODEL.VSSM.is_pretrain, map_location=torch.device(dist.get_rank()))
    # for k, v in checkpoint['model'].items():
    #     print(k)
    vmanba_model.load_state_dict(checkpoint['model'], strict=True)

    model = VmambaMLIC(cfg, classnames, pretrain_model=vmanba_model, return_interm_layers=False)
    print("Turning off gradients in both the image and the text encoder")

    # model.eval()
    # for name, param in model.named_parameters():
    #     if "classifier" in name:
    #         print(name)
    #         param.requires_grad_(True)
    #     else:
    #         param.requires_grad_(False)

    # for name, param in model.named_parameters():
    #     param.requires_grad_(True)


    return model

class SetCriterion(nn.Module):

    def __init__(self, weight_dict, losses, cfg):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.cls_loss = None

    def loss_cls(self, outputs, targets=None, **kwargs):
        
        # print(targets['label'][0])
        cls = self.cls_loss(outputs['logit'], targets['label'])
        losses = {"loss_cls": cls}

        return losses

    def loss_kcr(self, outputs, targets=None, **kwargs):
        
        # print(targets['label'][0])
        losses = {"loss_kcr": outputs['kcr']}

        return losses

    def get_loss_train(self, loss, outputs, targets, **kwargs):

        loss_map = {
            'cls': self.loss_cls,
            'kcr': self.loss_kcr,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)


    def forward(self, outputs, targets=None, is_val=None, **kwargs):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss_train(loss, outputs, targets, **kwargs))
        
        return losses


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "voc2007": "a photo of a {}.",
    "voc2012": "a photo of a {}.",
    "coco14": "a photo of a {}.",
    "coco17": "a photo of a {}.",
    "nuswide": "a photo of a {}.",
    "vg500": "a photo of a {}."
}
