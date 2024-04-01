# ------------------------------------------------------------------------
# Modified from Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import yaml
import copy
# from yacs.config import CfgNode as CN
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config Settings
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODE settings
# -----------------------------------------------------------------------------
_C.MODE = CN()
_C.MODE.name = 'PVLR' # ['PVLR', 'PVLR_Caption', 'PVLR_SemanticGrouping', 'PVLR_SelfTrain', 'PVLR_SelfTrain_Caption', 'PVLR_SelfTrain_SemanticGrouping']
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.dataset_dir = '/media/data2/MLICdataset'
_C.DATA.dataname = 'coco14' # ['coco14', 'voc2007', 'voc2012', 'nus_wide', 'nuswide', 'vg500']
_C.DATA.num_workers = 8
_C.DATA.num_class = 80
_C.DATA.len_train_loader = -1
_C.DATA.classnames = []

# data aug
_C.DATA.TRANSFORM = CN()
_C.DATA.TRANSFORM.img_size = 448
_C.DATA.TRANSFORM.crop = False
_C.DATA.TRANSFORM.cutout = True
_C.DATA.TRANSFORM.length = 224
_C.DATA.TRANSFORM.cut_fact = 0.5
_C.DATA.TRANSFORM.orid_norm = False
_C.DATA.TRANSFORM.remove_norm = False
_C.DATA.TRANSFORM.n_holes = -1

_C.DATA.TRANSFORM.TWOTYPE = CN()
_C.DATA.TRANSFORM.TWOTYPE.is_twotype = False
_C.DATA.TRANSFORM.TWOTYPE.img_size = 448
_C.DATA.TRANSFORM.TWOTYPE.min_scale = 0.08

# -----------------------------------------------------------------------------
# dataload settings
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)                 # Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bilinear"        # For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()                   # If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False              

_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]# Default mean and std come from ImageNet
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.INPUT.CROP_PADDING = 4                  # Padding for random crop

_C.INPUT.CUTOUT_N = 1                      # Cutout 
_C.INPUT.CUTOUT_LEN = 16

_C.INPUT.GN_MEAN = 0.0                     # Gaussian noise
_C.INPUT.GN_STD = 0.15

_C.INPUT.RANDAUGMENT_N = 2                 # RandomAugment
_C.INPUT.RANDAUGMENT_M = 10

_C.INPUT.COLORJITTER_B = 0.4               # ColorJitter (brightness, contrast, saturation, hue)
_C.INPUT.COLORJITTER_C = 0.4
_C.INPUT.COLORJITTER_S = 0.4
_C.INPUT.COLORJITTER_H = 0.1

_C.INPUT.RGS_P = 0.2                       # Random gray scale's probability
# Gaussian blur
_C.INPUT.GB_P = 0.5  # propability of applying this operation
_C.INPUT.GB_K = 21  # kernel size (should be an odd number)


# several param for spacific transform setting
_C.INPUT.random_resized_crop_scale = (0.8, 1.0)
_C.INPUT.cutout_proportion = 0.4
_C.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.arch = 'vit_small'
_C.MODEL.use_BN = False

# Model type
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.backbone = 'dino_vits16'
_C.MODEL.BACKBONE.pretrained = True
_C.MODEL.BACKBONE.frozen_backbone = False

# Model type
_C.MODEL.NUM_CLASSES = 1000 # for manba
_C.MODEL.TYPE='vssm'
_C.MODEL.NAME = 'vssm1_base_0229'
_C.MODEL.DROP_PATH_RATE = 0.6
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.is_pretrain = ''
_C.MODEL.VSSM.PATCH_SIZE= 4
_C.MODEL.VSSM.IN_CHANS= 3
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.GMLP = False

_C.MODEL.VSSM.EMBED_DIM= 128
_C.MODEL.VSSM.DEPTHS=[ 2, 2, 15, 2 ]
_C.MODEL.VSSM.SSM_D_STATE=1
_C.MODEL.VSSM.SSM_DT_RANK="auto"
_C.MODEL.VSSM.SSM_RATIO=2.0
_C.MODEL.VSSM.SSM_CONV=3
_C.MODEL.VSSM.SSM_CONV_BIAS=False
_C.MODEL.VSSM.SSM_FORWARDTYPE="v3noz"
_C.MODEL.VSSM.MLP_RATIO=4.0
_C.MODEL.VSSM.DOWNSAMPLE="v3"
_C.MODEL.VSSM.PATCHEMBED="v2"


# Transformer
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.enc_layers = 1
_C.MODEL.TRANSFORMER.dec_layers = 2
_C.MODEL.TRANSFORMER.dim_feedforward = 8192
_C.MODEL.TRANSFORMER.hidden_dim = 2048
_C.MODEL.TRANSFORMER.dropout = 0.1
_C.MODEL.TRANSFORMER.nheads = 4
_C.MODEL.TRANSFORMER.pre_norm = False
_C.MODEL.TRANSFORMER.position_embedding = 'v2'
_C.MODEL.TRANSFORMER.keep_other_self_attn_dec = False
_C.MODEL.TRANSFORMER.keep_first_self_attn_dec = False
_C.MODEL.TRANSFORMER.keep_input_proj = False

# Classifier
_C.MODEL.CLASSIFIER = CN()
_C.MODEL.CLASSIFIER.num_class = 80

# # model for SemanticGrouping
# _C.MODEL.SELFTRAIN = CN()
# _C.MODEL.SELFTRAIN.teacher_momentum = 0.99
# _C.MODEL.SELFTRAIN.center_momentum = 0.9
# _C.MODEL.SELFTRAIN.student_temp = 0.1
# _C.MODEL.SELFTRAIN.teacher_temp = 0.07
# # _C.MODEL.SELFTRAIN.temp = 0.07
# _C.MODEL.SELFTRAIN.temp = 0.2
# _C.MODEL.SELFTRAIN.tau = 0.2




# Caption
_C.MODEL.CAPTION = CN()
_C.MODEL.CAPTION.n_ctx_pos = 16
_C.MODEL.CAPTION.n_ctx_neg = 16
_C.MODEL.CAPTION.csc = True
_C.MODEL.CAPTION.ctx_init_pos = ""
_C.MODEL.CAPTION.ctx_init_neg = ""
_C.MODEL.CAPTION.class_token_position = "end"
_C.MODEL.CAPTION.gl_merge_rate = 0.5
_C.MODEL.CAPTION.n_ctx = 16
_C.MODEL.CAPTION.ctx_init = ""

# -----------------------------------------------------------------------------
# Loss settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.loss_dev = -1
_C.LOSS.loss_mode = 'asl' # ['asl', 'cls', 'bce', 'only_bce', 'SoftMarginLoss']
# for asl loss
_C.LOSS.ASL = CN()
# _C.LOSS.ASL.losses = 'asl'
_C.LOSS.ASL.eps = 1e-05
_C.LOSS.ASL.dtgfl = True
_C.LOSS.ASL.gamma_pos = 0.0
_C.LOSS.ASL.gamma_neg = 2.0
_C.LOSS.ASL.loss_clip = 0.0


_C.LOSS.Coef = CN()
_C.LOSS.Coef.cls_asl_coef = 1.0
_C.LOSS.Coef.cls_kcr_coef = 4.0
# _C.LOSS.Coef.distill = 0.5
# _C.LOSS.Coef.filtered = 0.5
# _C.LOSS.Coef.distillKL = 1.0
# _C.LOSS.Coef.bce = 1.0


# -----------------------------------------------------------------------------
# optimizer settings
# -----------------------------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.optim = 'AdamW'                   # ['AdamW', 'Adam_twd', 'SGD']
_C.OPTIMIZER.lr_scheduler = 'OneCycleLR'       # ['OneCycleLR', 'MultiStepLR', 'ReduceLROnPlateau']
_C.OPTIMIZER.pattern_parameters = 'single_lr'  # ['mutil_lr', 'single_lr', 'add_weight']
_C.OPTIMIZER.momentum = 0.9                    # for SGD
_C.OPTIMIZER.sgd_dampening = 0
_C.OPTIMIZER.sgd_nesterov = False
_C.OPTIMIZER.weight_decay = 0.0005
_C.OPTIMIZER.max_clip_grad_norm = 0
_C.OPTIMIZER.epoch_step = [10, 20]
_C.OPTIMIZER.batch_size = 64
_C.OPTIMIZER.lr = 5e-05
# _C.OPTIMIZER.base_lr = 1.0
_C.OPTIMIZER.lrp = 0.1
_C.OPTIMIZER.lr_mult = 1.0
_C.OPTIMIZER.warmup_scheduler = False
_C.OPTIMIZER.warmup_type = 'linear'
_C.OPTIMIZER.warmup_epoch = 5
_C.OPTIMIZER.warmup_multiplier = 50
_C.OPTIMIZER.warmup_lr = 1e-05

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
# distribution training
_C.DDP = CN()
_C.DDP.world_size = 1
_C.DDP.rank = 0
_C.DDP.dist_url = 'tcp://127.0.0.1:3719'
_C.DDP.local_rank = -1
_C.DDP.gpus = 0

# train
_C.TRAIN = CN()
_C.TRAIN.seed = 1
_C.TRAIN.amp = True
_C.TRAIN.early_stop = True
_C.TRAIN.kill_stop = False
_C.TRAIN.device = 'CUDA'
_C.TRAIN.start_epoch = 0
_C.TRAIN.epochs = 80
_C.TRAIN.ema_decay = 0.9998
_C.TRAIN.ema_epoch = 0
_C.TRAIN.ratio = 1.0
_C.TRAIN.evaluate = False
_C.TRAIN.cpus = None
_C.TRAIN.USE_CHECKPOINT = False

# save, resume and displayer
_C.INPUT_OUTPUT = CN()
_C.INPUT_OUTPUT.output = ''
_C.INPUT_OUTPUT.resume = ''
_C.INPUT_OUTPUT.resume_omit = []
_C.INPUT_OUTPUT.print_freq = 400
_C.INPUT_OUTPUT.out_aps = False

# -----------------------------------------------------------------------------
# Val settings
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.val_interval = 1
_C.EVAL.val_epoch_start = 1



def reset_cfg(cfg, args):
   
    if args.world_size:
        cfg.DDP.world_size = args.world_size    

    if args.rank:
        cfg.DDP.rank = args.rank  

    if args.dist_url:
        cfg.DDP.dist_url = args.dist_url  

    if args.world_size:
        cfg.DDP.local_rank = args.local_rank  


    if args.output:
        cfg.INPUT_OUTPUT.output = args.output 

    if args.resume:
        cfg.INPUT_OUTPUT.resume = args.resume

    if args.resume_omit:
        cfg.INPUT_OUTPUT.resume_omit = args.resume_omit

    if args.print_freq:
        cfg.INPUT_OUTPUT.print_freq = args.print_freq
    
    if args.gpus:
        cfg.DDP.gpus = args.gpus
    
    if args.seed:
        cfg.DDP.seed = args.seed
    
    if args.device:
        cfg.TRAIN.device = args.device
    
    if args.evaluate:
        cfg.TRAIN.evaluate = args.evaluate
    
    if args.prob:
        cfg.DATA.prob = args.prob
    
    if args.print_freq:
        cfg.INPUT_OUTPUT.print_freq = args.print_freq



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    reset_cfg(config, args)
    # config.freeze()

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # post_process(config)
    # config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    args.DATA = config.DATA
    args.MODEL = config.MODEL
    args.LOSS = config.LOSS
    args.OPTIMIZER = config.OPTIMIZER
    args.TRAIN = config.TRAIN
    args.EVAL = config.EVAL
    

    return args, config



def main():
    pass

if __name__ == '__main__':
    main()

        # print(config)
    # print(config)
    # for arg, content in args.__dict__.items():
    #     if arg not in ('DATA', 'MODEL', 'LOSS', 'OPTIMIZER', 'TRAIN', 'EVAL'):
    #         print("{}: {}".format(arg, content))


# def post_process(config):
#     # fix dilation config
#     dilation = config.MODEL.BACKBONE.RESNET.dilation
#     if isinstance(dilation, str):
#         if dilation.lower() == 'false':
#             config.MODEL.BACKBONE.RESNET.dilation = False
#         elif dilation.lower() == 'true':
#             config.MODEL.BACKBONE.RESNET.dilation = True
#         else:
#             raise ValueError("The dilation should be True or False")
#     # SeqDetectHead needs config for attention layer
#     if "Seq" in config.MODEL.OBJECT_DECODER.HEAD.type:
#         old_no_proj = config.MODEL.OBJECT_DECODER.HEAD.cross_attn_no_value_proj
#         config.MODEL.OBJECT_DECODER.HEAD.update(config.MODEL.OBJECT_DECODER.LAYER)
#         config.MODEL.OBJECT_DECODER.HEAD.cross_attn_no_value_proj = old_no_proj

#         config.MODEL.OBJECT_DECODER.HEAD.LOSS.task_category = config.MODEL.OBJECT_DECODER.HEAD.task_category
#         config.MODEL.OBJECT_DECODER.HEAD.LOSS.num_classes = config.MODEL.OBJECT_DECODER.HEAD.num_classes
