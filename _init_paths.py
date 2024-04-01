from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
# lib_path = osp.join(this_dir, 'code/lib_clip')
# lib_path = osp.join(this_dir, 'code/lib_dualcoop_in_taidptv2')
# lib_path = osp.join(this_dir, 'code/lib_clip_mlic')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_mlic')

# lib_path = osp.join(this_dir, 'code/lib_slotatt_clip_mlic')
# lib_path = osp.join(this_dir, 'code/lib_clip_mlic')
# lib_path = osp.join(this_dir, 'code/lib_clip_mlic_fc')
# lib_path = osp.join(this_dir, 'code/lib_clip_mlic_cap')
# lib_path = osp.join(this_dir, 'code/lib_clip_mlic_cap_kap_ifm')
# lib_path = osp.join(this_dir, 'code/lib_clip_pvlr')

# lib_path = osp.join(this_dir, 'code/lib_vit_pvlr')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_fc')
# lib_path = osp.join(this_dir, 'code/lib_dino_vit_fc')
# lib_path = osp.join(this_dir, 'code/lib_dinov2_vit_fc')

# lib_path = osp.join(this_dir, 'code/lib_imagent_resnet_fc_frozen')

# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt_slotatt')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt_transformer')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt_slotatt_768')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt_mldecoder')
# lib_path = osp.join(this_dir, 'code/lib_clip_vit_dual_prompt_avgpool')

lib_path = osp.join(this_dir, 'code/lib_manba_fc')

add_path(lib_path)
