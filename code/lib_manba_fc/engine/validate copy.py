import torch
import time
import torch
import time
import os
import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from utils.meter import AverageMeter, ProgressMeter
from utils.misc import concat_all_gather, MetricLogger, SmoothedValue, reduce_dict
from utils.hpc import pin_workers_iterator
from utils.metric import voc_mAP, asl_mAP
from models.builder_clip import do_forward_and_criterion

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, cfg, logger):

    model.eval()
    criterion.eval()
    # for k, v in criterion.items():
    #     criterion[k] = v.eval()
    
    saved_data = []
    saved_data_aux = []

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    the_iterator = iter(val_loader)

    # _cnt = 0
    # output_state_dict = {} # for debug only

    with torch.no_grad():
        for it, data_full in enumerate(metric_logger.log_every(the_iterator, cfg.INPUT_OUTPUT.print_freq, header, logger=logger)):
            images = data_full['image'].cuda(non_blocking=True)
            target = data_full['target'].cuda(non_blocking=True)
            target_full = data_full['target_full'].cuda(non_blocking=True)
            lable = target.clone()

            # compute output
            if cfg.TRAIN.amp:
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp):
                    output, losses, loss_dict, weight_dict = do_forward_and_criterion(cfg, images, target, model, criterion, True, target_full)
            else:
                output, losses, loss_dict, weight_dict  = do_forward_and_criterion(cfg, images, target, model, criterion, True, target_full)

            output_sm = output
            # output_aux = logits_l_aux
        
            #  save some data            
            lable[lable < 0] = 0
            _item = torch.cat((output_sm.detach().cpu().data, lable.detach().cpu().data), 1)
            saved_data.append(_item)
            _item = torch.cat((output_sm.detach().cpu().data, lable.detach().cpu().data), 1)
            saved_data_aux.append(_item)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()


        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        # metric_logger.update(all_loss=loss.item())
        # metric_logger.update(div_loss=div_loss.item())


        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()


        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(cfg.INPUT_OUTPUT.output, saved_name), saved_data)

        saved_data_aux = torch.cat(saved_data_aux, 0).numpy()
        saved_name = 'saved_aux_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(cfg.INPUT_OUTPUT.output, saved_name), saved_data_aux)

        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            logger.info("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP
            mAP, aps = metric_func([os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist], cfg.DATA.num_class, return_each=True, logger=logger)
            logger.info("  mAP: {}".format(mAP))

            logger.info("Calculating aux mAP:")
            filenamelist = ['saved_aux_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP
            mAP_aux, aps_aux = metric_func([os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist], cfg.DATA.num_class, return_each=True, logger=logger)
            logger.info("  aux mAP: {}".format(mAP_aux))

            if cfg.INPUT_OUTPUT.out_aps:
                logger.info("  aux aps: {}".format(np.array2string(aps_aux, precision=5)))
        else:
            mAP = 0
            mAP_aux = 0

        if dist.get_world_size() > 1:
            dist.barrier()

        mAP = max(mAP, mAP_aux)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat, mAP
