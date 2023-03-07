# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import json
import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from sgg_post_process import custom_sgg_post_precessing_and_save

def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # relation detection needs the targets
                output = model(images.to(device), targets)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
    torch.cuda.empty_cache()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(cfg, model, data_loader, box_only=False, device="cuda", logger = None):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on custom dataset ({} images).".format(len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    # if cfg.TEST.CUSTUM_EVAL:
    custom_sgg_post_precessing_and_save(predictions, save_folder = cfg.DETECTED_SGG_DIR)
    print('SAVED')
    return -1.0



# def _custom_sgg_post_precessing_and_save(predictions, save_folder, rel_max_num = 10):
#     mkdir(save_folder)
#     images_path = json.load(open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json')))['idx_to_files']
#     for idx, boxlist in tqdm(enumerate(predictions)):
#         xyxy_bbox = boxlist.convert('xyxy').bbox
#         # current sgg info
#         current_dict = {}
#         # sort bbox based on confidence
#         sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
#         # sorted bbox label and score
#         bbox = []
#         bbox_labels = []
#         bbox_scores = []
#         for i in sortedid:
#             bbox.append(xyxy_bbox[i].tolist())
#             bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
#             bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
#         current_dict['bbox'] = bbox
#         current_dict['bbox_labels'] = bbox_labels
#         current_dict['bbox_scores'] = bbox_scores
#         # sorted relationships
#         rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
#         # sorted rel
#         rel_pairs = []
#         rel_labels = []
#         rel_scores = []
#         rel_all_scores = []
#         for i in rel_sortedid:
#             rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
#             rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
#             rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
#             old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
#             rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
#         current_dict['rel_pairs'] = rel_pairs
#         current_dict['rel_labels'] = rel_labels
#         current_dict['rel_scores'] = rel_scores
#         current_dict['rel_all_scores'] = rel_all_scores
#         torch.save(current_dict, os.path.join(save_folder, os.path.splitext(os.path.basename(images_path[idx]))[0] + '.pkl'))

