import torch, json, os
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg

def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted


def custom_sgg_post_precessing_and_save(predictions, save_folder, rel_topk = 8, box_score_thr = 0.1, rel_score_thr = 0.03):
    # mkdir(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    images_path = json.load(open(os.path.join(cfg.DETECTED_SGG_DIR, 'data_info.json')))['idx_to_files']
    results = {}
    for idx, boxlist in tqdm(enumerate(predictions)):
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # filt the resutls according to the score of relation and box
        rel_all_scores = boxlist.get_field('pred_rel_scores') # rel_labels > 0 means foreground relation
        rel_pairs = boxlist.get_field('rel_pair_idxs')
        box_scores = boxlist.get_field('pred_scores')
        # get the filted index
        is_selected = (rel_all_scores[:,1:].max(1)[0] > rel_score_thr).bitwise_and(box_scores[rel_pairs[:,0]] > box_score_thr).bitwise_and(box_scores[rel_pairs[:,1]] > box_score_thr)
        selected_idx = torch.where(is_selected > 0)[0]
        if len(selected_idx) == 0: # no relations is selected, just use one relation
            selected_idx = ((rel_all_scores[:, 1:].max(1)[0]).max(0)[1]).view(1)
        elif len(selected_idx) > rel_topk:  # find the topk
            # rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
            rel_sortedid, _ = get_sorted_bbox_mapping(rel_all_scores[selected_idx][:,1:].max(1)[0].tolist())
            selected_idx = torch.as_tensor(selected_idx[rel_sortedid[: rel_topk]])
        else:
            pass
        selected_rel_pairs = rel_pairs[selected_idx]
        selected_rel_scores = rel_all_scores[selected_idx][:, 1:].max(1)[0]
        selected_rel_labels = rel_all_scores[selected_idx][:, 1:].max(1)[1] + 1
        selected_rel_all_scores = rel_all_scores[selected_idx]
        box_ids = torch.unique(selected_rel_pairs, sorted = False, return_inverse = False)
        temp_rel_pairs = selected_rel_pairs.clone()
        for i in range(len(box_ids)):
            selected_rel_pairs[temp_rel_pairs == box_ids[i]] = i
        selected_box = xyxy_bbox[box_ids]
        selected_box_labels = boxlist.get_field('pred_labels')[box_ids]
        selected_box_scores = boxlist.get_field('pred_scores')[box_ids]
        # calculate the weight
        current_dict['area'] = get_area_ratio(selected_box, selected_rel_pairs, boxlist.size)
        # save the result
        current_dict['box'] = selected_box
        # current_dict['box_scores'] = selected_box_scores
        current_dict['box_labels'] = selected_box_labels
        current_dict['rel_pairs'] = selected_rel_pairs
        # current_dict['rel_scores'] = selected_rel_scores
        current_dict['rel_labels'] = selected_rel_labels
        # current_dict['rel_all_scores'] = selected_rel_all_scores
        results[os.path.splitext(os.path.basename(images_path[idx]))[0]] = current_dict
    torch.save(results, os.path.join(save_folder, 'filted_scene_graphs.pkl'))

    # torch.save(current_dict, os.path.join(save_folder, os.path.splitext(os.path.basename(images_path[idx]))[0] + '.pkl'))

def get_area_ratio(boxes, pair_ids, img_size):
    union_areas = get_union_area(boxes, pair_ids)
    total_area = img_size[0] * img_size[1]
    return union_areas / total_area

def get_union_area(boxes, pair_ids):
    boxes_A = boxes[pair_ids[:,0]]
    boxes_B = boxes[pair_ids[:,1]]
    if torch.sum(boxes_A[:, 0] > boxes_A[:, 2]) > 0 or torch.sum(boxes_A[:, 1] > boxes_A[:, 3]) > 0 or torch.sum(boxes_B[:, 0] > boxes_B[:, 2]) > 0 or torch.sum(boxes_B[:, 1] > boxes_B[:, 3]) > 0:
        raise RuntimeError('x1y1 should be smaller than x2y2')
    left_top = torch.maximum(boxes_A[:, :2], boxes_B[:, 2:])
    right_down = torch.minimum(boxes_A[:, :2], boxes_B[:, 2:])
    wh = right_down - left_top
    wh = torch.clamp(wh, min = 0)
    inter = wh[:, 0] * wh[:, 1]
    area_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    area_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    return area_A + area_B - inter

