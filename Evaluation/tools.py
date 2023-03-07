import collections

def calc_precision_recall(ranking_result, cat_gt):
    pr_result = {}
    label_count = collections.Counter(cat_gt.values())
    for src_name, ranking_scores in ranking_result.items():
        count = 0
        for tgt_name in ranking_scores.keys():
            if cat_gt[src_name] == cat_gt[tgt_name]:
                count += 1
        pr_result[src_name] = {}
        pr_result[src_name]['precision'] = count / len(ranking_scores) * 100 if len(ranking_scores) > 0 else 0
        pr_result[src_name]['recall'] = count / label_count[cat_gt[src_name]] * 100 if label_count[cat_gt[src_name]] > 0 else 0
    precision, recall = 0, 0
    mean_result = {}
    for item_name, __result in pr_result.items():
        precision += __result['precision']
        recall += __result['recall']
        __label = cat_gt[item_name]
        if __label not in mean_result:
            mean_result[__label] = {'precision': __result['precision'], 'recall': __result['recall'], 'num': 1}
        else:
            mean_result[__label]['precision'] += __result['precision']
            mean_result[__label]['recall'] += __result['recall']
            mean_result[__label]['num'] += 1
    precision /= len(pr_result)
    recall /= len(pr_result)
    for label in mean_result.keys():
        mean_result[label]['precision'] /= mean_result[label]['num']
        mean_result[label]['recall'] /= mean_result[label]['num']
    eval_result = {'detail': pr_result, 'mean_result':mean_result, 'precision': precision, 'recall': recall}
    return eval_result

def filt_ranking_result(ranking_result, thr = 0.1):
    for src_name, ranking_scores in ranking_result.items():
        ranking_result[src_name] = dict([(key, val) for key, val in ranking_scores.items() if val > thr])
    return ranking_result