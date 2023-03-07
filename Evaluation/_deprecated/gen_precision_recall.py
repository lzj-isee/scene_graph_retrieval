import torch, os, matplotlib.pyplot as plt
from tqdm import tqdm


def calc_precision_recall(relevance_gt, relevance_pred, k):
    precision, recall = 0, 0
    for item in relevance_gt.keys():
        assert item in relevance_pred, 'item {} not in prediction'.format(item)
        predictions = relevance_pred[item]
        ground_truth = relevance_gt[item]
        if len(predictions) > k:
            predictions = predictions[:k]
        count = 0
        for pred in predictions:
            if pred in ground_truth:
                count += 1
        if len(predictions) == 0:
            debug = 1
        precision += count / len(predictions) if len(predictions) > 0 else 0
        recall += count / len(ground_truth) if len(predictions) > 0 else 0
    precision /= len(relevance_gt.keys()) / 100
    recall /= len(relevance_gt.keys()) / 100
    return precision, recall


def main(relevance_gt_path, relevance_pred_path):
    relevance_gt = torch.load(relevance_gt_path)
    relevance_pred = torch.load(relevance_pred_path)
    for key, items in list(relevance_pred.items()):
        temp = sorted(items.items(), key = lambda x: x[1], reverse = True)
        relevance_pred[key] = [key for key, val in temp if val > 0]
    ans = {5: {'p': -1, 'r': -1}, 10: {'p': -1, 'r': -1}, 20: {'p': -1, 'r': -1}, 50: {'p': -1, 'r': -1}}
    for k in tqdm(ans.keys()):
        precision, recall = calc_precision_recall(relevance_gt, relevance_pred, k)
        ans[k]['p'] = precision
        ans[k]['r'] = recall
    for k in ans.keys():
        print('Precision@{}: {:.2f}; Recall@{}: {:.2f}'.format(k, ans[k]['p'], k, ans[k]['r']))
    # print(ans)


if __name__ == '__main__':
    relevance_gt_path = '/home/lzj/code/sgg/outputs/relevance_multi.pkl'
    relevance_pred_path = '/home/lzj/code/sgg/outputs/ranking_results.pkl'
    main(relevance_gt_path, relevance_pred_path)