import argparse, torch, os, collections, copy
from tqdm import tqdm


def main(opts):
    ranking_result = torch.load(opts.ranking_result_path)
    cat_info = torch.load(opts.cat_info_path) # a dict: {image_name: label}
    save_dir = os.path.join(opts.out_dir, 'ranking_check/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # tranform the cat_info to label_index
    cat_count = collections.defaultdict(set) # a dict: {label: set(image_names)}
    for name, label in cat_info.items():
        cat_count[label].add(name)
    for src_name, ranking_scores in tqdm(ranking_result.items()):
        curr_label = cat_info[src_name]
        curr_gt_set = copy.deepcopy(cat_count[curr_label])
        curr_gt_set.remove(src_name)
        curr_dir = os.path.join(save_dir, src_name)
        tp_dir = os.path.join(curr_dir, 'TP/')
        fp_dir = os.path.join(curr_dir, 'FP/')
        fn_dir = os.path.join(curr_dir, 'FN/')
        if not os.path.exists(tp_dir):
            os.makedirs(tp_dir)
        if not os.path.exists(fp_dir):
            os.makedirs(fp_dir)
        if not os.path.exists(fn_dir):
            os.makedirs(fn_dir)
        for tgt_name in ranking_scores.keys():
            if cat_info[src_name] == cat_info[tgt_name]:
                os.system('ln -s {} {}'.format(os.path.join(opts.src_image_dir, tgt_name + '.jpg'), os.path.join(tp_dir, tgt_name + '.jpg')))
                curr_gt_set.remove(tgt_name)
            else:
                os.system('ln -s {} {}'.format(os.path.join(opts.src_image_dir, tgt_name + '.jpg'), os.path.join(fp_dir, tgt_name + '.jpg')))
        for fn_name in curr_gt_set:
            os.system('ln -s {} {}'.format(os.path.join(opts.src_image_dir, fn_name + '.jpg'), os.path.join(fn_dir, fn_name + '.jpg')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking_result_path', type = str, default = '/home/lzj/code/sgg/outputs/ranking_result_filted.pkl')
    parser.add_argument('--cat_info_path', type = str, default = '/home/lzj/code/sgg/outputs/categorize_info_raw.pkl')
    parser.add_argument('--src_image_dir', type = str, default = '/home/lzj/datasets/VisualGenome/VG_100K')
    parser.add_argument('--out_dir', type = str, default = '/home/lzj/code/sgg/outputs')
    opts = parser.parse_args()
    main(opts)