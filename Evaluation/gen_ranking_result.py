import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from GraphMatching.tools import matching
import argparse, torch, json
from GraphMatching.core.common import synchronize
from tools import calc_precision_recall, filt_ranking_result


def get_ranking_result(scene_graphs):
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend = "gloo", init_method = "env://", 
        rank = local_rank, world_size = world_size,
    )
    if local_rank == 0:
        ranking_result = {}
    querys = list(scene_graphs.keys())
    for i, src_name in enumerate(querys):
        matching_score = matching(scene_graphs[src_name], scene_graphs)
        if local_rank == 0:
            print(f'\rCalculating: {i + 1}/{len(querys)}', end = '')
            matching_score = dict([(key, val) for key, val in matching_score.items() if val > 0])
            if src_name in matching_score:
                matching_score.pop(src_name)
            ranking_result[src_name] = matching_score
    if local_rank == 0:
        return ranking_result


def main(opts):
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    scene_graphs = torch.load(opts.scene_graphs_path)
    cat_info = torch.load(opts.categorize_info_path)
    scene_graphs = dict([(key, val) for key, val in scene_graphs.items() if key in cat_info])
    if local_rank == 0: print(f'Total items: {len(scene_graphs)}')
    ranking_result = get_ranking_result(scene_graphs)
    synchronize()
    if local_rank == 0:
        assert len(ranking_result) == len(cat_info), 'Something Wrong, lost some items during matching'
        if not os.path.exists(opts.out_dir):
            os.makedirs(opts.out_dir)
        print('Save the ranking result.')
        torch.save(ranking_result, os.path.join(opts.out_dir, 'ranking_result.pkl'))
        ranking_result = filt_ranking_result(ranking_result)
        print('Save the filted ranking result with a thr')
        torch.save(ranking_result, os.path.join(opts.out_dir, 'ranking_result_filted.pkl'))
        # start the evaluation
        print('Evaluation')
        eval_result = calc_precision_recall(ranking_result, cat_info)
        print('Save the evaluation result, precision: {:.2f}, recall: {:.2f}'.format(eval_result['precision'], eval_result['recall']))
        torch.save(eval_result, os.path.join(opts.out_dir, 'evaluation_result.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_graphs_path', type = str, default = '/home/lzj/code/sgg/outputs/filted_scene_graphs.pkl')
    parser.add_argument('--categorize_info_path', type = str, default = '/home/lzj/code/sgg/outputs/categorize_info_raw.pkl')
    # parser.add_argument('--src_image_dir', type = str, default = '/home/lzj/datasets/VisualGenome/VG_100K')
    parser.add_argument('--out_dir', type = str, default = '/home/lzj/code/sgg/outputs')
    opts = parser.parse_args()
    main(opts)

