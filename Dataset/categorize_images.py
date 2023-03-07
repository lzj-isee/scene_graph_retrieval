import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from GraphMatching.tools import matching, text_to_graph
import argparse, torch, json
from GraphMatching.core.common import synchronize
from tqdm import tqdm

def get_categorize_result(query_text, scene_graphs, data_info, thr = 0.001):
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend = "gloo", init_method = "env://", 
        rank = local_rank, world_size = world_size,
    )
    if local_rank == 0:
        categorize_result = {}
    for i, text in enumerate(query_text):
        subject, predicate, object = text.replace('_', ' ').split('-')
        query_graph = text_to_graph(subject, predicate, object, data_info)
        matching_score = matching(query_graph, scene_graphs)
        if local_rank == 0:
            print(f'\rCalculating: {i + 1}/{len(query_text)}', end = '')
            matching_score = dict([(key, val) for key, val in matching_score.items() if val > thr])
            categorize_result[text] = matching_score
    if local_rank == 0:
        return categorize_result

def main(query_text, scene_graphs, data_info, opts):
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    result = get_categorize_result(query_text, scene_graphs, data_info)
    synchronize()
    if local_rank == 0:
        # remove duplicate items
        __temp = {}
        duplicate = set()
        for label in result.keys():
            for item in result[label].keys():
                if item in __temp:
                    duplicate.add(item)
                __temp[item] = label
        for item in duplicate:
            __temp.pop(item)
        result = __temp
        if not os.path.exists(opts.out_image_dir):
            os.makedirs(opts.out_image_dir)
        if not os.path.exists(opts.out_info_dir):
            os.makedirs(opts.out_info_dir)
        print('Copy the image from src_image_dir to out_image_dir')
        for name, label in tqdm(result.items()):
            curr_dir = os.path.join(opts.out_image_dir, label)
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)
            os.system('cp {} {}'.format(os.path.join(opts.src_image_dir, name + '.jpg'), curr_dir))
        print('Save the result_info')
        torch.save(result, os.path.join(opts.out_info_dir, 'categorize_info_raw.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text_path', type = str, default = '/home/lzj/code/sgg/Dataset/query_text.json')
    parser.add_argument('--scene_graphs_path', type = str, default = '/home/lzj/code/sgg/outputs/filted_scene_graphs.pkl')
    parser.add_argument('--data_info_path', type = str, default = '/home/lzj/code/sgg/outputs/data_info.json')
    parser.add_argument('--src_image_dir', type = str, default = '/home/lzj/datasets/VisualGenome/VG_100K')
    parser.add_argument('--out_image_dir', type = str, default = '/home/lzj/datasets/VisualGenome/categorized_raw')
    parser.add_argument('--out_info_dir', type = str, default = '/home/lzj/code/sgg/outputs')
    opts = parser.parse_args()
    query_text = json.load(open(opts.query_text_path))
    data_info = json.load(open(opts.data_info_path))
    scene_graphs = torch.load(opts.scene_graphs_path)
    main(query_text, scene_graphs, data_info, opts)




