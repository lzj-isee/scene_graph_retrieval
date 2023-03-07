from re import match
from threading import local
import torch, os
from torch._C import import_ir_module
from tqdm import tqdm
from PIL import ImageOps, Image
import json
from graph_matching.structure import scene_graph
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors
import graph_matching.matching_method as matching_method
from graph_matching.common import synchronize, all_gather
import argparse
import numpy as np

def main(opts):
    # whether use multi gpu
    local_rank = int(os.environ['LOCAL_RANK'])
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend = "nccl", init_method = "env://", 
            rank = local_rank, world_size = num_gpus,
        )
        synchronize()
    # loading the search list 
    data_info = json.load(open(os.path.join(opts.result_folder, 'custom_data_info.json')))
    scene_graph_files = os.listdir(opts.scene_graph_folder)
    scene_graph_files.sort()
    # loading the word dict for embedding
    ind_to_classes = data_info['ind_to_classes']
    ind_to_predicates = data_info['ind_to_predicates']
    obj_embedding = obj_edge_vectors(names = data_info['ind_to_classes'], wv_dir = '/home/lzj/datasets/glove', wv_dim = 200).to(opts.device)
    rel_embedding = obj_edge_vectors(names = data_info['ind_to_predicates'], wv_dir = '/home/lzj/datasets/glove', wv_dim = 200).to(opts.device)
    # generate the source scene graph
    pred_s = {}
    pred_s['box_labels'] = [ind_to_classes.index(opts.subject), ind_to_classes.index(opts.object)]
    pred_s['box_scores'] = [1, 1]
    pred_s['rel_pairs'] = [[0, 1]]
    pred_s['rel_scores'] = [1]
    pred_s['rel_labels'] = [ind_to_predicates.index(opts.predicate)]
    # get the source scene graph
    graph_s = scene_graph(pred_s, device = opts.device)
    # ranking by calculating the cost between graph_source and each graph_target
    files_this_gpu = scene_graph_files[local_rank::num_gpus]
    ranking_cost = {}
    ranking_cost_this_gpu = {}
    for file in tqdm(files_this_gpu):
        pred_t = torch.load(os.path.join(opts.scene_graph_folder, file))
        graph_t = scene_graph(pred_t, device = opts.device)
        ranking_cost_this_gpu[file] = matching_method.match_triplet_hete_gw(
            graph_s, graph_t, data_info['ind_to_classes'], data_info['ind_to_predicates'], obj_embedding, rel_embedding
        ).item()
    synchronize()
    # gather the results and post process at local_rank 0
    multi_gpu_ranking_cost = all_gather(ranking_cost_this_gpu, num_gpus = num_gpus, to_device = opts.device)
    synchronize()
    if local_rank == 0:
        for p in multi_gpu_ranking_cost:
            ranking_cost.update(p)
        ranking_cost = dict(sorted(ranking_cost.items(), key = lambda x : x[1], reverse = False))
        assert len(ranking_cost) == len(scene_graph_files), 'images might be missing'
        # save the full ranking result
        with open(os.path.join(opts.result_folder, 'ranking_list.json'), 'w') as outfile:  
            json.dump(ranking_cost, outfile, indent = 1)
        ranking_cost = list(ranking_cost.items()) # the first element is the name of *.pkl, the second is the value of cost
        # get top images save paste as a single one, and save the triplet of scene graphs
        img_size = 224
        plt_topk = 16
        img_list = []
        graph_list = []
        target = Image.new('RGB', (img_size * 4, img_size * 4))
        for i in range(plt_topk):
            retrieval_image = ImageOps.exif_transpose(Image.open(os.path.join(opts.image_folder, os.path.splitext(ranking_cost[i][0])[0] + '.jpg')))
            retrieval_graph = torch.load(os.path.join(opts.scene_graph_folder, os.path.splitext(ranking_cost[i][0])[0] + '.pkl'))
            img_list.append(retrieval_image.resize(size = (img_size, img_size)))
            graph_list.append(retrieval_graph)
        for i in range(4):
            for j in range(4):
                target.paste(img_list[i * 4 + j], box = (img_size * i, img_size * j))
        target.save(os.path.join(opts.result_folder, 'ranking.jpg'), quality = 100)
        # save the scene graph
        triplet_list = []
        for graph in graph_list:
            rel_pairs = graph['rel_pairs'] if isinstance(graph['rel_pairs'], list) else graph['rel_pairs'].tolist()
            rel_labels = graph['rel_labels'] if isinstance(graph['rel_labels'], list) else graph['rel_labels'].tolist()
            box_labels = graph['box_labels'] if isinstance(graph['box_labels'], list) else graph['box_labels'].tolist()
            triplet_list.append([str(rel_pairs[i][0]) + '_' + ind_to_classes[box_labels[rel_pairs[i][0]]] + ' ==> ' + ind_to_predicates[rel_labels[i]] + ' ==> ' + str(rel_pairs[i][1]) + '_' + ind_to_classes[box_labels[rel_pairs[i][1]]] for i in range(len(rel_pairs))])
        with open(os.path.join(opts.result_folder, 'graphs.json'), 'w') as outfile:
            json.dump(triplet_list, outfile, indent = 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type = str, default = '__background__')
    parser.add_argument('--predicate', type = str, default = '__background__')
    parser.add_argument('--object', type = str, default = '__background__')
    parser.add_argument('--image_folder', type = str, default = '/home/lzj/datasets/subVisualGenome')
    parser.add_argument('--scene_graph_folder', type = str, default = './outputs/subVisualGenome/out_scene_graphs')
    parser.add_argument('--result_folder', type = str, default = './outputs/subVisualGenome')
    parser.add_argument('--device', type = str, default = 'cuda') # I have not test those codes on cpu, therefore, only gpu version if available
    opts = parser.parse_args()
    main(opts)


