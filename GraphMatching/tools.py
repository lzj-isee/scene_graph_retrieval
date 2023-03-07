from .matching_method.match_triplet_hete_gw import match_triplet_hete_gw
from .core.structure import scene_graph
from .core.common import synchronize, all_gather
import os, torch
from tqdm import tqdm

def matching(query_graph, gallery_graphs):
    if not isinstance(query_graph, dict):
        raise RuntimeError('The query_graph should be type of dict containing raw information')
    if not isinstance(gallery_graphs, dict):
        raise RuntimeError('The gallery_graphs should be type of str')
    # check whether using multi-process
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"])
    # torch.distributed.init_process_group(
    #     backend = "gloo", init_method = "env://", 
    #     rank = local_rank, world_size = world_size,
    # )
    synchronize()
    # allocate gallery
    gallery_ids = sorted(list(gallery_graphs.keys()))
    ids_on_this_gpu = gallery_ids[local_rank::world_size]
    # calculate the similarity score
    ranking_this_gpu = {}
    graph_s = scene_graph(query_graph)
    for target_id in ids_on_this_gpu:
        graph_t = scene_graph(gallery_graphs[target_id], device = 'cpu')#"cuda:{}".format(local_rank % num_gpus))
        ranking_this_gpu[target_id] = match_triplet_hete_gw(
            graph_s, graph_t, None, None, None, None
        ).item()
    synchronize()
    # gather the results
    multi_gpu_ranking = all_gather(ranking_this_gpu, world_size = world_size, to_device = "cpu")
    synchronize()
    # ranking the results 
    if local_rank == 0:
        ranking_result = {}
        for p in multi_gpu_ranking:
            ranking_result.update(p)
        ranking_result = dict(sorted(ranking_result.items(), key = lambda x: x[1], reverse = True))
        assert len(ranking_result) == len(gallery_ids), 'images might be missing during gathering'
        return ranking_result

def text_to_graph(subject, predicate, object, data_info):
    ind_to_classes = data_info['ind_to_classes']
    ind_to_predicates = data_info['ind_to_predicates']
    result = {}
    result['area'] = torch.Tensor([1])
    result['box'] = torch.as_tensor([0, 0, 0, 0])
    result['box_labels'] = torch.as_tensor([ind_to_classes.index(subject), ind_to_classes.index(object)])
    result['rel_pairs'] = torch.as_tensor([[0, 1]])
    result['rel_labels'] = torch.as_tensor([ind_to_predicates.index(predicate)])
    return result
