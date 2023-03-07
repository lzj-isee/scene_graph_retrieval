import torch
from ..structure import scene_graph

@torch.no_grad()
def match_obj(graph_s: scene_graph, graph_t: scene_graph,
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    ):
    cost_obj = torch.bitwise_not(graph_s.box_labels[:, None] == graph_t.box_labels[None, :])
    cost_score = (cost_obj.min(1)[0].mean() + cost_obj.min(0)[0].mean()) / 2
    return cost_score