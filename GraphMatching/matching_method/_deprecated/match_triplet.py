import torch
from ..structure import scene_graph

@torch.no_grad()
def match_triplet(graph_s: scene_graph, graph_t: scene_graph, 
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor):
    device = graph_s.device
    ns, nt = graph_s.rel_pairs.shape[0], graph_t.rel_pairs.shape[0]
    triplet_s = torch.cat(
        [
            graph_s.box_labels[graph_s.rel_pairs[:, 0]].view(-1, 1), 
            graph_s.box_labels[graph_s.rel_pairs[:, 1]].view(-1, 1), 
            graph_s.rel_labels.view(-1, 1)
        ], dim = 1
    )
    triplet_t = torch.cat(
        [
            graph_t.box_labels[graph_t.rel_pairs[:, 0]].view(-1, 1), 
            graph_t.box_labels[graph_t.rel_pairs[:, 1]].view(-1, 1), 
            graph_t.rel_labels.view(-1, 1)
        ], dim = 1
    )
    mu_s = torch.ones(ns, device = device) / ns
    mu_t = torch.ones(nt, device = device) / nt
    # trans = torch.matmul(mu_s[:, None], mu_t[None, :])
    cost_m  = torch.bitwise_not(triplet_s[:, None, :] == triplet_t[None, :, :]).sum(2).float()
    cost_score = (cost_m.min(1)[0].mean() * 0.9 + cost_m.min(0)[0].mean() * 0.1)
    return cost_score