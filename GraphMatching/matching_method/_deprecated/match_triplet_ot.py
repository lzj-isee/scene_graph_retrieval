import torch
from ..structure import scene_graph
import ot

@torch.no_grad()
def match_triplet_ot(graph_s: scene_graph, graph_t: scene_graph,
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    rounds = 20, md_iter = 100, proj_iter = 20, eta = 100, add_prec = 1e-12, alpha = 0.1, mass_ratio = 1.0):
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
    cost_m  = torch.exp((torch.bitwise_not(triplet_s[:, None, :] == triplet_t[None, :, :]).sum(2) - 1) * 3)
    # for _ in range(rounds):
    #     temp = trans * torch.exp(-1 - eta * cost_m * alpha) + add_prec
    #     trans = peri_proj(trans = temp, mu_s = mu_s[:, None], mu_t = mu_t[:, None], total_mass = mass_ratio, n_iter = proj_iter)
    trans = ot.emd(mu_s, mu_t, cost_m)
    return torch.sum(trans * cost_m)