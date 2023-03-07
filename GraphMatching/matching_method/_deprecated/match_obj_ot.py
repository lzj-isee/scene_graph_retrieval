import torch
from ..structure import scene_graph
import ot

@torch.no_grad()
def match_obj_ot(graph_s: scene_graph, graph_t: scene_graph, 
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    rounds = 20, md_iter = 100, proj_iter = 20, eta = 10, add_prec = 1e-12, alpha = 0.1, mass_ratio = 1.0):
    device = graph_s.device
    # evaluate the distance between two graphs using the node information only (vanilla OT)
    ns, nt = graph_s.node_num, graph_t.node_num
    # set the marginal distribution according to the degree of each node / uniform marginal
    # edge_s = graph_s.get_rel_label_matrix() > 0
    # edge_s = torch.bitwise_or(edge_s > 0, edge_s.t() > 0).int()
    # edge_t = graph_t.get_rel_label_matrix() > 0
    # edge_t = torch.bitwise_or(edge_t > 0, edge_t.t() > 0).int()
    # mu_s = edge_s.sum(1) / edge_s.sum()
    # mu_t = edge_t.sum(1) / edge_t.sum()
    mu_s = torch.ones(ns, device = device) / ns
    mu_t = torch.ones(nt, device = device) / nt
    # get the transportation matrix
    # trans = torch.matmul(mu_s[:, None], mu_t[None, :])
    cost_m = torch.bitwise_not(graph_s.box_labels[:, None] == graph_t.box_labels[None, :]).float()
    # for _ in range(rounds):
    #     temp = trans * torch.exp(-1 - eta * cost_m * alpha) + add_prec
    #     trans = peri_proj(trans = temp, mu_s = mu_s[:, None], mu_t = mu_t[:, None], total_mass = mass_ratio, n_iter = proj_iter)
    trans = ot.emd(mu_s, mu_t, cost_m)
    return torch.sum(trans * cost_m)