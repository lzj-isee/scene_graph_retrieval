import torch
from ..structure import scene_graph
from ..torch_sinkhorn import peyre_expon_explicite, peri_proj

@torch.no_grad()
def match_gw(graph_s: scene_graph, graph_t: scene_graph, 
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    iter = 100, proj_iter = 30, eta = 3, alpha = 0.3, add_prec = 1e-12, mass_ratio = 1.0, thr = 1e-4, **kw
    ):
    # determine the cost matrix with both the node label and the edge label
    device = graph_s.device
    ns, nt = graph_s.node_num, graph_t.node_num
    cost_node = torch.bitwise_not(graph_s.box_labels[:, None] == graph_t.box_labels[None, :]).float()
    Fs = torch.zeros((ns, ns, len(idx_to_predicates)), device = device)
    Ft = torch.zeros((nt, nt, len(idx_to_predicates)), device = device)
    Fs[graph_s.rel_pairs[:,0], graph_s.rel_pairs[:,1], graph_s.rel_labels] = 1
    Ft[graph_t.rel_pairs[:,0], graph_t.rel_pairs[:,1], graph_t.rel_labels] = 1
    mu_s = torch.ones(ns, device = device) / ns # we use uniform distribution here,
    mu_t = torch.ones(nt, device = device) / nt
    # get the GW cost
    trans = torch.ones(ns, nt, device = device) / ns / nt
    for _ in range(iter):
        gw_cost_old = torch.sum((peyre_expon_explicite(Fs, Ft, trans) + alpha * cost_node) * trans)
        old_t = trans.sum(0)
        g = peyre_expon_explicite(Fs, Ft, trans)
        g = g + alpha * cost_node
        temp = trans * torch.exp(-1 - eta * g) + add_prec
        trans = peri_proj(
            trans = temp, mu_s = mu_s[:, None], mu_t = mu_t[:, None], 
            total_mass = mass_ratio, n_iter = proj_iter
        )
        gw_cost_curr = torch.sum((peyre_expon_explicite(Fs, Ft, trans) + alpha * cost_node) * trans)
        if (gw_cost_curr - gw_cost_old).abs() / gw_cost_curr < thr:
            return gw_cost_curr
    return gw_cost_curr