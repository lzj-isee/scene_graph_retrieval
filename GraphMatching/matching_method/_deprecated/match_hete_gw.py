import re
import torch, numpy as np
from ..structure import scene_graph, graph
from ..torch_sinkhorn import peyre_expon, peri_proj, slice_assign

def node_to_line(graph:scene_graph, directed = False):
    lg_nodes = graph.rel_pairs
    if directed:
        lg_w = (lg_nodes[:,1][:, None] == lg_nodes[:,0][None, :]).float() # direcited
    else:
        lg_w = torch.bitwise_or(lg_nodes[:,0][:, None] == lg_nodes[:,0][None, :], lg_nodes[:,1][:, None] == lg_nodes[:,1][None, :]).float()
        lg_w = lg_w - torch.diag(torch.diag(lg_w)) # undirected
    return lg_w

@torch.no_grad()
def match_hete_gw(
    graph_s: scene_graph, graph_t: scene_graph, 
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    iter = 100, proj_iter = 30, eta = 3, alpha = 0.3, add_prec = 1e-12, mass_ratio = 1.0, thr = 1e-4, **kw
    ):
    device = graph_s.device
    # transform the node_graph to line_graph
    lg_s = graph(node_to_line(graph_s, directed = False), graph_s.rel_labels) # the num of edge must not be 0
    lg_t = graph(node_to_line(graph_t, directed = False), graph_t.rel_labels)
    # get the transition according to line graph 
    _, trans = bcd_some_type(lg_s, lg_t, mask = None, add_proj = True, mass_thr = False, thr = 1e-9, directed = False)
    threshold = trans.max() / 3
    trans[trans < threshold] = False
    trans[trans > threshold] = True
    # get the mask
    ns, nt = graph_s.node_num, graph_t.node_num
    mask = torch.ones(ns, nt, device = device)
    index0, index1 = torch.where(trans > 0)
    mask[torch.unique(graph_s.rel_pairs[index0])] = 0
    pair_s, pair_t = graph_s.rel_pairs[index0], graph_t.rel_pairs[index1]
    mask[pair_s[:, 0], pair_t[:, 0]] = 1
    mask[pair_s[:, 0], pair_t[:, 1]] = 1
    mask[pair_s[:, 1], pair_t[:, 0]] = 1
    mask[pair_s[:, 1], pair_t[:, 1]] = 1
    rel_label_s = graph_s.get_rel_label_matrix()
    rel_label_t = graph_t.get_rel_label_matrix()
    w_s = torch.bitwise_or(rel_label_s > 0, rel_label_s.t() > 0).float()
    w_t = torch.bitwise_or(rel_label_t > 0, rel_label_t.t() > 0).float()
    ng_s = graph(w_s, graph_s.box_labels)
    ng_t = graph(w_t, graph_t.box_labels)
    _, trans = bcd_some_type(ng_s, ng_t, mask = mask, add_proj = False, mass_thr = True, thr = 1e-4, directed = False)
    cost = 1 / (trans.sum() + 1e-3)
    return cost

@torch.no_grad()
def bcd_some_type(
    graph_s: graph, graph_t: graph, 
    walk_len = 2, mask = None, add_proj = False, mass_thr = False,
    rounds = 10, md_iter = 20, proj_iter = 20, eta = 10, add_prec = 1e-12, mass_ratio = 1.0, thr = 1e-9, directed = False, **kw
    ): 
    device = graph_s.w.device
    ns, nt = graph_s.w.shape[0], graph_t.w.shape[0]
    if mask is None: mask = torch.ones(ns, nt, device = device)
    label_s, label_t = graph_s.node_label, graph_t.node_label
    mu_s, mu_t = torch.ones(ns, device = device) / ns, torch.ones(nt, device = device) / nt
    # calculate intra similarity (used in GW)
    ds, dt = graph_s.w.sum(1), graph_t.w.sum(1)
    ds[ds == 0], dt[dt == 0] = 0.1, 0.1
    rw_s, rw_t = graph_s.w / ds[:, None], graph_t.w / dt[:, None]
    ds[ds == 0.1], dt[dt == 0.1] = 0.0, 0.0
    Bs, Bt = rw_s.clone(), rw_t.clone()
    for _ in range(2, walk_len + 1):
        Bs += rw_s @ rw_s
        Bt += rw_t @ rw_t
    if not directed:
        Bs = Bs + Bs.t()
        Bt = Bt + Bt.t()
    Bs = Bs - torch.diag(torch.diag(Bs))
    Bt = Bt - torch.diag(torch.diag(Bt))
    # sort the label according to node number
    types = torch.unique(label_s)
    sort_index = torch.bincount(label_s)[types].sort(descending = True).indices
    prob_list = types[sort_index]
    # get the transition
    trans = torch.zeros(ns, nt, device = device)
    for _ in range(rounds):
        trans_old_r = trans.clone()
        # cost_old = torch.sum(peyre_expon(Bs, Bt, trans) * trans)
        for type_i in range(len(prob_list)):
            indices_s = torch.where(label_s == prob_list[type_i])[0].tolist()
            indices_t = torch.where(label_t == prob_list[type_i])[0].tolist()
            if len(indices_t) == 0: continue # target nodes do not has the type
            tmp = peyre_expon(Bs, Bt, trans)
            mass_type_i = mass_ratio * torch.minimum(mu_s[indices_s].sum(), mu_t[indices_t].sum())
            const_g = tmp[indices_s][:, indices_t] * 2
            del tmp
            if torch.sum(torch.abs(trans[indices_s][:, indices_t])) == 0:
                trans_type = torch.matmul(mu_s[indices_s][:, None], torch.t(mu_t[indices_t][:, None]))
            else:
                trans_type = trans[indices_s][:, indices_t].clone()
            for iter_num in range(md_iter):
                if mask[indices_s][:, indices_t].sum() == 0: continue
                g = peyre_expon(Bs[indices_s][:, indices_s], Bt[indices_t][:, indices_t], trans_type) + const_g
                tmp = trans_type * torch.exp( -1 - eta * g) + add_prec
                trans_type = peri_proj(
                    trans = tmp * mask[indices_s][:, indices_t], mu_s = mu_s[indices_s][:, None], mu_t = mu_t[indices_t][:, None],
                    total_mass = mass_type_i, n_iter = proj_iter, add_proj = add_proj
                )
                if iter_num == md_iter - 1:
                    slice_assign(big = trans, small = trans_type, indices_s = indices_s, indices_t = indices_t)
        # cost = torch.sum(peyre_expon(Bs, Bt, trans) * trans)
        if not mass_thr:
            if torch.linalg.norm(trans - trans_old_r) < thr:
                return None, trans
        else:
            if (trans_old_r.sum() - trans.sum()).abs() < thr:
                return None, trans
    return None, trans