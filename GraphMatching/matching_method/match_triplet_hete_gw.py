import torch, numpy as np
from ..core.structure import scene_graph
from ..core.torch_sinkhorn import peyre_expon, peri_proj, slice_assign
import collections

def node_to_triplet(graph: scene_graph):
    so_nodes = graph.rel_pairs
    w = torch.bitwise_or(so_nodes[:,0][:, None] == so_nodes[:,0][None, :], so_nodes[:,1][:, None] == so_nodes[:,1][None, :]).float()
    w = w - torch.diag(torch.diag(w))
    label = torch.cat([graph.box_labels[so_nodes[:,0]][:,None], graph.box_labels[so_nodes[:,1]][:,None], graph.rel_labels[:,None]], dim = 1)
    label = label.cpu().numpy().astype(np.str)
    str_label = np.char.add(label[:, 0], '_')
    str_label = np.char.add(str_label, label[:, 1])
    str_label = np.char.add(str_label, '_')
    str_label = np.char.add(str_label, label[:, 2])
    return w, str_label

def match_triplet_hete_gw(graph_s: scene_graph, graph_t: scene_graph,
    idx_to_class: list, idx_to_predicates: list, obj_embedding: torch.Tensor, rel_embedding: torch.Tensor, 
    walk_len = 5, rounds = 10, md_iter = 80, proj_iter = 20, eta = 1, add_prec = 1e-12, mass_ratio = 1.0, thr = 1e-4
    ):
    device = graph_s.device
    w_s, label_s = node_to_triplet(graph_s)
    w_t, label_t = node_to_triplet(graph_t)
    ns, nt = w_s.shape[0], w_t.shape[0]
    # mu_s, mu_t = torch.ones(ns, device = device) / ns, torch.ones(nt, device = device) / nt # we use uniform distribution here
    mu_s, mu_t = graph_s.pair_weights, graph_t.pair_weights
    # calculate intra similarity (used in GW)
    ds, dt = w_s.sum(1), w_t.sum(1)
    ds[ds == 0], dt[dt == 0] = 0.1, 0.1
    rw_s, rw_t = w_s / ds[:, None], w_t / dt[:, None]
    ds[ds == 0.1], dt[dt == 0.1] = 0.0, 0.0
    Bs, Bt = rw_s.clone(), rw_t.clone()
    for _ in range(2, walk_len + 1):
        Bs += rw_s @ rw_s
        Bt += rw_t @ rw_t
    Bs = Bs + Bs.t()
    Bt = Bt + Bt.t()
    Bs = Bs - torch.diag(torch.diag(Bs))
    Bt = Bt - torch.diag(torch.diag(Bt))
    # matching
    prob_list = collections.Counter(label_s).most_common()
    trans = torch.zeros(ns, nt, device = device)
    for _ in range(rounds):
        simi_score_old = torch.sum(peyre_simi(Bs, Bt, trans) * trans)
        # cost_old = torch.sum(peyre_expon(Bs, Bt, trans) * trans)
        for type_i,_ in prob_list:
            indices_s = np.where(label_s == type_i)[0].tolist()
            indices_t = np.where(label_t == type_i)[0].tolist()
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
                g = peyre_expon(Bs[indices_s][:, indices_s], Bt[indices_t][:, indices_t], trans_type) + const_g
                tmp = trans_type * torch.exp( -1 - eta * g) + add_prec
                trans_type = peri_proj(
                    trans = tmp, mu_s = mu_s[indices_s][:, None], mu_t = mu_t[indices_t][:, None],
                    total_mass = mass_type_i, n_iter = proj_iter, add_proj = False
                )
                if iter_num == md_iter - 1:
                    slice_assign(big = trans, small = trans_type, indices_s = indices_s, indices_t = indices_t)
        simi_score_curr = torch.sum(peyre_simi(Bs, Bt, trans) * trans)
        if (simi_score_old - simi_score_curr).abs() < thr:
            # return 1 / (simi_score_curr + 1e-3)
            return simi_score_curr
    # return 1 / (simi_score_curr + 1e-3)
    return simi_score_curr # 返回当前的similarity得分

def peyre_simi(Bs: torch.Tensor, Bt: torch.Tensor, trans):
    # dm = (Bs[:, :, None, None] - Bt[None, None, :, :]).abs().negative().exp()
    dm = 1 / ((Bs[:, :, None, None] - Bt[None, None, :, :]).abs() + 0.5) + 1 # cost 转为 similarity得分
    result = (dm * trans[None, :, None, :]).sum(dim = [1, 3])
    return result


