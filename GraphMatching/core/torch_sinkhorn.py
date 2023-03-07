import torch


def peyre_expon_explicite(Fs: torch.Tensor, Ft: torch.Tensor, trans):
    # method1
    # Ds = Fs.pow(2).sum(dim = -1)
    # Dt = Ft.pow(2).sum(dim = -1)
    # term1 = torch.matmul(Ds, trans.sum(1, keepdim = True)) # i * 1 array
    # term2 = torch.matmul(trans.sum(0, keepdim = True), Dt.t()) # 1 * i' array
    # term3 = ((Fs[:, :, None, :] * trans[None, :, :, None]).sum(1)[:, None, :, :] * Ft[None, :, :, :]).sum(dim = [2, 3])
    # result = term1 + term2 - 2 * term3
    # return result
    # method2
    dm = (Fs[:, :, None, None, :] - Ft[None, None, :, :, :]).pow(2).sum(-1)
    result = (dm * trans[None, :, None, :]).sum(dim = [1, 3])
    return result

def peyre_expon(Bs: torch.Tensor, Bt: torch.Tensor, trans: torch.Tensor):
    mu_s = torch.sum(trans, dim=1).unsqueeze_(1)
    mu_t = torch.sum(trans, dim=0).unsqueeze_(1)
    one_s = 0 * mu_s + 1
    one_t = 0 * mu_t + 1
    deg_terms = torch.matmul(torch.matmul(Bs ** 2, mu_s), torch.t(one_t))
    deg_terms += torch.matmul(one_s, torch.matmul(torch.t(mu_t), torch.t(Bt ** 2)))

    tmp = mem_matmul(Bs, trans)
    num = mem_matmul(tmp, Bt)
    del tmp
    return deg_terms - 2 * num

def peri_proj(trans, mu_s, mu_t, total_mass=0.9, n_iter=100, div_prec = 1e-16, thr = 1e-9, add_proj = False):
    dtype = trans.dtype
    device = trans.device
    p = mu_s.squeeze(1)
    q = mu_t.squeeze(1)
    one_s = torch.ones(p.size(), dtype=dtype, device=device)
    one_t = torch.ones(q.size(), dtype=dtype, device=device)
    for _ in range(n_iter):
        # torch.diagflat() builds a diagonal matrix
        P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), one_s)
        # P_p = torch.diagflat(P_p_d)
        # trans = torch.matmul(P_p, trans)
        trans *= P_p_d.unsqueeze_(1)

        P_q_d = cw_min(q / (div_prec + torch.sum(trans, dim=0)), one_t)
        # P_q = torch.diagflat(P_q_d)
        # trans = torch.matmul(trans, P_q)
        trans *= P_q_d

        # trans = trans / torch.sum(trans) * total_mass
        # print("trans={}".format(trans))
        trans /= torch.sum(trans)
        trans *= total_mass

    # --------- ending with this projection may be more appropriate ---------NOTE: LZJ, 12-27, keep the total mass equals total_mass
    if add_proj:
        P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), one_s)
        trans *= P_p_d.unsqueeze_(1)
    return trans

def slice_assign(big: torch.Tensor, small: torch.Tensor, indices_s: list, indices_t: list):
    # for i, index_s in enumerate(indices_s):
    #     for j, index_t in enumerate(indices_t):
    #         big[index_s, index_t] = small[i, j]
    big_flattened = big.view(-1)
    flattened_idx = [i * big.shape[1] + j for i in indices_s for j in indices_t]
    big_flattened[flattened_idx] = small.view(-1)
    return big


def mem_matmul(A: torch.Tensor, B: torch.Tensor):
    dtype = A.dtype
    device = A.device
    l, m1 = A.size()
    m2, n = B.size()
    assert m1 == m2
    res = torch.zeros((l, n), dtype=dtype, device=device)
    m = m1
    slice_size_m = 1000  # corresponds to 20000 * 20000 * 20000
    size_right_m = 20000 * 20000
    size_right = m * n
    slice_size = int(size_right_m * slice_size_m / size_right)
    # print("slice_size={}".format(slice_size))
    assert slice_size > 1, "the right matrix with size {} is too large".format((m, n))
    start = 0
    while start + slice_size < l:
        end = start + slice_size
        # print("start={}, end={}".format(start, end))
        res[start: end] = torch.matmul(A[start:end], B)
        start = end
    end = l
    res[start: end] = torch.matmul(A[start:end], B)
    return res

def cw_min(a, b):
    """
    component-wise minimum
    :param a: 1d FloatTensor
    :param b: 1d FloatTensor
    :return:
    """
    return torch.min(a, b)
