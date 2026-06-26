"""
attention_communities.py
------------------------------------------------------------------------
Turn a prefill self-attention map over LLaVA vision tokens into communities
using Ollivier-Ricci curvature (Ricci-flow community detection), and also
report Forman-Ricci curvature, on a k-NN-sparsified, symmetrized graph.

Pipeline:
    (N,N) attention  ->  symmetrize  ->  drop self-loops  ->  k-NN sparsify
                     ->  similarity-to-distance edge weights
                     ->  Ollivier-Ricci curvature + Ricci flow + surgery
                     ->  Forman-Ricci curvature (reported alongside)
                     ->  per-token community id

Dependencies:
    pip install numpy networkx GraphRicciCurvature POT

Main entry point:
    communities, info = attention_to_communities(attn_576x576)
where `communities` is a list of {"token": i, "community": c}.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #
# 1. attention tensor  ->  undirected, sparse, distance-weighted graph
# --------------------------------------------------------------------------- #
def _to_numpy(attn) -> np.ndarray:
    """Accept a torch.Tensor or anything array-like; return float64 ndarray."""
    try:
        import torch
        if isinstance(attn, torch.Tensor):
            return attn.detach().to(torch.float64).cpu().numpy()
    except ImportError:
        pass
    return np.asarray(attn, dtype=np.float64)


def build_graph_from_attention(
    attn,
    k: int = 10,
    symmetrize: str = "average",     # "average" | "max" | "min"
    sim_to_dist: str = "neglog",     # "neglog" | "inverse" | "one_minus"
    eps: float = 1e-8,
) -> nx.Graph:
    """
    Convert an (N, N) attention matrix into an undirected k-NN graph whose
    edge `weight` is a DISTANCE (small = strongly attended). GraphRicciCurvature
    interprets `weight` as distance, so similarities must be inverted first.
    """
    A = _to_numpy(attn)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], f"expected (N,N), got {A.shape}"
    n = A.shape[0]

    np.fill_diagonal(A, 0.0)  # no self-loops

    # --- symmetrize the (asymmetric) attention into an undirected similarity ---
    if symmetrize == "average":
        S = 0.5 * (A + A.T)
    elif symmetrize == "max":
        S = np.maximum(A, A.T)
    elif symmetrize == "min":
        S = np.minimum(A, A.T)
    else:
        raise ValueError(f"bad symmetrize={symmetrize!r}")

    # --- k-NN sparsification: per row keep the k strongest links, then make ---
    # --- the edge set undirected (keep an edge if EITHER endpoint chose it).  ---
    if k is not None and 0 < k < n - 1:
        keep = np.zeros((n, n), dtype=bool)
        idx = np.argpartition(-S, kth=k, axis=1)[:, :k]      # top-k per row
        keep[np.repeat(np.arange(n), k), idx.ravel()] = True
        keep = keep | keep.T
    else:
        keep = S > 0
    np.fill_diagonal(keep, False)

    # --- similarity -> distance on the surviving (upper-triangular) edges ---
    iu, ju = np.where(np.triu(keep, k=1))
    s = S[iu, ju]
    if s.size == 0:
        raise ValueError("no edges survived sparsification; increase k")

    smax = float(s.max())
    if sim_to_dist == "neglog":
        d = -np.log((s + eps) / (smax + eps))      # 0 for the strongest edge
    elif sim_to_dist == "inverse":
        d = 1.0 / (s + eps)
    elif sim_to_dist == "one_minus":
        d = 1.0 - s / (smax + eps)
    else:
        raise ValueError(f"bad sim_to_dist={sim_to_dist!r}")
    d = d - d.min() + eps                            # strictly positive distances

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, dist, sim in zip(iu.tolist(), ju.tolist(), d.tolist(), s.tolist()):
        G.add_edge(a, b, weight=float(dist), similarity=float(sim))
    return G


# --------------------------------------------------------------------------- #
# 2. curvature + Ricci-flow community detection
# --------------------------------------------------------------------------- #
def attention_to_communities(
    attn,
    k: int = 10,
    symmetrize: str = "average",
    sim_to_dist: str = "neglog",
    alpha: float = 0.5,              # ORC laziness: mass kept at the node
    flow_iterations: int = 20,
    ot_method: str = "Sinkhorn",    # "Sinkhorn" (fast) | "OTD" (exact LP) | "ATD"
    forman_method: str = "augmented",  # "augmented" (w/ triangles) | "1d" (simpler)
    cutoff_step: float = 0.025,
    drop_threshold: float = 0.01,
    return_graph: bool = False,
):
    """
    Returns
    -------
    communities : list[dict]   e.g. [{"token": 0, "community": 3}, ...]  (len N)
    info : dict with
        labels            : (N,) int array of community ids (-1 = isolated/unassigned)
        n_communities     : int
        ollivier_edges    : {(u,v): ollivier_ricci_curvature}
        forman_edges      : {(u,v): forman_ricci_curvature}
        ollivier_node_mean: (N,) mean incident ORC per token (redundancy signal)
        forman_node_mean  : (N,) mean incident FRC per token
        graph             : the networkx graph (only if return_graph=True)
    """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    from GraphRicciCurvature.FormanRicci import FormanRicci

    G = build_graph_from_attention(
        attn, k=k, symmetrize=symmetrize, sim_to_dist=sim_to_dist
    )
    n = G.number_of_nodes()

    labels = np.full(n, -1, dtype=int)
    orc_edges: dict[tuple[int, int], float] = {}
    frc_edges: dict[tuple[int, int], float] = {}
    next_cid = 0

    # Ricci flow needs finite shortest paths, so work per connected component.
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            (only,) = comp
            labels[only] = next_cid
            next_cid += 1
            continue

        sub = G.subgraph(comp).copy()

        # ---- Forman-Ricci curvature (cheap, no optimal transport) ----
        frc = FormanRicci(sub, weight="weight", method=forman_method, verbose="ERROR")
        frc.compute_ricci_curvature()
        for u, v, dd in frc.G.edges(data=True):
            frc_edges[(min(u, v), max(u, v))] = float(dd.get("formanCurvature", np.nan))

        # ---- Ollivier-Ricci curvature (optimal transport per edge) ----
        orc = OllivierRicci(sub, alpha=alpha, method=ot_method, weight="weight",
                            verbose="ERROR")
        orc.compute_ricci_curvature()
        for u, v, dd in orc.G.edges(data=True):
            orc_edges[(min(u, v), max(u, v))] = float(dd.get("ricciCurvature", np.nan))

        # ---- Ricci flow + surgery -> communities ----
        comm_map = _ricci_communities(
            orc, flow_iterations, cutoff_step, drop_threshold
        )
        local = {}
        for node, c in comm_map.items():
            if c not in local:
                local[c] = next_cid
                next_cid += 1
            labels[node] = local[c]

    # per-node curvature = mean over incident edges (token-level signal)
    orc_node = _node_mean(orc_edges, n)
    frc_node = _node_mean(frc_edges, n)

    communities = [{"token": int(i), "community": int(labels[i])} for i in range(n)]
    info = {
        "labels": labels,
        "n_communities": int(next_cid),
        "ollivier_edges": orc_edges,
        "forman_edges": frc_edges,
        "ollivier_node_mean": orc_node,
        "forman_node_mean": frc_node,
    }
    if return_graph:
        info["graph"] = G
    return communities, info


def _ricci_communities(orc, flow_iterations, cutoff_step, drop_threshold) -> dict:
    """Run Ricci flow then surgery; fall back to a curvature cut if surgery
    finds no clean cutoff (can happen on small / very dense components)."""
    orc.compute_ricci_flow(iterations=flow_iterations)
    try:
        _, clustering = orc.ricci_community(cutoff_step=cutoff_step,
                                            drop_threshold=drop_threshold)
        return clustering
    except (AssertionError, ValueError):
        # Fallback: cut edges whose (flowed) weight is an outlier, then take
        # connected components of what remains.
        Gf = orc.G.copy()
        w = np.array([d["weight"] for _, _, d in Gf.edges(data=True)])
        thr = w.mean() + w.std()
        Gf.remove_edges_from(
            [(u, v) for u, v, d in Gf.edges(data=True) if d["weight"] > thr]
        )
        mapping = {}
        for cid, cc in enumerate(nx.connected_components(Gf)):
            for node in cc:
                mapping[node] = cid
        return mapping


def _node_mean(edge_dict: dict, n: int) -> np.ndarray:
    s = np.zeros(n)
    c = np.zeros(n)
    for (u, v), val in edge_dict.items():
        if np.isnan(val):
            continue
        s[u] += val; c[u] += 1
        s[v] += val; c[v] += 1
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


if __name__ == "__main__":
    # quick self-test on a synthetic block-structured attention map
    rng = np.random.default_rng(0)
    N, n_blocks = 576, 6
    block = N // n_blocks
    base = rng.uniform(0, 0.05, size=(N, N))
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        base[sl, sl] += rng.uniform(0.5, 1.0, size=(block, block))
    base = base / base.sum(axis=1, keepdims=True)  # row-stochastic, like attention

    comms, info = attention_to_communities(base, k=12, flow_iterations=15)
    print(f"tokens={len(comms)}  communities found={info['n_communities']}")
    lab = info["labels"]
    # purity vs planted blocks
    planted = np.arange(N) // block
    from collections import Counter
    correct = 0
    for c in set(lab):
        members = np.where(lab == c)[0]
        if len(members):
            correct += Counter(planted[members]).most_common(1)[0][1]
    print(f"purity vs planted blocks = {correct / N:.3f}")
    print("sample:", comms[:3], "...", comms[-2:])