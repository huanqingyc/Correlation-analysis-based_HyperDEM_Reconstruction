#!/usr/bin/env python3
"""Core functions for multi-body correlation analysis."""

import itertools
import math
from typing import FrozenSet, Tuple

import numpy as np
import torch
import stim
import networkx as nx

HyperEdge = FrozenSet[int]


def targets_to_dets(targets):
    """Parse stim instruction targets into a frozenset of detector IDs."""
    dets_track = []
    dets_sep_track = []
    for t in targets:
        if t.is_relative_detector_id():
            dets_sep_track.append(t.val)
        elif t.is_separator():
            dets_track.append(dets_sep_track)
            dets_sep_track = []
    dets_track.append(dets_sep_track)
    return frozenset(i for d in dets_track for i in d)


def pair_correlations_inference(detection_events, device, eps=1e-4):
    """Compute pair correlations, return p_ij matrix and 2nd-order hyperedge set."""
    num_shots, num_dets = detection_events.shape
    sigma_events = 1 - 2 * detection_events  # Convert to (1, -1) format
    sigma_tensor = torch.from_numpy(sigma_events.astype(np.float32)).to(device)
    sum_vec = sigma_tensor.sum(dim=0)
    corr = sigma_tensor.T @ sigma_tensor
    f_ij = (torch.outer(sum_vec, sum_vec) / corr).to(torch.float32) / num_shots
    p_ij = (1 - torch.pow(f_ij.cpu(), 0.5)) * 0.5
    p_ij[p_ij < eps] = 0

    i_idx, j_idx = torch.triu_indices(num_dets, num_dets, offset=1)
    mask = p_ij[i_idx, j_idx] >= eps
    hyperedge_2d = {frozenset((int(i_idx[m]), int(j_idx[m]))) for m in torch.where(mask)[0]}
    return p_ij, hyperedge_2d


def search_potential_hyperedges(p_ij, max_order):
    """
    Search for potential hyperedges: treat p_ij as adjacency matrix, find cliques
    of size <= max_order, add all k-order subsets (3<=k<=max_order) to hyperedges[k-1].
    """
    adj = p_ij.detach().cpu().numpy()
    G = nx.from_numpy_array(adj)
    hyperedges = [set() for _ in range(max_order)]

    for clique in nx.enumerate_all_cliques(G):
        if len(clique) > max_order:
            break
        for k in range(3, len(clique) + 1):
            for comb in itertools.combinations(clique, k):
                hyperedges[k - 1].add(frozenset(comb))

    return hyperedges

def cal_p( f_list: list, hyperedge_list: list, mode: str = "given_dem", eps: Tuple[float, float] = (1e-4, 1e-5), correct_in_step: bool = True,
) -> Tuple[list, list]:
    max_order = min(len(f_list), len(hyperedge_list))
    w_dicts = [{} for _ in range(max_order)]
    p_dicts = [{} for _ in range(max_order)]

    for k in range(max_order - 1, -1, -1):
        dim = k + 1
        exp = 0.5 ** (dim - 1)
        eps_th = eps[0] if dim == 1 else eps[1] if mode == "pruning" else 0.0

        for h in hyperedge_list[k]:
            prod_w = 1.0
            for j in range(k + 1, max_order):
                for h_larger in hyperedge_list[j]:
                    if h.issubset(h_larger):
                        prod_w *= w_dicts[j][h_larger]
            w_dicts[k][h] = (f_list[k][h] ** exp) / prod_w
            p_dicts[k][h] = (1 - w_dicts[k][h]) * 0.5

            if mode == "given_dem":
                if correct_in_step and w_dicts[k][h] > 1:
                    w_dicts[k][h] = 1.0
                    p_dicts[k][h] = 0
            elif mode == "pruning":
                if p_dicts[k][h] <= eps_th:
                    del p_dicts[k][h]

        if mode == "pruning":
            hyperedge_list[k] = list(p_dicts[k].keys())

    if mode == "given_dem" and not correct_in_step:
        for k in range(max_order):
            for h in list(p_dicts[k].keys()):
                if p_dicts[k][h] < 0:
                    p_dicts[k][h] = 0

    return p_dicts, hyperedge_list


def cal_multi_body_correlations(
    detection_events,
    mode: str = "given_dem",
    *,
    hyperedge_list=None,
    max_order: int = 4,
    device: str = "cpu",
    eps: Tuple[float, float] = (1e-4, 1e-5),
    correct_in_step: bool = True,
) -> Tuple[list, list]:
    """
    Compute multi-body correlations. Two modes controlled by mode.

    Args:
        detection_events: Detection event array, shape (shots, num_detectors)
        mode: 'inference' or 'given_dem'
            - inference: No DEM prior, infer hyperedge structure from data and prune
            - given_dem: Use given hyperedge structure, no pruning
        hyperedge_list: Per-order hyperedge list, required only in given_dem mode
        max_order: Max order, effective only in inference mode, default 4
        device: Compute device, 'cpu' or 'cuda:X'
        eps: (eps_1d, eps_2d+) Pruning thresholds, effective only in inference mode
        correct_in_step: Whether to correct w>1 at each step, given_dem mode only

    Returns:
        (p_list, hyperedge_list): Per-order correlations and hyperedge list
    """
    if mode == "inference":
        _, num_dets = detection_events.shape
        p_ij, hyperedge_2d = pair_correlations_inference(detection_events, device, eps=eps[1])
        hyperedge_list = [
            [frozenset([i]) for i in range(num_dets)],
            hyperedge_2d,
        ]
        if max_order > 2:
            high_order = search_potential_hyperedges(p_ij, max_order)
            hyperedge_list.extend(high_order[2:max_order])
    else:
        if hyperedge_list is None:
            raise ValueError("given_dem mode requires hyperedge_list")

    f_list = cal_m_f_given_dem(detection_events, hyperedge_list, device)
    p_list, hyperedge_list = cal_p(
        f_list, hyperedge_list,
        mode="pruning" if mode == "inference" else "given_dem",
        eps=eps,
        correct_in_step=correct_in_step,
    )
    return p_list, hyperedge_list


def cal_m_f_given_dem(detection_events: np.ndarray, hyperedges: list, device: str = "cuda:0") -> list:
    num_shots, _ = detection_events.shape
    sigma_tensor = torch.from_numpy((1 - 2 * detection_events).astype(np.float32)).to(device)
    max_order = len(hyperedges)
    m_list = [{} for _ in range(max_order)]

    # Batch compute m by subset size to reduce Python loops and GPU calls
    for s in range(1, max_order + 1):
        needed = set()
        for order in range(s, max_order + 1):
            for h in hyperedges[order - 1]:
                for subset in itertools.combinations(h, s):
                    needed.add(frozenset(subset))
        if not needed:
            continue
        subsets = list(needed)
        indices = torch.tensor(
            [sorted(sub) for sub in subsets],
            device=sigma_tensor.device,
            dtype=torch.long,
        )
        prods = sigma_tensor[:, indices].prod(dim=2)
        m_vals = prods.sum(dim=0) / num_shots
        for h_sub, val in zip(subsets, m_vals.cpu().tolist()):
            m_list[s - 1][h_sub] = val

    f_list = [m_list[0].copy()]
    for order in range(2, max_order + 1):
        current_f = {}
        for h in hyperedges[order - 1]:
            num, den = 1.0, 1.0
            for k in range(1, order + 1):
                prod_k = math.prod(m_list[k - 1][frozenset(sub)] for sub in itertools.combinations(h, k))
                if k % 2 == 1:
                    num *= prod_k
                else:
                    den *= prod_k
            current_f[h] = num / (den or 1e-12)
        f_list.append(current_f)

    return f_list


def generate_test_circuit(distance=4, rounds=2, shots=500000, 
                         code_task='surface_code:rotated_memory_x',
                         after_clifford_depolarization=0.001,
                         before_round_data_depolarization=0.001,
                         before_measure_flip_probability=0.001,
                         after_reset_flip_probability=0.001):
    """
    Generate test circuit and detection events.

    Args:
        distance: Code distance
        rounds: Number of rounds
        shots: Number of samples
        code_task: Code type, e.g.:
            - 'surface_code:rotated_memory_x'
            - 'surface_code:rotated_memory_z'
            - 'repetition_code:memory'
            - 'color_code:memory'
            Default: 'surface_code:rotated_memory_x'
        Other args: Noise parameters

    Returns:
        circuit: stim circuit object
        dem: Detector error model
        dets: Detection event array
        num_dets: Number of detectors
    """
    circuit = stim.Circuit.generated(
        code_task=code_task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=after_clifford_depolarization,
        before_round_data_depolarization=before_round_data_depolarization,
        before_measure_flip_probability=before_measure_flip_probability,
        after_reset_flip_probability=after_reset_flip_probability,
    )
    
    # decompose_errors=True decomposes hyperedge errors into at most 2-detector edges, required by BeliefMatching etc.
    dem = circuit.detector_error_model()
    num_dets = dem.num_detectors
    dets = circuit.compile_detector_sampler().sample(shots=shots)
    
    return circuit, dem, dets, num_dets
