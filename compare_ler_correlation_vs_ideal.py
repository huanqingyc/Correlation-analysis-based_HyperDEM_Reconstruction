#!/usr/bin/env python3
"""Compare LER by matching: correlation analysis DEM vs ideal DEM from stim-generated circuit.

仅保留 run_decode_from_files（被 run_ler_comparison.py 调用）所需的函数。
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import stim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import multiprocessing as mp

from function import targets_to_dets


def create_dem_from_analysis(reference_dem, hyperedge_probs):
    """用关联分析得到的超边概率替换参考 DEM 中的概率，得到新的 DEM。

    Args:
        reference_dem: 参考 DEM（结构不变）
        hyperedge_probs: 超边 (frozenset) -> 概率

    Returns:
        新 DEM
    """
    from run_ler_comparison import extract_hyperedge_from_dem
    reference_probs, _ = extract_hyperedge_from_dem(reference_dem)

    new_dem = stim.DetectorErrorModel()
    for instruction in reference_dem.flattened():
        if instruction.type == "error":
            dets = targets_to_dets(instruction.targets_copy())
            prob = hyperedge_probs.get(
                dets, reference_probs.get(dets, instruction.args_copy()[0])
            )
            new_dem.append(stim.DemInstruction(
                "error", args=[prob], targets=instruction.targets_copy(),
            ))
        else:
            new_dem.append(instruction)
    return new_dem


def _decode_in_chunks(dem, dets, obvs, max_cores, chunk_decode_fn):
    """通用分块解码：构建 chunks，并行或串行调用 chunk_decode_fn，返回 (ler, predicted_obs)。"""
    n_shots = dets.shape[0]
    max_cores = max_cores or mp.cpu_count()
    chunk_size = max(1, n_shots // max_cores)
    n_chunks = (n_shots + chunk_size - 1) // chunk_size

    obvs_2d = obvs.reshape(-1, 1) if obvs.ndim == 1 else obvs
    chunks = [
        (i, i * chunk_size, min((i + 1) * chunk_size, n_shots),
         dets[i * chunk_size:min((i + 1) * chunk_size, n_shots)],
         obvs_2d[i * chunk_size:min((i + 1) * chunk_size, n_shots)],
         dem)
        for i in range(n_chunks)
    ]

    n_workers = min(n_chunks, max_cores)
    if n_workers > 1 and n_chunks > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(chunk_decode_fn, chunks))
        results.sort(key=lambda x: x[0])
        predicted_obs = np.vstack([r[3] for r in results])
    else:
        predicted_obs = chunk_decode_fn(chunks[0])[3]

    logical_flip = np.asarray(predicted_obs).squeeze()
    ler = np.mean(logical_flip.astype(bool) != obvs.astype(bool))
    return ler, logical_flip


def _decode_chunk_belief_matching(chunk_data):
    """单块 BeliefMatching 解码，供多进程调用（须在模块顶层以便 pickle）。"""
    from beliefmatching import BeliefMatching
    i, start_idx, end_idx, dets_chunk, obvs_chunk, dem_chunk = chunk_data
    bm = BeliefMatching(dem_chunk, max_bp_iters=10)
    pred = bm.decode_batch(dets_chunk)
    return i, start_idx, end_idx, pred.reshape(-1, 1) if pred.ndim == 1 else pred


def decode_with_belief_matching(dem, dets, obvs, max_cores=None):
    """用 BeliefMatching 对整批 dets 解码，与 obvs 比较得到 LER。"""
    return _decode_in_chunks(dem, dets, obvs, max_cores, _decode_chunk_belief_matching)


def _decode_chunk_bposd(chunk_data):
    """单块 BPOSD 解码，供多进程调用（须在模块顶层以便 pickle）。"""
    from stimbposd import BPOSD
    i, start_idx, end_idx, dets_chunk, obvs_chunk, dem_chunk = chunk_data
    decoder = BPOSD(dem_chunk, max_bp_iters=20)
    pred = decoder.decode_batch(dets_chunk)
    return i, start_idx, end_idx, pred.reshape(-1, 1) if pred.ndim == 1 else pred


def decode_with_bposd(dem, dets, obvs, max_cores=None):
    """用 BPOSD 对整批 dets 解码，与 obvs 比较得到 LER。"""
    return _decode_in_chunks(dem, dets, obvs, max_cores, _decode_chunk_bposd)


def sample_dets_and_observables(circuit, shots, seed=None):
    """从电路同一次采样中得到检测事件与 observable 翻转。"""
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    if obs.ndim == 2 and obs.shape[1] == 1:
        obs = obs.flatten()
    return dets, obs


def sample_until_logical_errors(
    circuit,
    dem,
    decode_fn,
    target_logical_errors=200,
    batch_size=100_000,
    seed=43,
    max_shots=1_000_000_000,
):
    """采样并解码，直到收集到 target_logical_errors 次逻辑错误。

    Args:
        circuit: stim 电路
        dem: 用于停止判定的 DEM（通常为 ideal_dem）
        decode_fn: 解码函数 (dem, dets, obvs) -> (ler, predicted_obs)
        target_logical_errors: 目标逻辑错误次数，默认 200
        batch_size: 每批采样量
        seed: 随机种子
        max_shots: 最大采样量上限

    Returns:
        (dets, obvs, total_shots, total_logical_errors, decode_time_sec, num_decode_runs)
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    all_dets = []
    all_obs = []
    total_logical_errors = 0
    total_shots = 0
    decode_time_sec = 0.0
    num_decode_runs = 0

    while total_logical_errors < target_logical_errors and total_shots < max_shots:
        dets_batch, obs_batch = sampler.sample(shots=batch_size, separate_observables=True)
        if obs_batch.ndim == 2 and obs_batch.shape[1] == 1:
            obs_batch = obs_batch.flatten()

        t0 = time.perf_counter()
        _, predicted = decode_fn(dem, dets_batch, obs_batch)
        decode_time_sec += time.perf_counter() - t0
        num_decode_runs += 1
        batch_errors = np.count_nonzero(predicted.astype(bool) != obs_batch.astype(bool))
        total_logical_errors += batch_errors
        total_shots += len(obs_batch)
        all_dets.append(dets_batch)
        all_obs.append(obs_batch)

    dets = np.vstack(all_dets) if len(all_dets) > 1 else all_dets[0]
    obvs = np.concatenate(all_obs) if len(all_obs) > 1 else all_obs[0]
    return dets, obvs, total_shots, total_logical_errors, decode_time_sec, num_decode_runs
