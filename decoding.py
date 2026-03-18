#!/usr/bin/env python3
"""Decoding and sampling helpers for LER comparison."""

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

from utils import targets_to_dets


def _flatten_single_observable(obs):
    """Flatten (N,1) observable array to (N,), keep other shapes unchanged."""
    if obs.ndim == 2 and obs.shape[1] == 1:
        return obs.flatten()
    return obs


def create_dem_from_analysis(reference_dem, hyperedge_probs):
    """Replace reference DEM probabilities with hyperedge probs from correlation analysis.

    Args:
        reference_dem: Reference DEM (structure unchanged)
        hyperedge_probs: hyperedge (frozenset) -> probability

    Returns:
        New DEM
    """
    from utils import extract_hyperedge_from_dem
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
    """Generic chunked decode: build chunks, call chunk_decode_fn in parallel or serial, return (ler, predicted_obs)."""
    n_shots = dets.shape[0]
    if n_shots == 0:
        return 0.0, np.array([], dtype=bool)
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
    """Single-chunk BeliefMatching decode, for multiprocessing (must be at module top for pickle)."""
    from beliefmatching import BeliefMatching
    i, start_idx, end_idx, dets_chunk, _obvs_chunk, dem_chunk = chunk_data
    bm = BeliefMatching(dem_chunk, max_bp_iters=10)
    pred = bm.decode_batch(dets_chunk)
    return i, start_idx, end_idx, pred.reshape(-1, 1) if pred.ndim == 1 else pred


def decode_with_belief_matching(dem, dets, obvs, max_cores=None):
    """Decode full batch of dets with BeliefMatching, compare with obvs to get LER."""
    return _decode_in_chunks(dem, dets, obvs, max_cores, _decode_chunk_belief_matching)


def _decode_chunk_bposd(chunk_data):
    """Single-chunk BPOSD decode, for multiprocessing (must be at module top for pickle)."""
    from stimbposd import BPOSD
    i, start_idx, end_idx, dets_chunk, _obvs_chunk, dem_chunk = chunk_data
    decoder = BPOSD(dem_chunk, max_bp_iters=20)
    pred = decoder.decode_batch(dets_chunk)
    return i, start_idx, end_idx, pred.reshape(-1, 1) if pred.ndim == 1 else pred


def decode_with_bposd(dem, dets, obvs, max_cores=None):
    """Decode full batch of dets with BPOSD, compare with obvs to get LER."""
    return _decode_in_chunks(dem, dets, obvs, max_cores, _decode_chunk_bposd)


def sample_dets_and_observables(circuit, shots, seed=None):
    """Sample detection events and observable flips from the circuit in one shot."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return dets, _flatten_single_observable(obs)


def sample_until_logical_errors(
    circuit,
    dem,
    decode_fn,
    target_logical_errors=200,
    batch_size=100_000,
    seed=43,
    max_shots=1_000_000_000,
):
    """Sample and decode until target_logical_errors logical errors are collected.

    Args:
        circuit: stim circuit
        dem: DEM for stop criterion (usually ideal_dem)
        decode_fn: Decode function (dem, dets, obvs) -> (ler, predicted_obs)
        target_logical_errors: Target number of logical errors, default 200
        batch_size: Shots per batch
        seed: Random seed
        max_shots: Max shots upper limit

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
        obs_batch = _flatten_single_observable(obs_batch)

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
