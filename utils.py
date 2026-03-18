#!/usr/bin/env python3
"""Shared utility helpers for DEM reconstruction experiments."""

import ast
import os
import time

import stim
import torch


def targets_to_dets(targets):
    """Parse stim instruction targets into a frozenset of detector IDs."""
    return frozenset(t.val for t in targets if t.is_relative_detector_id())


def extract_hyperedge_from_dem(dem: stim.DetectorErrorModel) -> tuple:
    """Parse hyperedges and their probabilities from DEM.

    Args:
        dem: Detector error model

    Returns:
        (hyperedge_probs, hyperedges): hyperedge->prob dict, and hyperedge list
    """
    hyperedge_probs = {}
    for instruction in dem.flattened():
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error":
            dets = targets_to_dets(instruction.targets_copy())
            prob = instruction.args_copy()[0]
            if dets in hyperedge_probs:
                p_prev = hyperedge_probs[dets]
                prob = p_prev * (1 - prob) + prob * (1 - p_prev)
            hyperedge_probs[dets] = prob
    hyperedges = list(hyperedge_probs.keys())
    return hyperedge_probs, hyperedges


def _format_inference_eps(eps):
    """Format inference_eps as a directory-safe string."""
    if isinstance(eps, (list, tuple)):
        return "_".join(str(e) for e in eps)
    return str(eps)


def _is_cuda_device(device):
    """Check if device uses CUDA."""
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.startswith("cuda")
    return False


def _measure_cpu_gpu_time(device, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (cpu_time_sec, gpu_time_sec)."""
    use_cuda = _is_cuda_device(device) and torch.cuda.is_available()
    cpu_start = time.process_time()
    gpu_time_sec = 0.0
    if use_cuda:
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
    result = fn(*args, **kwargs)
    if use_cuda:
        end_ev.record()
        torch.cuda.synchronize()
        gpu_time_sec = start_ev.elapsed_time(end_ev) / 1000.0  # ms -> s
    cpu_time_sec = time.process_time() - cpu_start
    return result, cpu_time_sec, gpu_time_sec


def get_output_dir(code_task, distance, rounds, shots_analysis, inference_eps, base_dir="data"):
    """Get output directory path: data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/"""
    code_task_safe = code_task.replace(":", "_")
    eps_str = _format_inference_eps(inference_eps)
    return os.path.join(base_dir, f"{code_task_safe}_d{distance}r{rounds}", f"{shots_analysis}_{eps_str}")


def get_ca_base_dir(code_task, distance, rounds, base_dir="data"):
    """Get correlation root directory for a code, for decoding multiple groups in one pass."""
    code_task_safe = code_task.replace(":", "_")
    return os.path.join(base_dir, f"{code_task_safe}_d{distance}r{rounds}")


def _to_json_scalar(x):
    """Convert Tensor/numpy scalar to Python native type."""
    return float(x.item()) if hasattr(x, "item") else float(x)


def _ensure_cpu_native(obj):
    """Recursively move tensors to CPU and convert to Python native types to avoid holding GPU memory."""
    if hasattr(obj, "cpu"):
        t = obj.cpu()
        return float(t.item()) if t.numel() == 1 else t.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (dict, list, tuple, set)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _ensure_cpu_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_ensure_cpu_native(x) for x in obj)
    return obj


def _ca_to_json_serializable(ca_results):
    """Convert ca_results to JSON-serializable format (frozenset -> tuple, Tensor -> float)."""
    def conv_key(k):
        return tuple(sorted(k)) if isinstance(k, (frozenset, set)) else k

    out = {"params": ca_results["params"]}
    out["ideal_probs"] = {str(conv_key(k)): _to_json_scalar(v) for k, v in ca_results["ideal_probs"].items()}
    out["given_probs_list"] = [
        {str(conv_key(k)): _to_json_scalar(v) for k, v in d.items()} for d in ca_results["given_probs_list"]
    ]
    out["infer_probs_list"] = [
        {str(conv_key(k)): _to_json_scalar(v) for k, v in d.items()} for d in ca_results["infer_probs_list"]
    ]
    out["ideal_set"] = [tuple(sorted(h)) for h in ca_results["ideal_set"]]
    out["extra_edges"] = [[tuple(sorted(h)) for h in s] for s in ca_results["extra_edges"]]
    out["has_extra"] = ca_results["has_extra"]
    out["all_rows"] = [
        [(tag, tuple(sorted(h)), row) for tag, h, row in rows]
        for rows in ca_results["all_rows"]
    ]
    out["given_cpu_time_list"] = ca_results.get("given_cpu_time_list", [])
    out["given_gpu_time_list"] = ca_results.get("given_gpu_time_list", [])
    out["infer_cpu_time_list"] = ca_results.get("infer_cpu_time_list", [])
    out["infer_gpu_time_list"] = ca_results.get("infer_gpu_time_list", [])
    return out


def _ca_from_json_serializable(data):
    """Restore ca_results from JSON format (tuple -> frozenset)."""
    out = {"params": data["params"]}

    def _parse_key(k):
        return frozenset(ast.literal_eval(k))

    out["ideal_probs"] = {_parse_key(k): v for k, v in data["ideal_probs"].items()}
    out["given_probs_list"] = [
        {_parse_key(k): v for k, v in d.items()} for d in data["given_probs_list"]
    ]
    out["infer_probs_list"] = [
        {_parse_key(k): v for k, v in d.items()} for d in data["infer_probs_list"]
    ]
    out["ideal_set"] = {frozenset(h) for h in data["ideal_set"]}
    out["extra_edges"] = [{frozenset(h) for h in s} for s in data["extra_edges"]]
    out["has_extra"] = data["has_extra"]
    out["all_rows"] = [
        [(tag, frozenset(h), row) for tag, h, row in rows]
        for rows in data["all_rows"]
    ]
    out["given_cpu_time_list"] = data.get("given_cpu_time_list", [])
    out["given_gpu_time_list"] = data.get("given_gpu_time_list", [])
    out["infer_cpu_time_list"] = data.get("infer_cpu_time_list", [])
    out["infer_gpu_time_list"] = data.get("infer_gpu_time_list", [])
    return out


def _fmt_row(h_str, p_id_s, p_gv_s, gv_e, p_if_s, if_e):
    """Format per-hyperedge table row."""
    return f"{h_str:<20} {p_id_s:>12} {p_gv_s:>12} {gv_e:>10} {p_if_s:>12} {if_e:>10}"


def _sort_hyperedge_key(edge):
    """Sort key for deterministic hyperedge display."""
    return len(edge), sorted(edge)


def _build_decomposed_targets(dets_frozenset):
    """Decompose hyperedge into graphlike (<=2 detector) components for belief_matching decode.

    Matches circuit.detector_error_model(decompose_errors=True) format:
    - <=2 body: return detector targets directly, no decomposition
    - 3+ body: split into disjoint <=2 body components, joined by ^ (separator)
      e.g. {D0,D1,D2} -> D0 D1 ^ D2
           {D0,D1,D2,D3} -> D0 D1 ^ D2 D3
    """
    sorted_dets = sorted(dets_frozenset)
    n = len(sorted_dets)

    if n <= 2:
        return [stim.target_relative_detector_id(d) for d in sorted_dets]

    targets = []
    for j in range(0, n, 2):
        if j > 0:
            targets.append(stim.target_separator())
        targets.append(stim.target_relative_detector_id(sorted_dets[j]))
        if j + 1 < n:
            targets.append(stim.target_relative_detector_id(sorted_dets[j + 1]))
    return targets
