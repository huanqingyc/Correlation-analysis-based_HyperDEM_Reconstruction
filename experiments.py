#!/usr/bin/env python3
"""LER comparison experiment: Ideal vs Given DEM vs Inference DEM.

Standalone script with two entry points:
1. run_correlation_analysis: Correlation analysis, output to .json and .txt
2. run_decode_from_files: Load correlation analysis from files and run decode experiment
"""

import json
import os

import stim

from inference_with_correlation_analysis import cal_multi_body_correlations
from decoding import (
    create_dem_from_analysis,
    decode_with_belief_matching,
    decode_with_bposd,
    sample_dets_and_observables,
    sample_until_logical_errors,
)
from utils import (
    _build_decomposed_targets,
    _ca_from_json_serializable,
    _ca_to_json_serializable,
    _ensure_cpu_native,
    _fmt_row,
    _format_inference_eps,
    _is_cuda_device,
    _measure_cpu_gpu_time,
    _sort_hyperedge_key,
    extract_hyperedge_from_dem,
    get_output_dir,
)


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


def run_correlation_analysis(
    distance=5,
    rounds=5,
    p_circuit=0.002,
    shots_analysis_list=None,
    max_order=2,
    inference_eps=(1e-4, 1e-5),
    code_task="repetition_code:memory",
    device="cpu",
    CA_mode=("given", "inference"),
    base_dir="data",
    batch_shots=100000,
):
    """Run correlation analysis, output to data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/.

    Correlation analysis only, no decode params (decoder etc.) stored. Decode params set by run_decode_from_files.

    Each (shots_analysis, inference_eps) combo maps to a subdir with correlation.json, correlation.txt.

    Args:
        distance, rounds, p_circuit, code_task: Circuit params
        shots_analysis_list: List of fit sample sizes
        max_order, inference_eps: Correlation analysis params
        CA_mode: ['given','inference']
        base_dir: Output root dir, default "data"
        batch_shots: Shots per batch for correlation analysis
    Returns:
        tuple: (ca_results dict, ca_json_paths list, ca_txt_paths list)
    """
    if shots_analysis_list is None:
        shots_analysis_list = [1000, 10000, 100000, 1000000]

    shots_analysis = max(shots_analysis_list)
    use_decompose = True  # Correlation analysis uses graphlike format

    circuit, _, _, num_dets = generate_test_circuit(
        distance=distance,
        rounds=rounds,
        shots=shots_analysis,
        code_task=code_task,
        after_clifford_depolarization=p_circuit,
        before_round_data_depolarization=p_circuit,
        before_measure_flip_probability=p_circuit,
        after_reset_flip_probability=p_circuit,
    )
    ideal_dem = circuit.detector_error_model(decompose_errors=use_decompose)
    ideal_probs, hyperedges = extract_hyperedge_from_dem(ideal_dem)
    print(f"Num detectors: {num_dets}, num hyperedges: {len(ideal_probs)}")

    dets_analysis, _ = sample_dets_and_observables(circuit, shots_analysis, seed=42)

    ideal_set = set(ideal_probs.keys())
    n_shots = len(shots_analysis_list)
    empty_list = [0.0] * n_shots

    given_probs_list = []
    given_cpu_time_list = []
    given_gpu_time_list = []
    if "given" in CA_mode:
        given_max_order = max(len(h) for h in hyperedges) if hyperedges else 1
        hyperedge_list = [set() for _ in range(given_max_order)]
        for h in hyperedges:
            hyperedge_list[len(h) - 1].add(h)
        for sa in shots_analysis_list:
            (p_given, _), cpu_t, gpu_t = _measure_cpu_gpu_time(
                device,
                lambda s=sa: cal_multi_body_correlations(
                    dets_analysis[:s],
                    mode="given_dem_topology",
                    hyperedge_list=hyperedge_list,
                    correct_in_step=False,
                    device=device,
                    batch_shots=batch_shots,
                ),
            )
            given_probs = {k: v for d in p_given for k, v in d.items()}
            given_cpu_time_list.append(cpu_t)
            given_gpu_time_list.append(gpu_t)
            print(f"given_dem_topology correlation analysis: {len(given_probs)} hyperedges, CPU {cpu_t:.6f}s, GPU {gpu_t:.6f}s")
            given_probs_list.append(given_probs)
    else:
        given_probs_list = [{} for _ in range(n_shots)]
        given_cpu_time_list = list(empty_list)
        given_gpu_time_list = list(empty_list)

    infer_probs_list = []
    infer_cpu_time_list = []
    infer_gpu_time_list = []
    extra_edges = []
    has_extra = []
    if "inference" in CA_mode:
        for sa in shots_analysis_list:
            (p_infer, _), cpu_t, gpu_t = _measure_cpu_gpu_time(
                device,
                lambda s=sa: cal_multi_body_correlations(
                    dets_analysis[:s],
                    mode="inference",
                    max_order=max_order,
                    device=device,
                    eps=inference_eps,
                    batch_shots=batch_shots,
                ),
            )
            infer_probs = {k: v for d in p_infer for k, v in d.items()}
            infer_cpu_time_list.append(cpu_t)
            infer_gpu_time_list.append(gpu_t)
            print(f"inference correlation analysis: {len(infer_probs)} hyperedges, CPU {cpu_t:.6f}s, GPU {gpu_t:.6f}s")
            infer_probs_list.append(infer_probs)
        for i, infer_probs in enumerate(infer_probs_list):
            infer_set = set(infer_probs.keys())
            extra = infer_set - ideal_set
            extra_edges.append(extra)
            has_extra.append(len(extra) > 0)
            print(
                f"\nIdeal hyperedges: {len(ideal_set)}, Inference hyperedges: {len(infer_set)}, "
                f"Inference extra edges: {len(extra)}"
            )
    else:
        infer_probs_list = [{} for _ in range(n_shots)]
        infer_cpu_time_list = list(empty_list)
        infer_gpu_time_list = list(empty_list)
        extra_edges = [set() for _ in range(n_shots)]
        has_extra = [False] * n_shots

    all_rows = []
    for i, sa in enumerate(shots_analysis_list):
        print(f"Correlation analysis with {sa} shots")
        header = _fmt_row("Hyperedge", "ideal p", "given", "given err%", "infer p", "infer err%")
        sep = "-" * 90
        print("\n" + "=" * 90)
        print(header)
        print(sep)
        rows_i = []
        for h in sorted(ideal_set, key=_sort_hyperedge_key):
            p_id = ideal_probs[h]
            p_gv = given_probs_list[i].get(h) if "given" in CA_mode else None
            p_if = infer_probs_list[i].get(h) if "inference" in CA_mode else None
            p_id_f = float(p_id)
            gv_e = (
                f"{abs(float(p_gv) - p_id_f) / p_id_f * 100:.2f}"
                if (p_gv is not None and abs(p_id_f) > 1e-10)
                else "-"
            )
            if_e = (
                f"{abs(float(p_if) - p_id_f) / p_id_f * 100:.2f}"
                if (p_if is not None and abs(p_id_f) > 1e-10)
                else "-"
            )
            p_gv_s = f"{float(p_gv):.8f}" if p_gv is not None else "     -      "
            p_if_s = f"{float(p_if):.8f}" if p_if is not None else "     -      "
            row = _fmt_row(str(tuple(sorted(h))), f"{p_id_f:.8f}", p_gv_s, gv_e, p_if_s, if_e)
            print(row)
            rows_i.append(("ideal", h, row))
        if "inference" in CA_mode and has_extra[i]:
            print(sep)
            print(f">>> Inference extra hyperedges ({len(extra_edges[i])}, not in ideal/given):")
            print(sep)
            for h in sorted(extra_edges[i], key=_sort_hyperedge_key):
                p_if = infer_probs_list[i][h]
                row = _fmt_row(
                    str(tuple(sorted(h))), "     -      ", "     -      ", "-",
                    f"{float(p_if):.8f}", "-"
                )
                print(row)
                rows_i.append(("extra", h, row))
        all_rows.append(rows_i)

    # Ensure all results are on CPU / Python native to avoid holding GPU memory in notebook
    ca_results = {
        "params": {
            "distance": distance,
            "rounds": rounds,
            "p_circuit": p_circuit,
            "code_task": code_task,
            "max_order": max_order,
            "inference_eps": inference_eps,
            "shots_analysis_list": shots_analysis_list,
            "batch_shots": batch_shots,
        },
        "ideal_probs": _ensure_cpu_native(ideal_probs),
        "given_probs_list": _ensure_cpu_native(given_probs_list),
        "infer_probs_list": _ensure_cpu_native(infer_probs_list),
        "ideal_set": ideal_set,
        "extra_edges": extra_edges,
        "has_extra": has_extra,
        "all_rows": all_rows,
        "given_cpu_time_list": given_cpu_time_list,
        "given_gpu_time_list": given_gpu_time_list,
        "infer_cpu_time_list": infer_cpu_time_list,
        "infer_gpu_time_list": infer_gpu_time_list,
    }
    if _is_cuda_device(device):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ca_json_paths = []
    ca_txt_paths = []
    for i, sa in enumerate(shots_analysis_list):
        out_dir = get_output_dir(code_task, distance, rounds, sa, inference_eps, base_dir)
        os.makedirs(out_dir, exist_ok=True)

        ca_results_i = {
            "params": {
                **ca_results["params"],
                "shots_analysis_list": [sa],
            },
            "ideal_probs": ideal_probs,
            "given_probs_list": [given_probs_list[i]],
            "infer_probs_list": [infer_probs_list[i]],
            "ideal_set": ideal_set,
            "extra_edges": [extra_edges[i]],
            "has_extra": [has_extra[i]],
            "all_rows": [all_rows[i]],
            "given_cpu_time_list": [given_cpu_time_list[i]],
            "given_gpu_time_list": [given_gpu_time_list[i]],
            "infer_cpu_time_list": [infer_cpu_time_list[i]],
            "infer_gpu_time_list": [infer_gpu_time_list[i]],
        }

        ca_json_path = os.path.join(out_dir, "correlation.json")
        ca_txt_path = os.path.join(out_dir, "correlation.txt")

        with open(ca_json_path, "w", encoding="utf-8") as f:
            json.dump(_ca_to_json_serializable(ca_results_i), f, indent=2, ensure_ascii=False)
        print(f"Correlation analysis saved: {ca_json_path}")

        given_cpu = given_cpu_time_list[i]
        given_gpu = given_gpu_time_list[i]
        infer_cpu = infer_cpu_time_list[i]
        infer_gpu = infer_gpu_time_list[i]
        with open(ca_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("Correlation Analysis: Ideal vs Given vs Inference\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Code task: {code_task}, d={distance}, r={rounds}, p={p_circuit}\n")
            f.write(f"shots_analysis={sa}, max_order={max_order}\n")
            f.write(f"Correlation analysis timing:\n")
            f.write(f"  Given DEM:   CPU core time: {given_cpu:.4f}s, GPU runtime: {given_gpu:.4f}s\n")
            f.write(f"  Inference:   CPU core time: {infer_cpu:.4f}s, GPU runtime: {infer_gpu:.4f}s\n")
            f.write(f"Ideal hyperedges: {len(ideal_set)}\n\n")
            f.write(f"correlation analysis with {sa} shots\n")
            f.write("\n" + "=" * 90 + "\n")
            f.write("Per-hyperedge probabilities:\n")
            f.write("=" * 90 + "\n")
            f.write(_fmt_row("Hyperedge", "Ideal", "Given", "G_err%", "Inference", "I_err%") + "\n")
            f.write("-" * 90 + "\n")
            for _tag, _h, row in all_rows[i]:
                f.write(row + "\n")
        print(f"Correlation table saved: {ca_txt_path}")

        ca_json_paths.append(ca_json_path)
        ca_txt_paths.append(ca_txt_path)

    return ca_results, ca_json_paths, ca_txt_paths


def collect_ca_from_base_dir(
    ca_base_dir: str,
    shots_analysis_list=None,
    inference_eps=None,
) -> dict:
    """Collect correlation.json from subdirs under base_dir, merge into multi-group ca_results.

    For decoding multiple groups in one pass. Dir structure: ca_base_dir/{shots}_{eps}/correlation.json

    Args:
        ca_base_dir: e.g. data/surface_code_rotated_memory_x_d5r5/
        shots_analysis_list: Optional, only collect specified shots groups, e.g. [150000, 1500000, 15000000]
        inference_eps: Optional, only collect matching inference_eps groups

    Returns:
        Merged ca_results, shots_analysis_list contains all groups
    """
    ca_base_dir = os.path.normpath(ca_base_dir)
    if not os.path.isdir(ca_base_dir):
        raise FileNotFoundError(f"Directory does not exist: {ca_base_dir}")

    shots_set = set(shots_analysis_list) if shots_analysis_list is not None else None
    eps_str = _format_inference_eps(inference_eps) if inference_eps is not None else None

    entries = []
    for sub in os.listdir(ca_base_dir):
        child = os.path.join(ca_base_dir, sub)
        if not os.path.isdir(child):
            continue
        parts = sub.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            sa = int(parts[0])
        except ValueError:
            continue
        if shots_set is not None and sa not in shots_set:
            continue
        if eps_str is not None and parts[1] != eps_str:
            continue
        p = os.path.join(child, "correlation.json")
        if os.path.isfile(p):
            entries.append((sub, p))

    if not entries:
        filter_msg = ""
        if shots_set is not None or eps_str is not None:
            filter_msg = f"(filter shots={shots_analysis_list}, inference_eps={inference_eps})"
        raise FileNotFoundError(f"No matching correlation.json under {ca_base_dir} {filter_msg}")

    def _sort_key(item):
        try:
            return int(item[0].split("_", 1)[0])
        except (ValueError, IndexError):
            return 0

    entries.sort(key=_sort_key)
    results = [load_correlation_analysis(p) for _, p in entries]

    shots_analysis_list = [r["params"]["shots_analysis_list"][0] for r in results]
    inference_eps_list = [r["params"].get("inference_eps", 0) for r in results]

    merged = {
        "params": {
            **results[0]["params"],
            "shots_analysis_list": shots_analysis_list,
            "inference_eps_list": inference_eps_list,
        },
        "ideal_probs": results[0]["ideal_probs"],
        "ideal_set": results[0]["ideal_set"],
        "given_probs_list": [r["given_probs_list"][0] for r in results],
        "infer_probs_list": [r["infer_probs_list"][0] for r in results],
        "extra_edges": [r["extra_edges"][0] for r in results],
        "has_extra": [r["has_extra"][0] for r in results],
        "all_rows": [r["all_rows"][0] for r in results],
        "given_cpu_time_list": [r.get("given_cpu_time_list", [0.0])[0] for r in results],
        "given_gpu_time_list": [r.get("given_gpu_time_list", [0.0])[0] for r in results],
        "infer_cpu_time_list": [r.get("infer_cpu_time_list", [0.0])[0] for r in results],
        "infer_gpu_time_list": [r.get("infer_gpu_time_list", [0.0])[0] for r in results],
    }
    return merged


def load_correlation_analysis(ca_path: str) -> dict:
    """Load correlation analysis from .json file.

    - File path: data/{code}_d{d}r{r}/{shots}_{eps}/correlation.json
    - Dir path: data/{code}_d{d}r{r}/{shots}_{eps}/ auto-finds correlation.json
    """
    ca_path = str(ca_path)
    if os.path.isdir(ca_path):
        json_path = os.path.join(ca_path, "correlation.json")
        if os.path.isfile(json_path):
            ca_path = json_path
        else:
            raise FileNotFoundError(f"correlation.json not found in directory {ca_path}")
    with open(ca_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _ca_from_json_serializable(data)


def run_decode_from_files(
    ca_path,
    target_logical_errors=200,
    batch_size=100_000,
    max_shots=1_000_000_000,
    decoder=None,
    shots_analysis_list=None,
    inference_eps=None,
    base_dir="data",
):
    """Load correlation analysis from files and run decode experiment.

    Args:
        ca_path: Correlation analysis path, supports:
            - Single file: correlation.json path
            - Single dir: dir containing correlation.json
            - Multi-group dir: e.g. data/surface_code_rotated_memory_x_d5r5/, collect from subdirs
        target_logical_errors: Target logical errors, default 200
        batch_size: Shots per batch
        max_shots: Max shots upper limit
        decoder: Decoder, 'belief_matching' or 'bposd'
        shots_analysis_list: For multi-group dir, only decode specified shots groups
        inference_eps: For multi-group dir, only decode matching inference_eps groups
        base_dir: Output root dir, default "data"

    Returns:
        list[str]: ler.txt paths for each shots_analysis
    """
    path = str(ca_path)
    if os.path.isdir(path):
        direct_json = os.path.join(path, "correlation.json")
        if os.path.isfile(direct_json):
            ca_results = load_correlation_analysis(path)
        else:
            ca_results = collect_ca_from_base_dir(
                path,
                shots_analysis_list=shots_analysis_list,
                inference_eps=inference_eps,
            )
    else:
        ca_results = load_correlation_analysis(path)

    return run_decode(
        ca_results=ca_results,
        target_logical_errors=target_logical_errors,
        batch_size=batch_size,
        max_shots=max_shots,
        decoder=decoder,
        base_dir=base_dir,
    )


def run_decode(
    ca_results=None,
    ca_path=None,
    target_logical_errors=200,
    batch_size=100_000,
    max_shots=1_000_000_000,
    decoder=None,
    output_dir=None,
    base_dir="data",
):
    """Decode based on correlation analysis results, compare LER.

    Sample and decode until target_logical_errors logical errors (ideal_dem as criterion),
    then evaluate each DEM's LER on the same batch.

    Pass ca_results from memory or load from ca_path.

    Args:
        ca_results: Correlation analysis dict (from run_correlation_analysis)
        ca_path: Or load from .json file path
        target_logical_errors: Target logical errors, default 200
        batch_size: Shots per batch
        max_shots: Max shots limit, prevent infinite loop for very low LER
        decoder: Override if inconsistent with ca_results
        output_dir: Deprecated, kept for compatibility; use base_dir
        base_dir: Output root dir, default "data", LER written to data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/ler.txt

    Returns:
        list[str]: ler.txt paths for each shots_analysis
    """
    if ca_results is None and ca_path is None:
        raise ValueError("Must provide ca_results or ca_path")
    if ca_results is None:
        ca_results = load_correlation_analysis(ca_path)
    _ = output_dir  # kept for compatibility

    # Support merged multi-group data from ca_base_dir (incl. inference_eps_list)
    if "inference_eps_list" in ca_results["params"]:
        inference_eps_list = ca_results["params"]["inference_eps_list"]
    else:
        inference_eps_list = None

    params = ca_results["params"]
    distance = params["distance"]
    rounds = params["rounds"]
    p_circuit = params["p_circuit"]
    code_task = params["code_task"]
    dec = decoder or params.get("decoder", "belief_matching")
    shots_analysis_list = params["shots_analysis_list"]
    given_probs_list = ca_results["given_probs_list"]
    infer_probs_list = ca_results["infer_probs_list"]
    ideal_set = ca_results["ideal_set"]
    extra_edges = ca_results["extra_edges"]
    has_extra = ca_results["has_extra"]

    _modes = []
    if any(given_probs_list):
        _modes.append("given")
    if any(infer_probs_list):
        _modes.append("inference")
    CA_mode = tuple(_modes) if _modes else params.get("CA_mode", ("given", "inference"))

    if dec not in ("belief_matching", "bposd"):
        raise ValueError(f"decoder must be 'belief_matching' or 'bposd', got {dec!r}")
    decode_fn = decode_with_belief_matching if dec == "belief_matching" else decode_with_bposd

    circuit, _, _, _ = generate_test_circuit(
        distance=distance,
        rounds=rounds,
        shots=batch_size,
        code_task=code_task,
        after_clifford_depolarization=p_circuit,
        before_round_data_depolarization=p_circuit,
        before_measure_flip_probability=p_circuit,
        after_reset_flip_probability=p_circuit,
    )
    use_decompose = dec == "belief_matching"
    ideal_dem = circuit.detector_error_model(decompose_errors=use_decompose)

    print(f"\nUsing decoder: {dec}, sampling until {target_logical_errors} logical errors collected...")
    dets_decode, obs_decode, total_shots, total_logical_errors, _, _ = sample_until_logical_errors(
        circuit, ideal_dem, decode_fn,
        target_logical_errors=target_logical_errors,
        batch_size=batch_size,
        seed=43,
        max_shots=max_shots,
    )
    ler_ideal = total_logical_errors / total_shots
    print(f"  Actual sampled: {total_shots} shots, logical errors: {total_logical_errors}")
    print("\n" + "=" * 70)
    print("LER comparison (held-out decode sample)")
    print("=" * 70)
    print(f"  Ideal DEM LER:                {ler_ideal:.6f}")

    ler_paths = []
    for i, sa in enumerate(shots_analysis_list):
        eps = inference_eps_list[i] if inference_eps_list is not None else params.get("inference_eps", 0)
        out_dir = get_output_dir(code_task, distance, rounds, sa, eps, base_dir)
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "ler.txt")
        ler_paths.append(txt_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("LER Comparison: Ideal DEM vs Given DEM Topology vs Infer DEM\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Code task: {code_task}, d={distance}, r={rounds}, p={p_circuit}\n")
            f.write(f"shots_analysis={sa}, target_logical_errors={target_logical_errors}, total_shots={total_shots}, max_order={params['max_order']}, decoder={dec}\n")
            f.write(f"Ideal hyperedges: {len(ideal_set)}\n")
            f.write(f"Ideal DEM LER:                {ler_ideal:.6f}\n")
        if "given" in CA_mode:
            given_dem_topology = create_dem_from_analysis(ideal_dem, given_probs_list[i])
            ler_given_dem_topology, _ = decode_fn(given_dem_topology, dets_decode, obs_decode)
        else:
            ler_given_dem_topology = None
        if "inference" in CA_mode:
            infer_set = set(infer_probs_list[i].keys())
            infer_dem_base = stim.DetectorErrorModel()
            for instr in ideal_dem.flattened():
                infer_dem_base.append(instr)
            for h in extra_edges[i]:
                p_val = float(infer_probs_list[i][h])
                if p_val <= 0 or p_val >= 1:
                    continue
                targets = (
                    [stim.target_relative_detector_id(d) for d in sorted(h)]
                    if dec == "bposd"
                    else _build_decomposed_targets(h)
                )
                infer_dem_base.append(stim.DemInstruction("error", args=[p_val], targets=targets))
            infer_dem = create_dem_from_analysis(infer_dem_base, infer_probs_list[i])
            ler_infer_dem, _ = decode_fn(infer_dem, dets_decode, obs_decode)
        else:
            ler_infer_dem = None

        if "given" in CA_mode:
            print(
                f"  Given DEM Topology LER from {sa} shots:       "
                f"{ler_given_dem_topology:.6f}  (gap: {ler_ideal - ler_given_dem_topology:+.6f})"
            )
        if "inference" in CA_mode:
            print(
                f"  Infer DEM LER from {sa} shots:                "
                f"{ler_infer_dem:.6f}  (gap: {ler_ideal - ler_infer_dem:+.6f})"
            )

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\ncorrelation analysis with {sa} shots\n")
            f.write(
                f"Given DEM Topology LER:       {ler_given_dem_topology:.6f}  (gap: {ler_ideal - ler_given_dem_topology:+.6f})\n"
                if ler_given_dem_topology is not None
                else "Given DEM Topology LER:       -      \n"
            )
            if "inference" in CA_mode:
                f.write(f"Infer hyperedges: {len(infer_set)}, extra edges: {len(extra_edges[i])}\n")
                f.write(f"Infer DEM LER:                {ler_infer_dem:.6f}  (gap: {ler_ideal - ler_infer_dem:+.6f})\n")
            else:
                f.write("Infer hyperedges: -, extra edges: -      \n")

    return ler_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LER comparison: correlation analysis or decode from files")
    parser.add_argument("--ca-path", type=str, default=None,
                        help="If provided, run run_decode_from_files; else run run_correlation_analysis")
    parser.add_argument("--decoder", choices=["belief_matching", "bposd"], default="belief_matching")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--code-task", default="repetition_code:memory")
    parser.add_argument("--shots-analysis-list", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--target-logical-errors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-shots", type=int, default=1_000_000_000)
    parser.add_argument("--base-dir", type=str, default="data")
    args, _ = parser.parse_known_args()

    if args.ca_path:
        ler_paths = run_decode_from_files(
            args.ca_path,
            target_logical_errors=args.target_logical_errors,
            batch_size=args.batch_size,
            max_shots=args.max_shots,
            decoder=args.decoder,
            base_dir=args.base_dir,
        )
        print(f"\nLER output:")
        for p in ler_paths:
            print(f"  {p}")
    else:
        ca_results, ca_json_paths, ca_txt_paths = run_correlation_analysis(
            distance=args.distance,
            rounds=args.rounds,
            p_circuit=0.002,
            shots_analysis_list=args.shots_analysis_list,
            max_order=2,
            inference_eps=0,
            code_task=args.code_task,
            device="cpu",
            base_dir=args.base_dir,
        )
        print(f"\nCorrelation analysis output:")
        for p in ca_json_paths:
            print(f"  {p}")
