#!/usr/bin/env python3
"""LER 比较实验：Ideal vs Given DEM vs Inference DEM。

独立脚本，提供两个入口：
1. run_correlation_analysis: 关联分析，输出至 .json 和 .txt
2. run_decode_from_files: 从文件读取关联分析结果并执行解码实验
"""

import ast
import json
import os
import time
import stim
from function import generate_test_circuit, cal_multi_body_correlations, targets_to_dets
from compare_ler_correlation_vs_ideal import (
    create_dem_from_analysis,
    decode_with_belief_matching,
    decode_with_bposd,
    sample_dets_and_observables,
    sample_until_logical_errors,
)


def extract_hyperedge_from_dem(dem: stim.DetectorErrorModel) -> tuple:
    """从 DEM 解析超边及其概率。

    Args:
        dem: 检测器错误模型

    Returns:
        (hyperedge_probs, hyperedges): 超边->概率的 dict，以及超边列表
    """
    hyperedges = []
    hyperedge_probs = {}

    for instruction in dem.flattened():
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error":
            dets = targets_to_dets(instruction.targets_copy())
            prob = instruction.args_copy()[0]
            if dets not in hyperedge_probs:
                hyperedges.append(dets)
                hyperedge_probs[dets] = prob
            else:
                prob_prev = hyperedge_probs[dets]
                hyperedge_probs[dets] = prob_prev * (1 - prob) + prob * (1 - prob_prev)

    return hyperedge_probs, hyperedges


def _format_inference_eps(eps):
    """将 inference_eps 格式化为可作目录名的字符串。"""
    if isinstance(eps, (list, tuple)):
        return "_".join(str(e) for e in eps)
    return str(eps)


def get_output_dir(code_task, distance, rounds, shots_analysis, inference_eps, base_dir="data"):
    """获取输出目录路径：data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/"""
    code_task_safe = code_task.replace(":", "_")
    eps_str = _format_inference_eps(inference_eps)
    return os.path.join(base_dir, f"{code_task_safe}_d{distance}r{rounds}", f"{shots_analysis}_{eps_str}")

def get_ca_base_dir(code_task, distance, rounds, base_dir="data"):
    """获取某代码的 correlation 根目录，用于一次遍历多组数据的解码。"""
    code_task_safe = code_task.replace(":", "_")
    return os.path.join(base_dir, f"{code_task_safe}_d{distance}r{rounds}")

def _to_json_scalar(x):
    """将 Tensor/numpy 标量转为 Python 原生类型。"""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

def _ca_to_json_serializable(ca_results):
    """将 ca_results 转为 JSON 可序列化格式（frozenset -> tuple, Tensor -> float）。"""
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
    return out

def _ca_from_json_serializable(data):
    """从 JSON 格式还原为 ca_results（tuple -> frozenset）。"""
    def to_frozenset(x):
        return frozenset(x) if isinstance(x, (list, tuple)) else x

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
    return out

def _fmt_row(h_str, p_id_s, p_gv_s, gv_e, p_if_s, if_e):
    """格式化 per-hyperedge 表格行。"""
    return f"{h_str:<20} {p_id_s:>12} {p_gv_s:>12} {gv_e:>10} {p_if_s:>12} {if_e:>10}"

def _build_decomposed_targets(dets_frozenset):
    """将超边分解为 graphlike (≤2 detector) 组件，兼容 belief_matching 解码。

    与 circuit.detector_error_model(decompose_errors=True) 的格式一致：
    - ≤2 体：直接返回 detector targets，无需分解
    - 3+ 体：拆成不相交的 ≤2 体组件，用 ^ (separator) 连接
      例如 {D0,D1,D2} → D0 D1 ^ D2
           {D0,D1,D2,D3} → D0 D1 ^ D2 D3
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
):
    """运行关联分析，输出至 data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/ 目录。

    仅做关联分析，不存储解码相关参数（decoder 等）。解码时由 run_decode_from_files 单独指定。

    每个 (shots_analysis, inference_eps) 组合对应一个子目录，内含 correlation.json、correlation.txt。

    Args:
        distance, rounds, p_circuit, code_task: 电路参数
        shots_analysis_list: 拟合样本量列表
        max_order, inference_eps: 关联分析参数
        CA_mode: ['given','inference']
        base_dir: 输出根目录，默认 "data"

    Returns:
        tuple: (ca_results dict, ca_json_paths list, ca_txt_paths list)
    """
    if shots_analysis_list is None:
        shots_analysis_list = [1000, 10000, 100000, 1000000]

    shots_analysis = max(shots_analysis_list)
    use_decompose = True  # 关联分析使用 graphlike 格式

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
    print(f"检测器数量: {num_dets}, 超边数量: {len(ideal_probs)}")

    dets_analysis, _ = sample_dets_and_observables(circuit, shots_analysis, seed=42)

    given_probs_list = []
    given_time_list = []
    if "given" in CA_mode:
        given_max_order = max(len(h) for h in hyperedges) if hyperedges else 1
        hyperedge_list = [set() for _ in range(given_max_order)]
        for h in hyperedges:
            hyperedge_list[len(h) - 1].add(h)
        for sa in shots_analysis_list:
            t0 = time.time()
            p_given, _ = cal_multi_body_correlations(
                dets_analysis[:sa],
                mode="given_dem",
                hyperedge_list=hyperedge_list,
                correct_in_step=False,
                device=device,
            )
            given_probs = {k: v for d in p_given for k, v in d.items()}
            dt = time.time() - t0
            given_time_list.append(dt)
            print(f"given_dem 关联分析: {len(given_probs)} 超边, 耗时 {dt:.6f}s")
            given_probs_list.append(given_probs)
    else:
        given_probs_list = [{} for _ in shots_analysis_list]
        given_time_list = [0.0] * len(shots_analysis_list)

    infer_probs_list = []
    infer_time_list = []
    ideal_set = set(ideal_probs.keys())
    extra_edges = []
    has_extra = []
    if "inference" in CA_mode:
        for sa in shots_analysis_list:
            t0 = time.time()
            p_infer, _ = cal_multi_body_correlations(
                dets_analysis[:sa],
                mode="inference",
                max_order=max_order,
                device=device,
                eps=inference_eps,
            )
            infer_probs = {k: v for d in p_infer for k, v in d.items()}
            dt = time.time() - t0
            infer_time_list.append(dt)
            print(f"inference 关联分析: {len(infer_probs)} 超边, 耗时 {dt:.6f}s")
            infer_probs_list.append(infer_probs)
        for i in range(len(shots_analysis_list)):
            infer_set = set(infer_probs_list[i].keys())
            extra_edges.append(infer_set - ideal_set)
            has_extra.append(len(extra_edges[i]) > 0)
            print(
                f"\nIdeal 超边: {len(ideal_set)}, Inference 超边: {len(infer_set)}, "
                f"Inference 多余边: {len(extra_edges[i])}"
            )
    else:
        infer_probs_list = [{} for _ in shots_analysis_list]
        infer_time_list = [0.0] * len(shots_analysis_list)
        extra_edges = [set() for _ in shots_analysis_list]
        has_extra = [False] * len(shots_analysis_list)

    all_rows = []
    for i in range(len(shots_analysis_list)):
        print(f"Correlation analysis with {shots_analysis_list[i]} shots")
        header = _fmt_row("超边", "ideal p", "given", "given err%", "infer p", "infer err%")
        sep = "-" * 90
        print("\n" + "=" * 90)
        print(header)
        print(sep)
        all_rows.append([])
        for h in sorted(ideal_set, key=lambda x: (len(x), sorted(x))):
            p_id = ideal_probs[h]
            p_gv = given_probs_list[i].get(h, None) if "given" in CA_mode else None
            p_if = infer_probs_list[i].get(h, None) if "inference" in CA_mode else None
            gv_e = (
                f"{abs(float(p_gv) - float(p_id)) / float(p_id) * 100:.2f}"
                if (p_gv is not None and abs(float(p_id)) > 1e-10)
                else "-"
            )
            if_e = (
                f"{abs(float(p_if) - float(p_id)) / float(p_id) * 100:.2f}"
                if (p_if is not None and abs(float(p_id)) > 1e-10)
                else "-"
            )
            p_gv_s = f"{float(p_gv):.8f}" if p_gv is not None else "     -      "
            p_if_s = f"{float(p_if):.8f}" if p_if is not None else "     -      "
            row = _fmt_row(str(tuple(sorted(h))), f"{float(p_id):.8f}", p_gv_s, gv_e, p_if_s, if_e)
            print(row)
            all_rows[i].append(("ideal", h, row))
        if "inference" in CA_mode and has_extra[i]:
            print(sep)
            print(f">>> Inference 额外发现的超边 ({len(extra_edges[i])} 条，ideal/given 中不存在):")
            print(sep)
            for h in sorted(extra_edges[i], key=lambda x: (len(x), sorted(x))):
                p_if = infer_probs_list[i][h]
                row = _fmt_row(
                    str(tuple(sorted(h))), "     -      ", "     -      ", "-",
                    f"{float(p_if):.8f}", "-"
                )
                print(row)
                all_rows[i].append(("extra", h, row))

    ca_results = {
        "params": {
            "distance": distance,
            "rounds": rounds,
            "p_circuit": p_circuit,
            "code_task": code_task,
            "max_order": max_order,
            "inference_eps": inference_eps,
            "shots_analysis_list": shots_analysis_list,
        },
        "ideal_probs": ideal_probs,
        "given_probs_list": given_probs_list,
        "infer_probs_list": infer_probs_list,
        "ideal_set": ideal_set,
        "extra_edges": extra_edges,
        "has_extra": has_extra,
        "all_rows": all_rows,
    }

    ca_json_paths = []
    ca_txt_paths = []
    for i in range(len(shots_analysis_list)):
        sa = shots_analysis_list[i]
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
        }

        ca_json_path = os.path.join(out_dir, "correlation.json")
        ca_txt_path = os.path.join(out_dir, "correlation.txt")

        with open(ca_json_path, "w", encoding="utf-8") as f:
            json.dump(_ca_to_json_serializable(ca_results_i), f, indent=2, ensure_ascii=False)
        print(f"关联分析结果已保存: {ca_json_path}")

        run_time = given_time_list[i] + infer_time_list[i]
        with open(ca_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("Correlation Analysis: Ideal vs Given vs Inference\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Code task: {code_task}, d={distance}, r={rounds}, p={p_circuit}\n")
            f.write(f"shots_analysis={sa}, max_order={max_order}\n")
            f.write(f"运行时间: {run_time:.2f}s\n")
            f.write(f"Ideal 超边: {len(ideal_set)}\n\n")
            f.write(f"correlation analysis with {sa} shots\n")
            f.write("\n" + "=" * 90 + "\n")
            f.write("Per-hyperedge probabilities:\n")
            f.write("=" * 90 + "\n")
            f.write(_fmt_row("Hyperedge", "Ideal", "Given", "G_err%", "Inference", "I_err%") + "\n")
            f.write("-" * 90 + "\n")
            for _tag, _h, row in all_rows[i]:
                f.write(row + "\n")
        print(f"关联分析表格已保存: {ca_txt_path}")

        ca_json_paths.append(ca_json_path)
        ca_txt_paths.append(ca_txt_path)

    return ca_results, ca_json_paths, ca_txt_paths

def collect_ca_from_base_dir(
    ca_base_dir: str,
    shots_analysis_list=None,
    inference_eps=None,
) -> dict:
    """从 base_dir 下收集子目录中的 correlation.json，合并为多组 ca_results。

    用于一次遍历多组数据的解码。目录结构：ca_base_dir/{shots}_{eps}/correlation.json

    Args:
        ca_base_dir: 如 data/surface_code_rotated_memory_x_d5r5/
        shots_analysis_list: 可选，只收集指定的 shots 组，如 [150000, 1500000, 15000000]
        inference_eps: 可选，只收集匹配的 inference_eps 组

    Returns:
        合并后的 ca_results，shots_analysis_list 包含所有组
    """
    ca_base_dir = os.path.normpath(ca_base_dir)
    if not os.path.isdir(ca_base_dir):
        raise FileNotFoundError(f"目录不存在: {ca_base_dir}")

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
            filter_msg = f"（过滤 shots={shots_analysis_list}, inference_eps={inference_eps}）"
        raise FileNotFoundError(f"目录 {ca_base_dir} 下未找到匹配的 correlation.json {filter_msg}")

    def _sort_key(item):
        subname = item[0]
        parts = subname.split("_")
        try:
            return int(parts[0])
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
    }
    return merged

def load_correlation_analysis(ca_path: str) -> dict:
    """从 .json 文件加载关联分析结果。

    - 文件路径：data/{code}_d{d}r{r}/{shots}_{eps}/correlation.json
    - 目录路径：data/{code}_d{d}r{r}/{shots}_{eps}/ 时自动查找 correlation.json
    """
    ca_path = str(ca_path)
    if os.path.isdir(ca_path):
        json_path = os.path.join(ca_path, "correlation.json")
        if os.path.isfile(json_path):
            ca_path = json_path
        else:
            raise FileNotFoundError(f"目录 {ca_path} 中未找到 correlation.json")
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
    """从文件读取关联分析结果并执行解码实验。

    Args:
        ca_path: 关联分析结果路径，支持：
            - 单文件：correlation.json 路径
            - 单目录：含 correlation.json 的目录
            - 多组目录：如 data/surface_code_rotated_memory_x_d5r5/，遍历子目录收集
        target_logical_errors: 目标逻辑错误次数，默认 200
        batch_size: 每批采样量
        max_shots: 最大采样量上限
        decoder: 解码器，'belief_matching' 或 'bposd'
        shots_analysis_list: 多组目录时，只解码指定的 shots 组
        inference_eps: 多组目录时，只解码匹配的 inference_eps 组
        base_dir: 输出根目录，默认 "data"

    Returns:
        list[str]: 各 shots_analysis 对应的 ler.txt 路径列表
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
    """基于关联分析结果进行解码，比较 LER。

    采样并解码直到收集到 target_logical_errors 次逻辑错误（以 ideal_dem 为判定），
    再用同一批样本评估各 DEM 的 LER。

    可从内存传入 ca_results，或从 ca_path 读取。

    Args:
        ca_results: 关联分析结果 dict（由 run_correlation_analysis 返回）
        ca_path: 或从 .json 文件路径读取
        target_logical_errors: 目标逻辑错误次数，默认 200
        batch_size: 每批采样量
        max_shots: 最大采样量上限，防止极端低 LER 时无限循环
        decoder: 若与 ca_results 中不一致可覆盖
        output_dir: 已弃用，保留兼容；实际使用 base_dir
        base_dir: 输出根目录，默认 "data"，LER 写入 data/{code_task}_d{d}r{r}/{shots_analysis}_{inference_eps}/ler.txt

    Returns:
        list[str]: 各 shots_analysis 对应的 ler.txt 路径列表
    """
    if ca_results is None and ca_path is None:
        raise ValueError("必须提供 ca_results 或 ca_path")
    if ca_results is None:
        ca_results = load_correlation_analysis(ca_path)

    # 支持从 ca_base_dir 合并的多组数据（含 inference_eps_list）
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
    _modes = []
    if any(given_probs_list):
        _modes.append("given")
    if any(infer_probs_list):
        _modes.append("inference")
    CA_mode = tuple(_modes) if _modes else params.get("CA_mode", ("given", "inference"))
    shots_analysis_list = params["shots_analysis_list"]
    ideal_probs = ca_results["ideal_probs"]
    given_probs_list = ca_results["given_probs_list"]
    infer_probs_list = ca_results["infer_probs_list"]
    ideal_set = ca_results["ideal_set"]
    extra_edges = ca_results["extra_edges"]
    has_extra = ca_results["has_extra"]

    if dec not in ("belief_matching", "bposd"):
        raise ValueError(f"decoder 必须是 'belief_matching' 或 'bposd'，当前为 {dec!r}")
    decode_fn = decode_with_belief_matching if dec == "belief_matching" else decode_with_bposd

    circuit, _, _, num_dets = generate_test_circuit(
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

    print(f"\n使用解码器: {dec}，采样直到收集 {target_logical_errors} 次逻辑错误...")
    dets_decode, obs_decode, total_shots, total_logical_errors, _, _ = sample_until_logical_errors(
        circuit, ideal_dem, decode_fn,
        target_logical_errors=target_logical_errors,
        batch_size=batch_size,
        seed=43,
        max_shots=max_shots,
    )
    ler_ideal = total_logical_errors / total_shots
    print(f"  实际采样: {total_shots} shots, 逻辑错误: {total_logical_errors}")
    print("\n" + "=" * 70)
    print("LER 比较 (held-out decode sample)")
    print("=" * 70)
    print(f"  Ideal DEM LER:                {ler_ideal:.6f}")

    ler_paths = []
    for i in range(len(shots_analysis_list)):
        sa = shots_analysis_list[i]
        eps = inference_eps_list[i] if inference_eps_list is not None else params.get("inference_eps", 0)
        out_dir = get_output_dir(code_task, distance, rounds, sa, eps, base_dir)
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "ler.txt")
        ler_paths.append(txt_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("LER Comparison: Ideal vs Given DEM vs Inference DEM\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Code task: {code_task}, d={distance}, r={rounds}, p={p_circuit}\n")
            f.write(f"shots_analysis={sa}, target_logical_errors={target_logical_errors}, total_shots={total_shots}, max_order={params['max_order']}, decoder={dec}\n")
            f.write(f"Ideal 超边: {len(ideal_set)}\n")
            f.write(f"Ideal DEM LER:                {ler_ideal:.6f}\n")
        if "given" in CA_mode:
            given_dem = create_dem_from_analysis(ideal_dem, given_probs_list[i])
            ler_given, _ = decode_fn(given_dem, dets_decode, obs_decode)
        else:
            ler_given = None
        if "inference" in CA_mode:
            infer_set = set(infer_probs_list[i].keys())
            infer_shared = {h: infer_probs_list[i][h] for h in infer_set & ideal_set}
            infer_dem_shared = create_dem_from_analysis(ideal_dem, infer_shared)
            ler_infer_shared, _ = decode_fn(infer_dem_shared, dets_decode, obs_decode)
        else:
            ler_infer_shared = None

        if "given" in CA_mode:
            print(f"  Given DEM LER from {shots_analysis_list[i]} shots:                {ler_given:.6f}  (gap: {ler_ideal - ler_given:+.6f})")
        if "inference" in CA_mode:
            print(f"  Inference DEM LER (共有边) from {shots_analysis_list[i]} shots:   {ler_infer_shared:.6f}  (gap: {ler_ideal - ler_infer_shared:+.6f})")

        if "inference" in CA_mode and has_extra[i]:
            extra_dem = stim.DetectorErrorModel()
            for instr in ideal_dem.flattened():
                extra_dem.append(instr)
            for h in extra_edges[i]:
                p_val = float(infer_probs_list[i][h])
                if p_val <= 0 or p_val >= 1:
                    continue
                targets = (
                    [stim.target_relative_detector_id(d) for d in sorted(h)]
                    if dec == "bposd"
                    else _build_decomposed_targets(h)
                )
                extra_dem.append(stim.DemInstruction("error", args=[p_val], targets=targets))
            infer_dem_full = create_dem_from_analysis(extra_dem, infer_probs_list[i])
            ler_infer_full, _ = decode_fn(infer_dem_full, dets_decode, obs_decode)
            print(f"  Inference DEM LER (含额外边)from {shots_analysis_list[i]} shots: {ler_infer_full:.6f}  (gap: {ler_ideal - ler_infer_full:+.6f})")
        else:
            ler_infer_full = ler_infer_shared if "inference" in CA_mode else None
            if "inference" in CA_mode:
                print("  (Inference 超边与 ideal 一致，无额外边)")

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\ncorrelation analysis with {shots_analysis_list[i]} shots\n")
            f.write(f"Given DEM LER:                {ler_given:.6f}  (gap: {ler_ideal - ler_given:+.6f})\n" if ler_given is not None else "Given DEM LER:                -      \n")
            if "inference" in CA_mode:
                f.write(f"Inference 超边: {len(infer_set)}, 额外边: {len(extra_edges[i])}\n")
                f.write(f"Inference DEM LER (共有边):   {ler_infer_shared:.6f}  (gap: {ler_ideal - ler_infer_shared:+.6f})\n")
                if has_extra[i]:
                    f.write(f"Inference DEM LER (含额外边): {ler_infer_full:.6f}  (gap: {ler_ideal - ler_infer_full:+.6f})\n")
                else:
                    f.write("(Inference 超边与 ideal 一致，无额外边)\n")
            else:
                f.write("Inference 超边: -, 额外边: -      \n")

    return ler_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LER 比较：关联分析 或 从文件解码")
    parser.add_argument("--ca-path", type=str, default=None,
                        help="若提供，则执行 run_decode_from_files；否则执行 run_correlation_analysis")
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
        print(f"\nLER 输出:")
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
        print(f"\n关联分析结果:")
        for p in ca_json_paths:
            print(f"  {p}")
