#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import sys
import numpy as np

from .maxcut_interface import MaxCutInterface


def _to_int8_J_from_rram(adj_matrix: np.ndarray,
                          resistance_matrix: np.ndarray) -> np.ndarray:
    """
    將 RRAM 電阻矩陣轉為 pSAv 需要的 J (int8) 矩陣：
    - 以電導 G = 1/R 表示權重強度
    - 正規化到 [0,1]，再映射到 [0,127]
    - MaxCut 的 Ising 轉換使用 J = -G_scaled
    - 對角設為 0，並確保對稱
    """
    conductance = np.where(resistance_matrix > 0, 1.0 / resistance_matrix, 0.0)
    max_g = np.max(conductance)
    if max_g <= 0:
        norm = np.zeros_like(conductance)
    else:
        norm = conductance / max_g
    J_float = -np.round(norm * 127.0)
    J = J_float.astype(np.int8)
    # 清理對角與非對稱
    n = adj_matrix.shape[0]
    for i in range(n):
        J[i, i] = 0
        for j in range(i+1, n):
            val = J[i, j] if J[i, j] != 0 else J[j, i]
            J[i, j] = val
            J[j, i] = val
    return J


def run_psa_with_mnsim_mapping(graph_file: str,
                               SimConfig_path: str,
                               gpu: int = 0,
                               cycles: int = 200,
                               trials: int = 50,
                               tau: int = 1,
                               thread: int = 32,
                               param: int = 2):
    """
    使用 MNSIM 的 RRAM 權重映射生成 pSAv 所需的 J/h，並直接調用 pSAv 的 GPU 核心執行 pSA。
    """
    # 匯入 pSAv 核心（動態加入路徑）
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    psav_root = os.path.join(root, 'GPU-pSAv-main')
    if psav_root not in sys.path:
        sys.path.append(psav_root)

    # 延後匯入避免全域初始化
    from gpu_MAXCUT import initialize_cuda, load_gpu_code, set_annealing_parameters, run_trials, cleanup_cuda

    # 建立介面並做 RRAM 權重量化
    mc = MaxCutInterface(graph_file, SimConfig_path)
    W = mc.adjacency_matrix
    R = mc.quantize_weights()  # 量化後的電阻矩陣
    J = _to_int8_J_from_rram(W, R)
    n = J.shape[0]
    h = np.zeros((n, 1), dtype=np.int8)

    # 初始化 GPU 與 kernel
    class _Args:
        pass
    args = _Args()
    args.gpu = gpu
    args.thread = thread
    args.cycle = cycles
    args.trial = trials
    args.tau = tau
    args.param = param
    args.unique = 1
    args.config = 2  # 固定 pSA

    device = initialize_cuda(args)
    gpu_code = load_gpu_code(args.config)

    # pSAv 的參數計算
    # 注意：pSAv 的 set_annealing_parameters 期望 h_vector/J_matrix 為 float32/int32
    J32 = J.astype(np.int32)
    h32 = h.astype(np.int8)  # 他們的 run_trials 內轉成 int8 後傳 GPU
    min_cycle, trial, tau_out, Mshot, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid = \
        set_annealing_parameters(n, args, h32, J32)

    try:
        # run_trials 需要 stall_prop, mean_range，但 pSA 不用，傳預設
        cut_list, time_list, energy_list = run_trials(
            0.5, 4, 'mnsim_psa', args.config, n,
            min_cycle, trial, tau_out, Mshot,
            gpu_code, h32, None, J32, sigma_vector, nrnd_vector,
            I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid
        )
    finally:
        cleanup_cuda()

    return {
        'cut_list': cut_list,
        'time_list': time_list,
        'energy_list': energy_list,
        'n': n
    }


