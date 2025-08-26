#!/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import time
import math
from typing import Tuple, List, Dict
from .maxcut_interface import MaxCutInterface


class RRAMpSA:
    """
    基於 RRAM crossbar 的 pSA Max Cut 求解器
    用 RRAM MVM 替代 GPU 的矩陣運算，CPU 執行退火邏輯
    """
    
    def __init__(self, maxcut_interface: MaxCutInterface):
        self.mc = maxcut_interface
        self.n = maxcut_interface.num_nodes
        self.G_matrix = maxcut_interface.adjacency_matrix  # 原始權重矩陣
        
        # 建立 RRAM crossbar 分割
        self.crossbar_partitions = maxcut_interface.partition_matrix_to_crossbars()
        
        # Ising 轉換：J = -G (Max Cut → Min Ising Energy)
        self.J_matrix = -self.G_matrix.astype(np.float32)
        self.h_vector = np.zeros(self.n, dtype=np.float32)  # Max Cut 無外場
        
        print(f"RRAM pSA 初始化完成：{self.n} 節點，{len(self.crossbar_partitions)} 個 crossbar")
    
    def rram_matrix_vector_multiply(self, vector: np.ndarray) -> np.ndarray:
        """
        使用 RRAM crossbar 執行矩陣向量乘法
        模擬 MNSIM crossbar 的 MVM 運算
        """
        result = np.zeros(self.n, dtype=np.float32)
        
        for partition_info in self.crossbar_partitions:
            pos_i, pos_j = partition_info['position']
            actual_rows, actual_cols = partition_info['actual_size']
            resistance_matrix = partition_info['resistance_matrix']
            
            # 取得對應的向量片段
            vec_segment = vector[pos_j:pos_j + actual_cols]
            if len(vec_segment) < resistance_matrix.shape[1]:
                vec_padded = np.zeros(resistance_matrix.shape[1])
                vec_padded[:len(vec_segment)] = vec_segment
                vec_segment = vec_padded
            
            # RRAM MVM：電導矩陣 × 電壓向量 = 電流向量
            # 避免除零警告：只有當電阻 > 小閾值時才計算電導
            min_resistance = 1e-10  # 最小電阻閾值
            conductance_matrix = np.where(resistance_matrix > min_resistance, 
                                        1.0 / resistance_matrix, 0.0)
            current_segment = np.dot(conductance_matrix, vec_segment)
            
            # 累加到結果向量
            result[pos_i:pos_i + actual_rows] += current_segment[:actual_rows]
        
        return result
    
    def calculate_local_field_rram(self, spin_vector: np.ndarray) -> np.ndarray:
        """
        用 RRAM crossbar 計算局部場 D_i = h_i + Σ_j J_ij * s_j
        """
        # 將 spin 向量轉為電壓（映射 {-1,1} → 電壓等級）
        voltage_vector = self._spin_to_voltage(spin_vector)
        
        # RRAM MVM 計算 J × s
        j_dot_s = self.rram_matrix_vector_multiply(voltage_vector)
        
        # 轉回實際值並加上外場
        local_field = self._voltage_to_field(j_dot_s) + self.h_vector
        
        return local_field
    
    def _spin_to_voltage(self, spin_vector: np.ndarray) -> np.ndarray:
        """將 spin {-1,1} 轉為 RRAM 輸入電壓"""
        # 簡化：{-1,1} → {0,1} 電壓映射
        return (spin_vector + 1) / 2
    
    def _voltage_to_field(self, voltage_output: np.ndarray) -> np.ndarray:
        """將 RRAM 輸出電流轉回局部場值"""
        # 依據 crossbar 的電導範圍做逆映射
        # 這裡需要根據實際的電阻量化範圍調整
        max_conductance = 1.0 / min(self.mc.device_resistance)
        field_scale = max_conductance * np.max(np.abs(self.J_matrix))
        return voltage_output * field_scale
    
    def calculate_cut_value_rram(self, spin_vector: np.ndarray) -> float:
        """
        用 RRAM 輔助計算 cut 值
        cut = Σ_{i<j} w_{ij} (1 - s_i s_j) / 2
        """
        # 方法1：直接用 CPU 計算（參考 gpu_MAXCUT.py 的 cut_calculate）
        spin_reshaped = spin_vector.reshape(-1)
        upper_triangle = np.triu_indices(self.n, k=1)
        cut_val = np.sum(self.G_matrix[upper_triangle] * 
                        (1 - np.outer(spin_reshaped, spin_reshaped)[upper_triangle]))
        return cut_val / 2
    
    def psa_annealing_step(self, spin_vector: np.ndarray, I0: float, 
                          noise_vector: np.ndarray) -> np.ndarray:
        """
        單步 pSA 退火更新（參考 psa_annealing_kernel.cu 的邏輯）
        """
        # 用 RRAM 計算局部場
        local_field = self.calculate_local_field_rram(spin_vector)
        
        # pSA 更新規則
        Itanh = np.tanh(I0 * local_field) + noise_vector
        new_spin = np.where(Itanh > 0, 1, -1)
        
        return new_spin.astype(np.int32)
    
    def set_annealing_parameters(self, cycles: int = 200, tau: int = 1, 
                               param_type: int = 2) -> Tuple:
        """
        設定退火參數（參考 gpu_MAXCUT.py 的 set_annealing_parameters）
        """
        # 統計 J 矩陣的特性
        mean_each = []
        std_each = []
        for j in range(self.n):
            mean_each.append((self.n - 1) * np.mean(self.J_matrix[j]))
            std_each.append(np.sqrt((self.n - 1) * 
                           np.var(np.concatenate([self.J_matrix[j], -self.J_matrix[j]]))))
        
        sigma = np.mean(std_each)
        
        # 設定溫度範圍
        if param_type == 1:
            I0_min = np.max(std_each) * 0.01 + np.min(np.abs(mean_each))
            I0_max = np.max(std_each) * 2 + np.min(np.abs(mean_each))
        else:  # param_type == 2
            I0_min = 0.1 / sigma if sigma > 0 else 0.1
            I0_max = 10 / sigma if sigma > 0 else 10
        
        # 計算退火排程
        beta = (I0_min / I0_max) ** (tau / (cycles - 1)) if cycles > 1 else 0.5
        
        return I0_min, I0_max, beta, tau, cycles
    
    def run_psa_trials(self, trials: int = 50, cycles: int = 200, 
                      tau: int = 1, param_type: int = 2) -> Dict:
        """
        執行多次 pSA trials（參考 gpu_MAXCUT.py 的 run_trials 結構）
        """
        I0_min, I0_max, beta, tau, cycles = self.set_annealing_parameters(
            cycles, tau, param_type)
        
        print(f"退火參數：I0_min={I0_min:.3f}, I0_max={I0_max:.3f}, beta={beta:.3f}")
        
        cut_list = []
        time_list = []
        best_cut = -1
        best_partition = None
        
        for trial in range(trials):
            print(f"Trial {trial + 1}/{trials}")
            
            # 隨機初始化 spin
            spin_vector = np.random.choice([-1, 1], size=self.n).astype(np.int32)
            
            trial_start = time.time()
            
            # pSA 退火主循環
            I0 = I0_min
            while I0 <= I0_max:
                for _ in range(tau):
                    # 產生雜訊
                    noise = np.random.uniform(-1, 1, size=self.n)
                    
                    # 執行一步退火更新
                    spin_vector = self.psa_annealing_step(spin_vector, I0, noise)
                
                I0 /= beta  # 降溫
            
            trial_time = (time.time() - trial_start) * 1000  # 轉為 ms
            
            # 計算最終 cut 值
            final_cut = self.calculate_cut_value_rram(spin_vector)
            
            cut_list.append(int(final_cut))
            time_list.append(trial_time)
            
            # 保存最佳結果
            if final_cut > best_cut:
                best_cut = final_cut
                # 將 spin 向量轉換為分割：spin = 1 → partition = 1, spin = -1 → partition = 0
                best_partition = np.where(spin_vector > 0, 1, 0)
            
            print(f"  Cut value: {final_cut:.0f}, Time: {trial_time:.2f} ms")
        
        # 統計結果
        results = {
            'cut_list': cut_list,
            'time_list': time_list,
            'cut_avg': np.mean(cut_list),
            'cut_max': np.max(cut_list),
            'cut_min': np.min(cut_list),
            'cut_std': np.std(cut_list),
            'time_avg': np.mean(time_list),
            'trials': trials,
            'n_nodes': self.n,
            'best_partition': best_partition,
            'best_cut_value': best_cut
        }
        
        return results
    
    def print_results(self, results: Dict):
        """輸出結果統計"""
        print("\n" + "="*50)
        print("RRAM pSA 結果統計")
        print("="*50)
        print(f"節點數: {results['n_nodes']}")
        print(f"試驗次數: {results['trials']}")
        print(f"平均 cut 值: {results['cut_avg']:.2f}")
        print(f"最大 cut 值: {results['cut_max']}")
        print(f"最小 cut 值: {results['cut_min']}")
        print(f"cut 值標準差: {results['cut_std']:.2f}")
        print(f"平均時間: {results['time_avg']:.2f} ms")
        print(f"Cut 值列表: {results['cut_list']}")


def run_rram_psa(graph_file: str, SimConfig_path: str, 
                trials: int = 50, cycles: int = 200, 
                tau: int = 1, param_type: int = 2) -> Dict:
    """
    執行基於 RRAM 的 pSA Max Cut 求解
    
    Args:
        graph_file: 圖檔案路徑
        SimConfig_path: MNSIM 硬體設定檔
        trials: 試驗次數
        cycles: 每次試驗的退火週期數
        tau: 每個溫度點的更新次數
        param_type: 參數類型 (1 或 2)
    
    Returns:
        結果字典包含 cut 值、時間等統計
    """
    # 建立 MaxCut 介面
    mc_interface = MaxCutInterface(graph_file, SimConfig_path)
    
    # 建立 RRAM pSA 求解器
    rram_psa = RRAMpSA(mc_interface)
    
    # 執行 pSA trials
    results = rram_psa.run_psa_trials(trials, cycles, tau, param_type)
    
    # 輸出結果
    rram_psa.print_results(results)
    
    return results


if __name__ == "__main__":
    # 測試範例
    import os
    
    # 測試檔案路徑
    graph_file = "test_graphs/defined_3nodes.txt"
    config_path = "SimConfig.ini"
    
    if os.path.exists(graph_file) and os.path.exists(config_path):
        results = run_rram_psa(graph_file, config_path, trials=10, cycles=100)
    else:
        print("請確認測試檔案存在")
