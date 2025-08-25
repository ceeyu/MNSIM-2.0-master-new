#!/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import configparser
import networkx as nx
from typing import Dict, List, Tuple, Optional
import os

class MaxCutInterface:
    """
    Max Cut 問題的介面層，替代原本的神經網路介面
    處理圖論問題的加權矩陣輸入與 RRAM 硬體的對接
    """
    
    def __init__(self, graph_file: str, SimConfig_path: str, algorithm: str = 'goemans_williamson', device: str = 'cpu'):
        """
        初始化 Max Cut 介面
        
        Args:
            graph_file: 圖檔案路徑 (支援 .txt, .csv, .graphml 等格式)
            SimConfig_path: 硬體設定檔路徑
            algorithm: 求解算法 ('goemans_williamson', 'semidefinite', 'greedy')
            device: 計算裝置
        """
        self.graph_file = graph_file
        self.SimConfig_path = SimConfig_path
        self.algorithm = algorithm
        self.device = device
        
        # 讀取硬體設定
        self.config = configparser.ConfigParser()
        self.config.read(SimConfig_path, encoding='UTF-8')
        
        # 載入圖資料
        self.graph = self._load_graph()
        self.adjacency_matrix = self._get_adjacency_matrix()
        self.num_nodes = len(self.graph.nodes())
        
        # RRAM 硬體參數
        self.crossbar_size = list(map(int, self.config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.device_levels = int(self.config.get('Device level', 'Device_Level'))
        self.device_resistance = list(map(float, self.config.get('Device level', 'Device_Resistance').split(',')))
        
        # 量化參數
        self.weight_bits = 8  # 權重量化位數
        self.voltage_levels = int(self.config.get('Device level', 'Read_Level'))
        
        print(f"載入圖檔: {graph_file}")
        print(f"節點數量: {self.num_nodes}")
        print(f"邊數量: {len(self.graph.edges())}")
        print(f"RRAM Crossbar 尺寸: {self.crossbar_size}")
        
    def _load_graph(self) -> nx.Graph:
        """載入圖檔案"""
        if self.graph_file.endswith('.txt') or self.graph_file.endswith('.csv'):
            # 假設格式: node1 node2 weight
            G = nx.Graph()
            with open(self.graph_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # 跳過空行和註解行
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        weight = float(parts[2]) if len(parts) > 2 else 1.0
                        G.add_edge(u, v, weight=weight)
            return G
        elif self.graph_file.endswith('.graphml'):
            return nx.read_graphml(self.graph_file)
        else:
            raise ValueError(f"不支援的圖檔案格式: {self.graph_file}")
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        """取得加權鄰接矩陣"""
        return nx.adjacency_matrix(self.graph, weight='weight').toarray().astype(np.float32)
    
    def get_structure(self) -> Dict:
        """
        取得 Max Cut 問題的結構資訊，對應原本的神經網路結構
        """
        structure = {
            'type': 'maxcut',
            'nodes': self.num_nodes,
            'edges': len(self.graph.edges()),
            'matrix_shape': self.adjacency_matrix.shape,
            'algorithm': self.algorithm,
            'layers': [
                {
                    'type': 'fc',  # 使用 fc 類型以相容 TCG
                    'input_size': self.num_nodes,
                    'output_size': self.num_nodes,
                    'weight_matrix': self.adjacency_matrix,
                    'operation': 'linear'
                }
            ]
        }
        return structure
    
    def quantize_weights(self) -> np.ndarray:
        """
        將加權矩陣量化到 RRAM 電阻值
        """
        # 正規化權重到 [0, 1]
        W = self.adjacency_matrix
        W_min, W_max = W.min(), W.max()
        if W_max > W_min:
            W_norm = (W - W_min) / (W_max - W_min)
        else:
            W_norm = W
        
        # 量化到設備電阻等級
        quantized_levels = np.round(W_norm * (self.device_levels - 1)).astype(int)
        
        # 映射到實際電阻值
        resistance_values = np.zeros_like(quantized_levels, dtype=float)
        for i in range(self.device_levels):
            mask = (quantized_levels == i)
            resistance_values[mask] = self.device_resistance[i]
        
        return resistance_values
    
    def partition_matrix_to_crossbars(self) -> List[Dict]:
        """
        將大矩陣分割到多個 crossbar 中
        """
        matrix = self.quantize_weights()
        crossbar_partitions = []
        
        rows_per_xbar = self.crossbar_size[0]
        cols_per_xbar = self.crossbar_size[1]
        
        for i in range(0, matrix.shape[0], rows_per_xbar):
            for j in range(0, matrix.shape[1], cols_per_xbar):
                # 取得子矩陣
                end_i = min(i + rows_per_xbar, matrix.shape[0])
                end_j = min(j + cols_per_xbar, matrix.shape[1])
                
                submatrix = matrix[i:end_i, j:end_j]
                
                # 如果需要，填充到 crossbar 大小
                if submatrix.shape[0] < rows_per_xbar or submatrix.shape[1] < cols_per_xbar:
                    padded = np.zeros((rows_per_xbar, cols_per_xbar))
                    padded[:submatrix.shape[0], :submatrix.shape[1]] = submatrix
                    submatrix = padded
                
                partition_info = {
                    'position': (i, j),
                    'actual_size': (end_i - i, end_j - j),
                    'matrix': submatrix,
                    'resistance_matrix': submatrix  # 已經是電阻值
                }
                crossbar_partitions.append(partition_info)
        
        return crossbar_partitions
    
    def create_test_vectors(self, num_iterations: int = 100) -> List[np.ndarray]:
        """
        建立測試向量序列，用於 Max Cut 迭代算法
        """
        test_vectors = []
        
        if self.algorithm == 'goemans_williamson':
            # Goemans-Williamson 算法的隨機向量
            for _ in range(num_iterations):
                # 產生隨機單位向量
                vec = np.random.randn(self.num_nodes)
                vec = vec / np.linalg.norm(vec)
                # 量化到電壓等級
                vec_quantized = self._quantize_vector(vec)
                test_vectors.append(vec_quantized)
                
        elif self.algorithm == 'greedy':
            # 貪婪算法的向量序列
            for iteration in range(num_iterations):
                vec = np.random.choice([-1, 1], size=self.num_nodes)
                vec_quantized = self._quantize_vector(vec)
                test_vectors.append(vec_quantized)
        
        return test_vectors
    
    def _quantize_vector(self, vector: np.ndarray) -> np.ndarray:
        """將輸入向量量化到電壓等級"""
        # 正規化到 [-1, 1]
        v_norm = np.tanh(vector)
        # 映射到電壓等級 [0, voltage_levels-1]
        v_quantized = np.round((v_norm + 1) * (self.voltage_levels - 1) / 2).astype(int)
        return v_quantized

    def export_to_psav_file(self, out_path: str, best_known: int = 0,
                             edge_value: int = 0, edge_type: int = 0) -> str:
        """
        以 GPU-pSAv 所需的格式輸出圖檔：
        第一行: 頂點數
        第二行: edge value (保留欄位，可填 0)
        第三行: edge type (保留欄位，可填 0)
        第四行: best-known cut 值（未知則 0）
        其餘行: i j w （1-based 節點編號，整數權重）
        回傳輸出檔路徑。
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(f"{self.num_nodes}\n")
            f.write(f"{edge_value}\n")
            f.write(f"{edge_type}\n")
            f.write(f"{best_known}\n")
            # 輸出上三角（避免重複），1-based 編號
            A = self.adjacency_matrix
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    w = A[i, j]
                    if abs(w) > 0:
                        f.write(f"{i+1} {j+1} {int(round(w))}\n")
        return out_path
    
    def evaluate_cut_value(self, partition: np.ndarray) -> float:
        """評估分割的目標函數值"""
        cut_value = 0.0
        for u, v, data in self.graph.edges(data=True):
            if partition[u] != partition[v]:  # 跨分割的邊
                cut_value += data.get('weight', 1.0)
        return cut_value
    
    def solve_maxcut_hardware(self, crossbar_partitions: List[Dict], 
                            test_vectors: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        使用硬體模擬求解 Max Cut
        這裡會與 RRAM crossbar 硬體模型整合
        """
        best_partition = None
        best_value = -float('inf')
        
        for test_vec in test_vectors:
            # 在每個 crossbar 分割上執行 MVM
            result_parts = []
            
            for partition_info in crossbar_partitions:
                pos_i, pos_j = partition_info['position']
                actual_rows, actual_cols = partition_info['actual_size']
                
                # 取得對應的輸入向量片段
                vec_segment = test_vec[pos_j:pos_j + actual_cols]
                if len(vec_segment) < self.crossbar_size[1]:
                    # 填充向量
                    vec_padded = np.zeros(self.crossbar_size[1])
                    vec_padded[:len(vec_segment)] = vec_segment
                    vec_segment = vec_padded
                
                # 執行矩陣向量乘法 (這裡會調用 RRAM crossbar 模擬)
                resistance_matrix = partition_info['resistance_matrix']
                # 簡化的 MVM 計算：以電導 G=1/R 模擬，對於補零位置(R=0)視為 G=0
                conductance_matrix = np.where(resistance_matrix > 0, 1.0 / resistance_matrix, 0.0)
                result_segment = np.dot(conductance_matrix, vec_segment)
                result_parts.append(result_segment[:actual_rows])
            
            # 組合結果
            full_result = np.concatenate(result_parts)
            
            # 根據結果決定分割：若輸入為 spin 向量 {-1,1}，以 0 作為門檻；否則用中位數
            if np.all(np.isin(test_vec, [0, 1, -1])):
                threshold = 0.0
            else:
                threshold = float(np.median(full_result))
            partition = (full_result > threshold).astype(int)
            
            # 評估分割品質
            cut_value = self.evaluate_cut_value(partition)
            
            if cut_value > best_value:
                best_value = cut_value
                best_partition = partition
        
        return best_partition, best_value

def create_sample_graph(num_nodes: int = 10, edge_prob: float = 0.3, 
                       weight_range: Tuple[float, float] = (1.0, 10.0)) -> str:
    """建立範例圖檔案"""
    filename = f"sample_graph_{num_nodes}nodes.txt"
    
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    with open(filename, 'w') as f:
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            f.write(f"{u} {v} {weight:.2f}\n")
    
    print(f"建立範例圖檔: {filename}")
    return filename

if __name__ == "__main__":
    # 測試範例
    sample_file = create_sample_graph(20, 0.4, (1.0, 5.0))
    
    # 假設 SimConfig.ini 在上層目錄
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    
    maxcut_interface = MaxCutInterface(sample_file, config_path, 'goemans_williamson')
    structure = maxcut_interface.get_structure()
    partitions = maxcut_interface.partition_matrix_to_crossbars()
    test_vectors = maxcut_interface.create_test_vectors(50)
    
    print(f"矩陣分割數量: {len(partitions)}")
    print(f"測試向量數量: {len(test_vectors)}")
