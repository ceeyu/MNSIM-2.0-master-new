#!/usr/bin/python
# -*-coding:utf-8-*-
import sys
import os
import math
import argparse
import numpy as np
import time
from importlib import import_module
from MNSIM.Interface.maxcut_interface import MaxCutInterface
from MNSIM.Interface.psav_adapter import run_psa_with_mnsim_mapping
from MNSIM.Interface.rram_psa import run_rram_psa
from MNSIM.Mapping_Model.Behavior_mapping import behavior_mapping
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy


class MaxCutTCG(TCG):
    """
    專為 Max Cut 問題設計的 Tile Connection Graph
    """
    def __init__(self, maxcut_structure, SimConfig_path, multiple=None):
        # 將 Max Cut 結構轉換為類似神經網路的格式
        self.maxcut_structure = maxcut_structure
        
        # 建立假的層結構以相容現有的 TCG
        fake_structure = []
        for layer_info in maxcut_structure['layers']:
            # 建立符合 TCG 期望的字典格式
            fake_layer_dict = {
                'type': 'fc',  # 使用 fc 類型，因為 Max Cut 是矩陣向量乘法
                'Infeature': layer_info['input_size'],
                'Outfeature': layer_info['output_size'],
                'Weightbit': 8,  # 8-bit 權重
                'Inputbit': 8,   # 8-bit 輸入
                'outputbit': 8,  # 8-bit 輸出
                'Inputindex': [-1],  # 輸入層索引
                'Outputindex': [1],  # 輸出層索引
                'row_split_num': 1,  # 行分割數
                'weight_bit_split_part': 1,  # 權重位元分割部分
                'Layerindex': 0  # 層索引
            }
            
            # TCG 期望的格式：[[[layer_dict]]] - 三層嵌套列表
            # 每個層是一個包含一個列表的列表，該列表包含一個字典
            fake_layer = [[[fake_layer_dict]]]
            fake_structure.append(fake_layer)
        
        # 初始化父類別
        super().__init__(fake_structure, SimConfig_path, multiple)
        
        # 覆寫特定於 Max Cut 的映射邏輯
        self._customize_for_maxcut()
    
    def _customize_for_maxcut(self):
        """客製化 Max Cut 的映射策略"""
        # Max Cut 主要是矩陣向量乘法，調整資源分配
        for layer_id in range(len(self.net_structure)):
            # 增加 crossbar 利用率（矩陣運算密集）
            if hasattr(self, 'layer_tileinfo'):
                for tile_info in self.layer_tileinfo[layer_id]:
                    # 提高每個 tile 的 PE 利用率
                    tile_info['max_row'] = min(tile_info['max_row'] * 2, self.tile.PE_xbar_list[0][0].xbar_row)
                    tile_info['max_column'] = min(tile_info['max_column'] * 2, self.tile.PE_xbar_list[0][0].xbar_column)


def main():
    home_path = os.getcwd()
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    
    parser = argparse.ArgumentParser(description='MNSIM Max Cut Example')
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
        help="Whether delete the unnecessary files automatically")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
        help="Hardware description file location & name")
    parser.add_argument("-Graph", "--graph_file", required=True,
        help="Graph file location & name (txt/csv/graphml format)")
    parser.add_argument("-Alg", "--algorithm", default='goemans_williamson',
        choices=['goemans_williamson', 'greedy', 'semidefinite', 'psav_psa', 'rram_psa'],
        help="Max Cut algorithm choice")
    parser.add_argument("-Iter", "--iterations", type=int, default=100,
        help="Number of iterations for the algorithm")
    parser.add_argument("--spin_vector", type=str, default=None,
        help="Optional spin vector in the form '1,-1,1' to be used as test input")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
        help="Disable hardware modeling")
    parser.add_argument("-D", "--device", default=0,
        help="Determine hardware device (CPU or GPU-id) for simulation")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
        help="Disable module simulation results output")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
        help="Disable layer-wise simulation results output")
    # pSAv 相關參數（僅支援 pSA）
    parser.add_argument("--psav_cycles", type=int, default=200, help="pSAv cycles")
    parser.add_argument("--psav_trials", type=int, default=50, help="pSAv trials")
    parser.add_argument("--psav_tau", type=int, default=1, help="pSAv tau")
    parser.add_argument("--psav_param", type=int, default=2, help="pSAv param (2 推薦)")
    parser.add_argument("--psav_thread", type=int, default=32, help="pSAv threads per block (y 維度)")
    parser.add_argument("--psav_gpu", type=int, default=0, help="pSAv GPU device id")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MNSIM Max Cut 模擬器")
    print("=" * 60)
    print("硬體描述檔位置:", args.hardware_description)
    print("圖檔案位置:", args.graph_file)
    print("求解算法:", args.algorithm)
    print("迭代次數:", args.iterations)
    print("是否執行硬體模擬:", not args.disable_hardware_modeling)
    print("=" * 60)
    
    # 檢查檔案是否存在
    if not os.path.exists(args.graph_file):
        print(f"錯誤: 找不到圖檔案 {args.graph_file}")
        return
    
    if not os.path.exists(args.hardware_description):
        print(f"錯誤: 找不到硬體設定檔 {args.hardware_description}")
        return
    
    mapping_start_time = time.time()
    
    # 建立 Max Cut 介面
    maxcut_interface = MaxCutInterface(
        graph_file=args.graph_file,
        SimConfig_path=args.hardware_description,
        algorithm=args.algorithm,
        device=args.device
    )
    
    # 取得問題結構
    structure = maxcut_interface.get_structure()
    
    # 建立客製化的 TCG 映射（如果啟用硬體建模）
    TCG_mapping = None
    if not args.disable_hardware_modeling:
        try:
            TCG_mapping = MaxCutTCG(structure, args.hardware_description)
        except Exception as e:
            print(f"警告: 硬體映射失敗，跳過硬體建模: {e}")
            args.disable_hardware_modeling = True
    
    mapping_end_time = time.time()
    
    # 硬體模擬
    if not args.disable_hardware_modeling and TCG_mapping is not None:
        hardware_modeling_start_time = time.time()
        
        print("\n" + "=" * 50)
        print("開始硬體效能模擬...")
        print("=" * 50)
        
        # 延遲模擬
        latency_model = Model_latency(
            NetStruct=structure, 
            SimConfig_path=args.hardware_description, 
            TCG_mapping=TCG_mapping
        )
        latency_model.calculate_model_latency(mode=1)
        
        print("\n========================延遲模擬結果=================================")
        latency_model.model_latency_output(
            not args.disable_module_output, 
            not args.disable_layer_output
        )
        
        # 面積模擬
        area_model = Model_area(
            NetStruct=structure, 
            SimConfig_path=args.hardware_description, 
            TCG_mapping=TCG_mapping
        )
        
        print("\n========================面積模擬結果=================================")
        area_model.model_area_output(
            not args.disable_module_output, 
            not args.disable_layer_output
        )
        
        # 功率模擬
        power_model = Model_inference_power(
            NetStruct=structure, 
            SimConfig_path=args.hardware_description,
            TCG_mapping=TCG_mapping
        )
        
        print("\n========================功率模擬結果=================================")
        power_model.model_power_output(
            not args.disable_module_output, 
            not args.disable_layer_output
        )
        
        # 能量模擬
        energy_model = Model_energy(
            NetStruct=structure, 
            SimConfig_path=args.hardware_description,
            TCG_mapping=TCG_mapping,
            model_latency=latency_model, 
            model_power=power_model
        )
        
        print("\n========================能量模擬結果=================================")
        energy_model.model_energy_output(
            not args.disable_module_output, 
            not args.disable_layer_output
        )
        
        hardware_modeling_end_time = time.time()
    
    # Max Cut 求解
    print("\n" + "=" * 50)
    print("開始 Max Cut 求解...")
    print("=" * 50)
    
    maxcut_start_time = time.time()
    
    # 準備硬體映射資料
    crossbar_partitions = maxcut_interface.partition_matrix_to_crossbars()
    # 若提供了使用者的 spin 向量，直接使用該向量（量化後）作為測試向量
    if args.spin_vector is not None:
        try:
            raw_vals = [float(x) for x in args.spin_vector.replace("[", "").replace("]", "").replace(" ", "").split(',') if x != '']
            user_vec = np.array(raw_vals, dtype=float)
            if user_vec.shape[0] != maxcut_interface.num_nodes:
                print(f"警告: 提供的 spin 向量長度({user_vec.shape[0]}) 與圖節點數({maxcut_interface.num_nodes})不一致，將嘗試截斷或填零")
                vec = np.zeros(maxcut_interface.num_nodes, dtype=float)
                m = min(maxcut_interface.num_nodes, user_vec.shape[0])
                vec[:m] = user_vec[:m]
                user_vec = vec
            quantized_vec = maxcut_interface._quantize_vector(user_vec)
            test_vectors = [quantized_vec]
            # 另外計算並回報此 spin 向量直接對應的 cut value（以 0/1 分割方式）
            partition_from_spin = (user_vec > 0).astype(int)
            user_cut_value = maxcut_interface.evaluate_cut_value(partition_from_spin)
            print(f"使用者提供 spin 向量的 cut 值: {user_cut_value:.2f}")
        except Exception as e:
            print(f"解析 --spin_vector 發生錯誤: {e}")
            print("改為使用預設隨機測試向量")
            test_vectors = maxcut_interface.create_test_vectors(args.iterations)
    else:
        test_vectors = maxcut_interface.create_test_vectors(args.iterations)
    
    print(f"矩陣分割到 {len(crossbar_partitions)} 個 crossbar")
    print(f"產生 {len(test_vectors)} 個測試向量")
    
    # 執行硬體加速的 Max Cut 求解
    if args.algorithm == 'psav_psa':
        # 直接以 MNSIM 的 RRAM 權重映射生成 J/h，呼叫 GPU pSA 核心
        psav_res = run_psa_with_mnsim_mapping(
            graph_file=args.graph_file,
            SimConfig_path=args.hardware_description,
            gpu=int(args.psav_gpu),
            cycles=int(args.psav_cycles),
            trials=int(args.psav_trials),
            tau=int(args.psav_tau),
            thread=int(args.psav_thread),
            param=int(args.psav_param)
        )
        print("pSA cut 值清單:", psav_res['cut_list'])
        print("pSA 時間(ms):", psav_res['time_list'])
        best_partition, best_value = (None, float('nan'))
    elif args.algorithm == 'rram_psa':
        # 使用 RRAM crossbar 的 pSA 實作
        rram_res = run_rram_psa(
            graph_file=args.graph_file,
            SimConfig_path=args.hardware_description,
            trials=args.psav_trials,
            cycles=args.psav_cycles,
            tau=args.psav_tau,
            param_type=args.psav_param
        )
        best_partition, best_value = (None, rram_res['cut_max'])
    else:
        best_partition, best_value = maxcut_interface.solve_maxcut_hardware(
            crossbar_partitions, test_vectors
        )
    
    maxcut_end_time = time.time()
    
    # 結果輸出
    print("\n" + "=" * 50)
    print("Max Cut 求解結果")
    print("=" * 50)
    print(f"最佳分割值: {best_value:.2f}")
    print(f"分割結果: {best_partition}")
    
    # 分析分割品質
    if best_partition is not None:
        partition_0 = np.sum(best_partition == 0)
        partition_1 = np.sum(best_partition == 1)
        print(f"分割 0 的節點數: {partition_0}")
        print(f"分割 1 的節點數: {partition_1}")
        if max(partition_0, partition_1) > 0:
            print(f"分割平衡度: {min(partition_0, partition_1) / max(partition_0, partition_1):.3f}")
        else:
            print("分割平衡度: N/A")
    else:
        print("分割結果: 無有效分割")
    
    # 時間統計
    mapping_time = mapping_end_time - mapping_start_time
    maxcut_time = maxcut_end_time - maxcut_start_time
    
    print("\n" + "=" * 50)
    print("效能統計")
    print("=" * 50)
    print(f"映射時間: {mapping_time:.3f} 秒")
    
    if not args.disable_hardware_modeling:
        hardware_modeling_time = hardware_modeling_end_time - hardware_modeling_start_time
        print(f"硬體模擬時間: {hardware_modeling_time:.3f} 秒")
        total_time = mapping_time + hardware_modeling_time + maxcut_time
    else:
        hardware_modeling_time = 0
        total_time = mapping_time + maxcut_time
    
    print(f"Max Cut 求解時間: {maxcut_time:.3f} 秒")
    print(f"總模擬時間: {total_time:.3f} 秒")
    
    # 儲存結果
    result_file = f"maxcut_result_{os.path.basename(args.graph_file)}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Graph file: {args.graph_file}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Best cut value: {best_value}\n")
        if best_partition is not None:
            f.write(f"Partition: {' '.join(map(str, best_partition))}\n")
        else:
            f.write(f"Partition: None\n")
        f.write(f"Total time: {total_time:.3f} seconds\n")
    
    print(f"\n結果已儲存至: {result_file}")


if __name__ == '__main__':
    main()
