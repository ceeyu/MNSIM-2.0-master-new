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
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy


class MaxCutTCG:
    """
    專為 Max Cut 問題設計的簡化 Tile Connection Graph
    跳過複雜的硬體建模，提供基本相容性
    """
    def __init__(self, maxcut_structure, SimConfig_path, multiple=None):
        # 保存結構資訊以供後續使用
        self.maxcut_structure = maxcut_structure
        self.SimConfig_path = SimConfig_path
        
        # 基本屬性以相容硬體模擬模組
        self.layer_num = len(maxcut_structure['layers'])
        self.layer_tileinfo = []
        
        # 為每個層建立基本的 tile 資訊
        for i, layer_info in enumerate(maxcut_structure['layers']):
            # Tile 資訊
            tile_info = {
                'type': 'fc',
                'startid': i,
                'mx': 1,  # PE 數量 x 軸
                'my': 1,  # PE 數量 y 軸
                'max_group': 1,
                'max_row': layer_info['input_size'],
                'max_column': layer_info['output_size'],
                'Inputindex': [-1] if i == 0 else [i-1],
                'Outputindex': [i+1] if i < self.layer_num-1 else [],
                'is_branchin': -1,
                'is_branchout': -1
            }
            self.layer_tileinfo.append([tile_info])
            
            # 網路結構 (硬體模擬模組需要的格式)
            layer_dict = {
                'type': 'fc',
                'Infeature': layer_info['input_size'],
                'Outfeature': layer_info['output_size'],
                'Weightbit': 8,
                'Inputbit': 8,
                'outputbit': 8,
                'Inputsize': [1, layer_info['input_size']],  # [batch, features]
                'Outputsize': [1, layer_info['output_size']],
                'Inputindex': [-1] if i == 0 else [i-1],
                'Outputindex': [i+1] if i < self.layer_num-1 else [],
                'Layerindex': i
            }
            
            # 硬體模擬模組期望的格式：NetStruct[layer_id][0][0] = layer_dict
            # 確保 [0][0] 訪問得到字典，而不是列表
            self.NetStruct.append([[layer_dict]])
        
        print(f"MaxCut TCG 初始化完成：{self.layer_num} 層")
    
    def mapping_net(self):
        """提供硬體模擬模組期望的網路映射方法"""
        pass  # Max Cut 的映射已經在初始化中完成
    
    def calculate_transfer_distance(self):
        """計算傳輸距離 - 硬體建模需要"""
        # 為簡化，假設所有傳輸距離為 1
        pass
    
    def calculate_output_stationary(self):
        """計算輸出固定策略 - 硬體建模需要"""
        pass
    
    def calculate_weight_stationary(self):
        """計算權重固定策略 - 硬體建模需要"""
        pass
    
    def calculate_input_stationary(self):
        """計算輸入固定策略 - 硬體建模需要"""
        pass
    
    def __getattr__(self, name):
        """提供基本屬性以相容硬體模擬模組"""
        if name == 'net_structure':
            return self.maxcut_structure
        elif name == 'tile_total_num':
            return 1  # 簡化為單一 tile
        elif name == 'mapping_result':
            # 返回簡單的映射結果
            return [[0]]  # 單一 tile 映射
        elif name == 'PE_graph':
            # 返回簡單的 PE 圖結構
            return []
        elif name == 'global_buf_size':
            return 1024  # 1KB 緩衝區
        elif name == 'global_adder_num':
            return 1
        elif name == 'global_multiplier_num':
            return 1
        elif name == 'max_inbuf_size':
            return 256  # 256 bytes
        elif name == 'max_outbuf_size':
            return 256  # 256 bytes
        else:
            # 對於其他未定義的屬性，返回預設值而不是拋出異常
            print(f"警告: MaxCutTCG 缺少屬性 '{name}'，使用預設值")
            return 0


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
            print("硬體映射成功建立")
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
        
        # 延遲模擬 - 使用 TCG 的 NetStruct 格式
        latency_model = Model_latency(
            NetStruct=TCG_mapping.NetStruct, 
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
            NetStruct=TCG_mapping.NetStruct, 
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
            NetStruct=TCG_mapping.NetStruct, 
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
            NetStruct=TCG_mapping.NetStruct, 
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
        best_partition = rram_res.get('best_partition', None)
        best_value = rram_res.get('best_cut_value', rram_res['cut_max'])
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
