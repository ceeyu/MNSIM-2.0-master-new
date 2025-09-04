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
from MNSIM.Interface.rram_psa import run_rram_psa
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy


class MockTile:
    """模擬 MNSIM 期望的 tile 物件"""
    def __init__(self):
        # 提供所有 MNSIM 硬體建模需要的靜態屬性
        # 總面積
        self.tile_area = 1000000  # 1 mm²
        
        # 各模組面積 (um²)
        self.tile_xbar_area = 500000      # Crossbar 面積
        self.tile_ADC_area = 100000       # ADC 面積
        self.tile_DAC_area = 80000        # DAC 面積
        self.tile_digital_area = 150000   # 數位電路面積
        self.tile_adder_area = 50000      # 加法器面積
        self.tile_shiftreg_area = 20000   # 移位暫存器面積
        self.tile_iReg_area = 15000       # 輸入暫存器面積
        self.tile_oReg_area = 15000       # 輸出暫存器面積
        self.tile_input_demux_area = 10000 # 輸入解多工器面積
        self.tile_output_mux_area = 10000  # 輸出多工器面積
        self.tile_jointmodule_area = 5000  # 聯合模組面積
        self.tile_buffer_area = 80000     # 緩衝區面積
        self.tile_pooling_area = 0        # 池化模組面積 (Max Cut 不需要)
        
        # 功率相關屬性 (W)
        self.tile_power = 0.1  # 總功率
        self.tile_read_power = 0.08       # 讀取功率
        self.tile_xbar_read_power = 0.03  # Crossbar 讀取功率
        self.tile_ADC_read_power = 0.02   # ADC 讀取功率
        self.tile_DAC_read_power = 0.01   # DAC 讀取功率
        self.tile_digital_read_power = 0.015 # 數位電路讀取功率
        self.tile_adder_read_power = 0.005  # 加法器讀取功率
        self.tile_shiftreg_read_power = 0.002 # 移位暫存器讀取功率
        self.tile_iReg_read_power = 0.001   # 輸入暫存器讀取功率
        self.tile_oReg_read_power = 0.001   # 輸出暫存器讀取功率
        self.tile_input_demux_read_power = 0.001 # 輸入解多工器讀取功率
        self.tile_output_mux_read_power = 0.001  # 輸出多工器讀取功率
        self.tile_jointmodule_read_power = 0.001  # 聯合模組讀取功率
        self.tile_buffer_read_power = 0.01      # 緩衝區讀取功率
        self.tile_buffer_r_read_power = 0.006   # 緩衝區讀取功率
        self.tile_buffer_w_read_power = 0.004   # 緩衝區寫入功率
        self.tile_pooling_read_power = 0.0      # 池化模組讀取功率 (Max Cut 不需要)
    
    def calculate_tile_area(self, **kwargs):
        """計算 tile 面積"""
        return self.tile_area
    
    def calculate_tile_power(self, **kwargs):
        """計算 tile 功率"""
        return self.tile_power
    
    def calculate_tile_read_power_fast(self, **kwargs):
        """計算 tile 快速讀取功率 - 功率建模需要"""
        # 這個方法被 Model_inference_power 調用，但我們已經在 __init__ 中設定了所有功率值
        # 所以這裡不需要做任何計算，只需要確保方法存在
        pass


class MaxCutTCG:
    """
    專為 Max Cut 問題設計的簡化 Tile Connection Graph
    提供與標準 TCG 完全相容的介面
    """
    def __init__(self, maxcut_structure, SimConfig_path, multiple=None):
        # 保存結構資訊以供後續使用
        self.maxcut_structure = maxcut_structure
        self.SimConfig_path = SimConfig_path
        
        # 基本屬性以相容硬體模擬模組
        self.layer_num = len(maxcut_structure['layers'])
        self.layer_tileinfo = []
        
        # 關鍵：提供與標準 TCG 相容的 net 屬性
        # 標準 TCG 使用 self.net[layer_id][0][0] 來訪問 layer_dict
        self.net = []
        
        # 為每個層建立基本的 tile 資訊
        for i, layer_info in enumerate(maxcut_structure['layers']):
            # Tile 資訊 - 需要提供 MNSIM 期望的完整屬性
            tile_info = {
                'type': 'fc',
                'startid': i,
                'mx': 1,  # PE 數量 x 軸
                'my': 1,  # PE 數量 y 軸
                'max_group': 1,
                'max_row': layer_info['input_size'],
                'max_column': layer_info['output_size'],
                'max_PE': 1,  # 每個 tile 的 PE 數量
                'tilenum': 1,  # 每個層的 tile 數量
                'Inputindex': [-1] if i == 0 else [i-1],
                'Outputindex': [i+1] if i < self.layer_num-1 else [],
                'is_branchin': -1,
                'is_branchout': -1
            }
            self.layer_tileinfo.append(tile_info)
            
            # 網路結構 (硬體模擬模組需要的格式)
            # 第一層需要模擬卷積層的格式以滿足 MNSIM 的硬編碼假設
            if i == 0:
                # 第一層：模擬卷積層格式
                layer_dict = {
                    'type': 'conv',  # 第一層必須是 conv
                    'Infeature': layer_info['input_size'],
                    'Outfeature': layer_info['output_size'],
                    'Weightbit': 8,
                    'Inputbit': 8,
                    'outputbit': 8,
                    'Inputsize': [1, layer_info['input_size']],  # [batch, features]
                    'Outputsize': [1, layer_info['output_size']],
                    'Inputindex': [-1] if i == 0 else [i-1],
                    'Outputindex': [i+1] if i < self.layer_num-1 else [],
                    'Layerindex': i,
                    # 卷積層特有參數
                    'Kernelsize': 1,  # 1x1 卷積，等效於全連接
                    'Stride': 1,
                    'Inputchannel': layer_info['input_size'],
                    'Outputchannel': layer_info['output_size'],
                    'Padding': 0
                }
            else:
                # 其他層：全連接層格式
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
            
            # 關鍵：提供與標準 TCG 相容的 net 格式
            # 標準 TCG 期望：self.net[layer_id][0][0] = layer_dict
            self.net.append([[layer_dict]])
        
        # 為了向後相容，也提供 NetStruct 屬性
        self.NetStruct = self.net
        
        print(f"MaxCut TCG 初始化完成：{self.layer_num} 層")
    def mapping_net(self):
        """提供硬體模擬模組期望的網路映射方法"""
        # 初始化傳輸距離矩陣
        self.inLayer_distance = [[0] * self.layer_num for _ in range(self.layer_num)]
        self.transLayer_distance = [[0] * self.layer_num for _ in range(self.layer_num)]
        self.layer_split = [[] for _ in range(self.layer_num)]
        
        # 為每個層設定基本的 tile 資訊
        for i in range(self.layer_num):
            if i == 0:
                # 第一層：卷積層的 tile 資訊
                self.layer_tileinfo[i].update({
                    'max_PE': 1,
                    'max_row': self.maxcut_structure['layers'][i]['input_size'],
                    'max_column': self.maxcut_structure['layers'][i]['output_size']
                })
            else:
                # 其他層：全連接層的 tile 資訊
                self.layer_tileinfo[i].update({
                    'max_PE': 1,
                    'max_row': self.maxcut_structure['layers'][i]['input_size'],
                    'max_column': self.maxcut_structure['layers'][i]['output_size']
                })
    
    def calculate_transfer_distance(self):
        """計算傳輸距離 - 硬體建模需要"""
        # 簡化：假設所有傳輸距離為 1
        for i in range(self.layer_num):
            for j in range(self.layer_num):
                if i == j:
                    self.inLayer_distance[i][j] = 0
                    self.transLayer_distance[i][j] = 0
                else:
                    self.inLayer_distance[i][j] = 1
                    self.transLayer_distance[i][j] = 1
    
    def calculate_output_stationary(self):
        """計算輸出固定策略 - 硬體建模需要"""
        pass  # 簡化實作
    
    def calculate_weight_stationary(self):
        """計算權重固定策略 - 硬體建模需要"""
        pass  # 簡化實作
    
    def calculate_input_stationary(self):
        """計算輸入固定策略 - 硬體建模需要"""
        pass  # 簡化實作
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
        elif name == 'global_adder_bitwidth':
            return 8  # 8-bit 加法器
        elif name == 'global_multiplier_bitwidth':
            return 8  # 8-bit 乘法器
        elif name == 'max_inbuf_size':
            return 256  # 256 bytes
        elif name == 'max_outbuf_size':
            return 256  # 256 bytes
        elif name == 'inLayer_distance':
            # 層內傳輸距離矩陣
            return [[0] * self.layer_num for _ in range(self.layer_num)]
        elif name == 'transLayer_distance':
            # 層間傳輸距離矩陣
            return [[0] * self.layer_num for _ in range(self.layer_num)]
        elif name == 'layer_split':
            # 層分割資訊
            return [[] for _ in range(self.layer_num)]
        elif name == 'max_PE':
            # 最大 PE 數量
            return 1
        elif name == 'tile':
            # 返回一個模擬的 tile 物件
            return MockTile()
        else:
            # 對於其他未定義的屬性，返回預設值而不是拋出異常
            print(f"警告: MaxCutTCG 缺少屬性 '{name}'，使用預設值")
            return 0


def main():
    home_path = os.getcwd()
    SimConfig_path = os.path.join(home_path, "SimConfig_MaxCut.ini")
    
    parser = argparse.ArgumentParser(description='MNSIM Max Cut Example')
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
        help="Whether delete the unnecessary files automatically")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
        help="Hardware description file location & name")
    parser.add_argument("-Graph", "--graph_file", required=True,
        help="Graph file location & name (txt/csv/graphml format)")
    parser.add_argument("-Alg", "--algorithm", default='rram_psa',
        choices=['rram_psa'],
        help="Max Cut algorithm choice (專注於 RRAM pSA)")

    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
        help="Disable hardware modeling")
    parser.add_argument("-D", "--device", default=0,
        help="Determine hardware device (CPU or GPU-id) for simulation")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
        help="Disable module simulation results output")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
        help="Disable layer-wise simulation results output")
    # RRAM pSA 相關參數
    parser.add_argument("--rram_cycles", type=int, default=200, help="RRAM pSA 退火週期數")
    parser.add_argument("--rram_trials", type=int, default=50, help="RRAM pSA 試驗次數")
    parser.add_argument("--rram_tau", type=int, default=1, help="RRAM pSA tau 參數")
    parser.add_argument("--rram_param", type=int, default=2, help="RRAM pSA 參數類型 (2 推薦)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MNSIM RRAM pSA Max Cut 模擬器")
    print("=" * 60)
    print("硬體描述檔位置:", args.hardware_description)
    print("圖檔案位置:", args.graph_file)
    print("求解算法:", args.algorithm)
    print("RRAM pSA 試驗次數:", args.rram_trials)
    print("RRAM pSA 退火週期:", args.rram_cycles)
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
    
    # 建立客製化的 TCG 映射（如果啟用硬體建模）Tile Connect Graph
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
    
    # RRAM pSA Max Cut 求解
    print("\n" + "=" * 50)
    print("開始 RRAM pSA Max Cut 求解...")
    print("=" * 50)
    
    maxcut_start_time = time.time()
    
    # 準備硬體映射資料
    crossbar_partitions = maxcut_interface.partition_matrix_to_crossbars()
    
    print(f"RRAM 矩陣分割到 {len(crossbar_partitions)} 個 crossbar")
    
    # 執行 RRAM pSA 求解
    rram_res = run_rram_psa(
        graph_file=args.graph_file,
        SimConfig_path=args.hardware_description,
        trials=args.rram_trials,
        cycles=args.rram_cycles,
        tau=args.rram_tau,
        param_type=args.rram_param
    )
    best_partition = rram_res.get('best_partition', None)
    best_value = rram_res.get('best_cut_value', rram_res['cut_max'])
    
    maxcut_end_time = time.time()
    
    # 結果輸出
    print("\n" + "=" * 50)
    print("RRAM pSA Max Cut 求解結果")
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
