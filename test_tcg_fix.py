#!/usr/bin/python
# -*-coding:utf-8-*-
"""
測試 MaxCutTCG 修改後的硬體建模功能
"""

import os
import sys
from MNSIM.Interface.maxcut_interface import MaxCutInterface

def test_tcg_initialization():
    """測試 MaxCutTCG 初始化"""
    print("🧪 測試 MaxCutTCG 初始化")
    print("=" * 50)
    
    # 測試圖檔案
    graph_file = "test_graphs/defined_3nodes.txt"
    config_file = "SimConfig.ini"
    
    if not os.path.exists(graph_file):
        print(f"❌ 找不到測試圖檔案: {graph_file}")
        return False
    
    try:
        # 建立 Max Cut 介面
        maxcut_interface = MaxCutInterface(
            graph_file=graph_file,
            SimConfig_path=config_file,
            algorithm='rram_psa'
        )
        
        # 取得問題結構
        structure = maxcut_interface.get_structure()
        print(f"✅ 結構載入成功: {structure['type']}, {len(structure['layers'])} 層")
        
        # 測試 MaxCutTCG 初始化
        from maxcut_main import MaxCutTCG
        
        tcg = MaxCutTCG(structure, config_file)
        print(f"✅ TCG 初始化成功: {tcg.layer_num} 層")
        
        # 檢查 NetStruct 格式
        print(f"✅ NetStruct 長度: {len(tcg.NetStruct)}")
        for i, layer in enumerate(tcg.NetStruct):
            layer_dict = layer[0][0]
            print(f"  層 {i}: type={layer_dict['type']}, "
                  f"input={layer_dict['Infeature']}, output={layer_dict['Outfeature']}")
            if i == 0:
                print(f"    卷積參數: kernel={layer_dict.get('Kernelsize', 'N/A')}, "
                      f"stride={layer_dict.get('Stride', 'N/A')}")
        
        # 測試必要方法
        tcg.mapping_net()
        tcg.calculate_transfer_distance()
        print(f"✅ 必要方法執行成功")
        
        # 檢查屬性
        print(f"✅ 屬性檢查:")
        print(f"  tile_total_num: {tcg.tile_total_num}")
        print(f"  max_inbuf_size: {tcg.max_inbuf_size}")
        print(f"  max_outbuf_size: {tcg.max_outbuf_size}")
        print(f"  inLayer_distance: {len(tcg.inLayer_distance)}x{len(tcg.inLayer_distance[0])}")
        print(f"  transLayer_distance: {len(tcg.transLayer_distance)}x{len(tcg.transLayer_distance[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gxx_benchmark_loading():
    """測試 Gxx benchmark 圖檔案載入"""
    print("\n🧪 測試 Gxx Benchmark 圖檔案載入")
    print("=" * 50)
    
    # 測試較小的 Gxx 圖檔案
    gxx_files = [
        "GPU-pSAv-main/graph/G47.txt",  # 1000 節點
        "GPU-pSAv-main/graph/G48.txt",  # 3000 節點
    ]
    
    config_file = "SimConfig.ini"
    
    for graph_file in gxx_files:
        if not os.path.exists(graph_file):
            print(f"⚠️  跳過 {graph_file} - 檔案不存在")
            continue
            
        print(f"\n📊 測試 {os.path.basename(graph_file)}")
        print("-" * 40)
        
        try:
            # 載入圖檔案
            mc_interface = MaxCutInterface(
                graph_file=graph_file,
                SimConfig_path=config_file,
                algorithm='rram_psa'
            )
            
            print(f"✅ 圖載入成功")
            print(f"   節點數: {mc_interface.num_nodes}")
            print(f"   邊數: {len(mc_interface.graph.edges())}")
            
            # 測試 TCG 建立
            from maxcut_main import MaxCutTCG
            structure = mc_interface.get_structure()
            tcg = MaxCutTCG(structure, config_file)
            
            print(f"✅ TCG 建立成功")
            print(f"   層數: {tcg.layer_num}")
            print(f"   NetStruct 格式: {len(tcg.NetStruct)} 層")
            
            # 檢查第一層是否為卷積層
            first_layer = tcg.NetStruct[0][0][0]
            print(f"   第一層類型: {first_layer['type']}")
            if first_layer['type'] == 'conv':
                print(f"   卷積參數: kernel={first_layer['Kernelsize']}, "
                      f"stride={first_layer['Stride']}, padding={first_layer['Padding']}")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主測試函式"""
    print("🚀 MaxCutTCG 修改驗證測試")
    print("此測試驗證 MaxCutTCG 是否能正確初始化並支援硬體建模")
    print("=" * 80)
    
    # 測試基本 TCG 初始化
    success = test_tcg_initialization()
    
    if success:
        # 測試 Gxx benchmark 載入
        test_gxx_benchmark_loading()
        
        print("\n" + "=" * 60)
        print("🎉 測試完成！")
        print("=" * 60)
        print("💡 如果測試通過，現在可以嘗試:")
        print("python maxcut_main.py --graph_file GPU-pSAv-main/graph/G47.txt --algorithm rram_psa")
    else:
        print("\n❌ 基本測試失敗，請檢查修改")

if __name__ == "__main__":
    main()
