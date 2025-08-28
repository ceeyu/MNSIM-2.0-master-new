#!/usr/bin/python
# -*-coding:utf-8-*-
"""
測試 Gxx Benchmark 圖檔案載入功能
"""

import os
import sys
from MNSIM.Interface.maxcut_interface import MaxCutInterface

def test_gxx_benchmark_loading():
    """測試 Gxx benchmark 圖檔案載入"""
    print("🧪 測試 Gxx Benchmark 圖檔案載入")
    print("=" * 60)
    
    # 測試不同的 Gxx 圖檔案
    gxx_files = [
        "GPU-pSAv-main/graph/G1.txt",
        "GPU-pSAv-main/graph/G22.txt", 
        "GPU-pSAv-main/graph/G47.txt",
        "GPU-pSAv-main/graph/G48.txt",
        "GPU-pSAv-main/graph/G55.txt",
        "GPU-pSAv-main/graph/G60.txt",
        "GPU-pSAv-main/graph/G67.txt",
        "GPU-pSAv-main/graph/G77.txt",
        "GPU-pSAv-main/graph/G81.txt"
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
            
            # 顯示圖資訊
            print(f"✅ 載入成功")
            print(f"   節點數: {mc_interface.num_nodes}")
            print(f"   邊數: {len(mc_interface.graph.edges())}")
            print(f"   鄰接矩陣形狀: {mc_interface.adjacency_matrix.shape}")
            
            # 檢查一些邊的權重
            edges = list(mc_interface.graph.edges(data=True))[:5]
            print(f"   前5條邊: {[(u, v, d.get('weight', 1.0)) for u, v, d in edges]}")
            
        except Exception as e:
            print(f"❌ 載入失敗: {e}")

def test_benchmark_comparison():
    """比較不同 benchmark 圖的規模"""
    print("\n" + "=" * 60)
    print("📈 Benchmark 圖規模比較")
    print("=" * 60)
    
    config_file = "SimConfig.ini"
    benchmark_info = []
    
    gxx_files = [
        "GPU-pSAv-main/graph/G1.txt",
        "GPU-pSAv-main/graph/G22.txt", 
        "GPU-pSAv-main/graph/G47.txt",
        "GPU-pSAv-main/graph/G48.txt",
        "GPU-pSAv-main/graph/G55.txt",
        "GPU-pSAv-main/graph/G60.txt",
        "GPU-pSAv-main/graph/G67.txt",
        "GPU-pSAv-main/graph/G77.txt",
        "GPU-pSAv-main/graph/G81.txt"
    ]
    
    for graph_file in gxx_files:
        if not os.path.exists(graph_file):
            continue
            
        try:
            mc_interface = MaxCutInterface(
                graph_file=graph_file,
                SimConfig_path=config_file,
                algorithm='rram_psa'
            )
            
            benchmark_info.append({
                'name': os.path.basename(graph_file),
                'nodes': mc_interface.num_nodes,
                'edges': len(mc_interface.graph.edges()),
                'density': len(mc_interface.graph.edges()) / (mc_interface.num_nodes * (mc_interface.num_nodes - 1) / 2)
            })
            
        except Exception as e:
            print(f"❌ {os.path.basename(graph_file)} 載入失敗: {e}")
    
    # 排序並顯示結果
    benchmark_info.sort(key=lambda x: x['nodes'])
    
    print(f"{'圖檔案':<12} {'節點數':<8} {'邊數':<8} {'密度':<8}")
    print("-" * 40)
    for info in benchmark_info:
        print(f"{info['name']:<12} {info['nodes']:<8} {info['edges']:<8} {info['density']:.4f}")

def test_small_benchmark():
    """測試較小的 benchmark 圖"""
    print("\n" + "=" * 60)
    print("🔍 測試較小的 Benchmark 圖")
    print("=" * 60)
    
    # 選擇較小的圖進行測試
    small_graphs = [
        "GPU-pSAv-main/graph/G47.txt",  # 1000 節點
        "GPU-pSAv-main/graph/G48.txt",  # 3000 節點
    ]
    
    config_file = "SimConfig.ini"
    
    for graph_file in small_graphs:
        if not os.path.exists(graph_file):
            print(f"⚠️  跳過 {graph_file} - 檔案不存在")
            continue
            
        print(f"\n🎯 測試 {os.path.basename(graph_file)}")
        print("-" * 40)
        
        try:
            mc_interface = MaxCutInterface(
                graph_file=graph_file,
                SimConfig_path=config_file,
                algorithm='rram_psa'
            )
            
            print(f"✅ 圖載入成功")
            print(f"   節點數: {mc_interface.num_nodes}")
            print(f"   邊數: {len(mc_interface.graph.edges())}")
            
            # 測試結構取得
            structure = mc_interface.get_structure()
            print(f"   結構類型: {structure['type']}")
            print(f"   層數: {len(structure['layers'])}")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")

def main():
    """主測試函式"""
    print("🚀 Gxx Benchmark 圖檔案載入測試")
    print("此測試驗證系統是否能正確載入 GPU-pSAv-main/graph/ 目錄下的 benchmark 圖檔案")
    print("=" * 80)
    
    # 檢查目錄是否存在
    if not os.path.exists("GPU-pSAv-main/graph/"):
        print("❌ 找不到 GPU-pSAv-main/graph/ 目錄")
        print("請確保該目錄存在並包含 Gxx.txt 檔案")
        return
    
    # 執行測試
    test_gxx_benchmark_loading()
    test_benchmark_comparison()
    test_small_benchmark()
    
    print("\n" + "=" * 60)
    print("🎉 測試完成！")
    print("=" * 60)
    print("💡 使用方式:")
    print("python maxcut_main.py --graph_file GPU-pSAv-main/graph/G47.txt --algorithm rram_psa")

if __name__ == "__main__":
    main()
