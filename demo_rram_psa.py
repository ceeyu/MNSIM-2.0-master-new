#!/usr/bin/python
# -*-coding:utf-8-*-
"""
MNSIM RRAM pSA Max Cut 演示程式
專注於 RRAM 硬體加速的參數化模擬退火演算法

使用方式：
python demo_rram_psa.py
"""

import os
import sys
from MNSIM.Interface.rram_psa import run_rram_psa
from MNSIM.Interface.maxcut_interface import MaxCutInterface

def demo_basic_rram_psa():
    """基本 RRAM pSA 演示"""
    print("🚀 MNSIM RRAM pSA Max Cut 演示")
    print("=" * 60)
    
    # 測試參數
    graph_file = "test_graphs/defined_3nodes.txt"
    config_file = "SimConfig_MaxCut.ini"
    
    if not os.path.exists(graph_file):
        print(f"❌ 找不到測試圖檔案: {graph_file}")
        print("請先執行: python create_sample_graphs.py")
        return
    
    if not os.path.exists(config_file):
        print(f"⚠️  使用預設設定檔: SimConfig.ini")
        config_file = "SimConfig.ini"
    
    print(f"📊 圖檔案: {graph_file}")
    print(f"⚙️  硬體設定: {config_file}")
    print("-" * 60)
    
    # 執行 RRAM pSA
    try:
        results = run_rram_psa(
            graph_file=graph_file,
            SimConfig_path=config_file,
            trials=10,
            cycles=100,
            tau=1,
            param_type=2
        )
        
        print("\n🎯 演示成功！")
        print(f"✅ 最佳 Cut 值: {results['best_cut_value']}")
        print(f"✅ 最佳分割: {results['best_partition']}")
        print(f"✅ 平均時間: {results['time_avg']:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失敗: {e}")
        return False

def demo_parameter_comparison():
    """參數比較演示"""
    print("\n" + "=" * 60)
    print("🔬 RRAM pSA 參數比較演示")
    print("=" * 60)
    
    graph_file = "test_graphs/defined_3nodes.txt"
    config_file = "SimConfig.ini"
    
    if not os.path.exists(graph_file):
        print(f"❌ 找不到測試圖檔案: {graph_file}")
        return
    
    # 不同參數組合
    param_sets = [
        {"trials": 5, "cycles": 50, "name": "快速版本"},
        {"trials": 10, "cycles": 100, "name": "標準版本"},
        {"trials": 20, "cycles": 200, "name": "精確版本"}
    ]
    
    results = []
    
    for params in param_sets:
        print(f"\n🧪 測試 {params['name']} (試驗={params['trials']}, 週期={params['cycles']})")
        
        try:
            res = run_rram_psa(
                graph_file=graph_file,
                SimConfig_path=config_file,
                trials=params['trials'],
                cycles=params['cycles'],
                tau=1,
                param_type=2
            )
            
            results.append({
                'name': params['name'],
                'cut_value': res['best_cut_value'],
                'time': res['time_avg'],
                'std': res['cut_std']
            })
            
            print(f"  ✅ Cut 值: {res['best_cut_value']:.1f}, 時間: {res['time_avg']:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ 失敗: {e}")
    
    # 比較結果
    if results:
        print("\n📊 參數比較結果:")
        print("-" * 50)
        for r in results:
            print(f"{r['name']:12} | Cut值: {r['cut_value']:4.1f} | 時間: {r['time']:6.2f}ms | 標準差: {r['std']:.2f}")

def demo_different_graphs():
    """不同圖大小演示"""
    print("\n" + "=" * 60)
    print("📈 不同圖大小演示")
    print("=" * 60)
    
    config_file = "SimConfig.ini"
    
    # 不同圖檔案
    graphs = [
        {"file": "test_graphs/defined_3nodes.txt", "name": "3節點圖"},
        {"file": "test_graphs/random_10nodes.txt", "name": "10節點隨機圖"},
        {"file": "test_graphs/grid_4x4.txt", "name": "4x4網格圖"}
    ]
    
    for graph in graphs:
        if not os.path.exists(graph["file"]):
            print(f"⚠️  跳過 {graph['name']} - 檔案不存在")
            continue
        
        print(f"\n🔍 測試 {graph['name']}")
        
        try:
            # 先檢查圖資訊
            mc = MaxCutInterface(graph["file"], config_file)
            print(f"  節點數: {mc.num_nodes}, 邊數: {len(mc.graph.edges())}")
            
            # 執行 RRAM pSA
            res = run_rram_psa(
                graph_file=graph["file"],
                SimConfig_path=config_file,
                trials=5,
                cycles=50,
                tau=1,
                param_type=2
            )
            
            print(f"  ✅ 最佳 Cut 值: {res['best_cut_value']:.1f}")
            print(f"  ✅ 平均時間: {res['time_avg']:.2f} ms")
            
        except Exception as e:
            print(f"  ❌ 失敗: {e}")

def main():
    """主演示函式"""
    print("🎯 MNSIM RRAM pSA Max Cut 完整演示")
    print("此演示程式展示 RRAM 硬體加速的參數化模擬退火演算法")
    print("=" * 80)
    
    # 基本演示
    success = demo_basic_rram_psa()
    
    if success:
        # 參數比較演示
        demo_parameter_comparison()
        
        # 不同圖演示
        demo_different_graphs()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("=" * 60)
        print("🔗 後續步驟:")
        print("1. 嘗試自己的圖檔案")
        print("2. 調整 RRAM pSA 參數")
        print("3. 啟用硬體建模功能")
        print("4. 比較不同圖的求解效果")
        
        print("\n📖 使用完整程式:")
        print("python maxcut_main.py --graph_file your_graph.txt")
        
    else:
        print("\n❌ 基本演示失敗，請檢查環境設定")

if __name__ == "__main__":
    main()
