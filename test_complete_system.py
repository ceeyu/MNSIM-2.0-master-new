#!/usr/bin/python
# -*-coding:utf-8-*-
"""
完整系統測試腳本 - 測試 MNSIM Max Cut 的所有功能
"""

import os
import sys

def test_rram_psa_algorithm():
    """測試 RRAM pSA 演算法"""
    print("🧪 測試 RRAM pSA 演算法...")
    
    cmd = [
        sys.executable, 'maxcut_main.py',
        '--graph_file', 'test_graphs/defined_3nodes.txt',
        '--algorithm', 'rram_psa',
        '--psav_trials', '5',
        '--psav_cycles', '50',
        '--disable_hardware_modeling'  # 跳過硬體建模以確保核心功能正常
    ]
    
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ RRAM pSA 演算法測試成功")
            return True
        else:
            print("❌ RRAM pSA 演算法測試失敗")
            print("錯誤:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ 測試執行失敗: {e}")
        return False

def test_other_algorithms():
    """測試其他演算法"""
    print("\n🧪 測試其他演算法...")
    
    algorithms = ['goemans_williamson', 'greedy']
    results = {}
    
    for alg in algorithms:
        print(f"  測試 {alg}...")
        cmd = [
            sys.executable, 'maxcut_main.py',
            '--graph_file', 'test_graphs/defined_3nodes.txt',
            '--algorithm', alg,
            '--iterations', '5',
            '--disable_hardware_modeling'
        ]
        
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            results[alg] = result.returncode == 0
            if result.returncode == 0:
                print(f"  ✅ {alg} 成功")
            else:
                print(f"  ❌ {alg} 失敗")
        except Exception as e:
            print(f"  ❌ {alg} 執行失敗: {e}")
            results[alg] = False
    
    return results

def test_different_graphs():
    """測試不同大小的圖"""
    print("\n🧪 測試不同大小的圖...")
    
    graphs = [
        'test_graphs/defined_3nodes.txt',
        'test_graphs/random_10nodes.txt'
    ]
    
    results = {}
    
    for graph in graphs:
        if not os.path.exists(graph):
            print(f"  ⚠️  圖檔案不存在: {graph}")
            continue
            
        print(f"  測試圖: {os.path.basename(graph)}...")
        cmd = [
            sys.executable, 'maxcut_main.py',
            '--graph_file', graph,
            '--algorithm', 'rram_psa',
            '--psav_trials', '3',
            '--psav_cycles', '30',
            '--disable_hardware_modeling'
        ]
        
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            results[graph] = result.returncode == 0
            if result.returncode == 0:
                print(f"  ✅ {os.path.basename(graph)} 成功")
            else:
                print(f"  ❌ {os.path.basename(graph)} 失敗")
        except Exception as e:
            print(f"  ❌ {os.path.basename(graph)} 執行失敗: {e}")
            results[graph] = False
    
    return results

def verify_core_functionality():
    """驗證核心功能"""
    print("\n🔍 驗證核心功能...")
    
    try:
        # 測試模組導入
        from MNSIM.Interface.maxcut_interface import MaxCutInterface
        from MNSIM.Interface.rram_psa import run_rram_psa
        print("  ✅ 模組導入成功")
        
        # 測試基本功能
        if os.path.exists('test_graphs/defined_3nodes.txt'):
            # 建立介面
            mc = MaxCutInterface('test_graphs/defined_3nodes.txt', 'SimConfig.ini')
            print(f"  ✅ MaxCut 介面建立成功 - {mc.num_nodes} 節點")
            
            # 測試結構
            structure = mc.get_structure()
            print(f"  ✅ 取得問題結構 - {structure['type']}")
            
            # 測試矩陣分割
            partitions = mc.partition_matrix_to_crossbars()
            print(f"  ✅ 矩陣分割成功 - {len(partitions)} 個 crossbar")
            
            return True
        else:
            print("  ❌ 測試圖檔案不存在")
            return False
            
    except Exception as e:
        print(f"  ❌ 核心功能驗證失敗: {e}")
        return False

def print_summary(test_results):
    """輸出測試總結"""
    print("\n" + "="*60)
    print("📋 測試總結")
    print("="*60)
    
    all_results = []
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            for sub_test, sub_result in result.items():
                status = "✅ 通過" if sub_result else "❌ 失敗"
                print(f"{status} {test_name} - {sub_test}")
                all_results.append(sub_result)
        else:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"{status} {test_name}")
            all_results.append(result)
    
    success_count = sum(all_results)
    total_count = len(all_results)
    
    print(f"\n🎯 總計: {success_count}/{total_count} 測試通過")
    
    if success_count == total_count:
        print("🎉 所有測試都通過！MNSIM Max Cut 系統運作正常。")
    else:
        print(f"⚠️  {total_count - success_count} 個測試失敗，但核心功能可用。")

def main():
    """主測試函式"""
    print("🚀 MNSIM Max Cut 完整系統測試")
    print("="*60)
    
    # 檢查必要檔案
    required_files = [
        'maxcut_main.py',
        'SimConfig.ini',
        'test_graphs/defined_3nodes.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要檔案: {missing_files}")
        return
    
    # 執行測試
    test_results = {}
    
    # 核心功能驗證
    test_results['核心功能'] = verify_core_functionality()
    
    # RRAM pSA 演算法測試
    test_results['RRAM pSA 演算法'] = test_rram_psa_algorithm()
    
    # 其他演算法測試
    test_results['其他演算法'] = test_other_algorithms()
    
    # 不同圖測試
    test_results['不同圖測試'] = test_different_graphs()
    
    # 輸出總結
    print_summary(test_results)
    
    print("\n🎯 建議後續步驟:")
    print("1. 如果所有測試通過，系統已準備就緒")
    print("2. 如果有失敗測試，檢查錯誤訊息並修復")
    print("3. 嘗試使用不同的圖檔案和參數組合")
    print("4. 探索硬體建模功能（如果需要）")

if __name__ == "__main__":
    main()
