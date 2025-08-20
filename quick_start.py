#!/usr/bin/python
# -*-coding:utf-8-*-
"""
MNSIM Max Cut 快速開始腳本
自動執行所有演算法並比較結果
"""

import os
import subprocess
import sys
import time

def run_command(cmd, description):
    """執行命令並顯示結果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"執行命令: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print("✅ 執行成功")
            print(result.stdout)
        else:
            print("❌ 執行失敗")
            print("錯誤輸出:", result.stderr)
        
        print(f"⏱️  執行時間: {end_time - start_time:.2f} 秒")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 執行超時 (5分鐘)")
        return False
    except Exception as e:
        print(f"💥 執行出錯: {e}")
        return False

def check_dependencies():
    """檢查依賴套件"""
    print("🔍 檢查系統依賴...")
    
    required_packages = ['numpy', 'networkx']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (缺少)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n請安裝缺少的套件: pip install {' '.join(missing_packages)}")
        return False
    
    # 檢查可選的 GPU 支援
    try:
        import pycuda.autoinit
        print("✅ PyCUDA (GPU 支援)")
        gpu_available = True
    except ImportError:
        print("⚠️  PyCUDA (GPU 支援不可用，跳過 GPU 演算法)")
        gpu_available = False
    
    return True, gpu_available

def create_test_graphs():
    """建立測試圖"""
    print("\n📊 建立測試圖檔案...")
    
    if not os.path.exists('create_sample_graphs.py'):
        print("❌ 找不到 create_sample_graphs.py")
        return False
    
    return run_command([sys.executable, 'create_sample_graphs.py'], "建立測試圖")

def test_algorithms(gpu_available=False):
    """測試所有演算法"""
    
    # 測試圖檔案
    test_graph = "test_graphs/defined_3nodes.txt"
    if not os.path.exists(test_graph):
        print(f"❌ 找不到測試圖: {test_graph}")
        return
    
    # 測試案例
    test_cases = [
        {
            'name': '內建啟發式 (Goemans-Williamson)',
            'cmd': [sys.executable, 'maxcut_main.py', 
                   '--graph_file', test_graph, 
                   '--algorithm', 'goemans_williamson', 
                   '--iterations', '10',
                   '--disable_module_output'],
            'required': True
        },
        {
            'name': 'RRAM pSA (推薦)',
            'cmd': [sys.executable, 'maxcut_main.py', 
                   '--graph_file', test_graph, 
                   '--algorithm', 'rram_psa', 
                   '--psav_trials', '5', 
                   '--psav_cycles', '50',
                   '--disable_module_output'],
            'required': True
        },
        {
            'name': '貪婪演算法',
            'cmd': [sys.executable, 'maxcut_main.py', 
                   '--graph_file', test_graph, 
                   '--algorithm', 'greedy', 
                   '--iterations', '10',
                   '--disable_module_output'],
            'required': True
        }
    ]
    
    # GPU 測試（可選）
    if gpu_available:
        test_cases.append({
            'name': 'GPU pSA (需要 CUDA)',
            'cmd': [sys.executable, 'maxcut_main.py', 
                   '--graph_file', test_graph, 
                   '--algorithm', 'psav_psa', 
                   '--psav_trials', '3', 
                   '--psav_cycles', '50',
                   '--psav_gpu', '0',
                   '--disable_module_output'],
            'required': False
        })
    
    # 執行測試
    results = {}
    for test_case in test_cases:
        success = run_command(test_case['cmd'], test_case['name'])
        results[test_case['name']] = success
        
        if not success and test_case['required']:
            print(f"⚠️  必要測試失敗: {test_case['name']}")
    
    return results

def test_custom_spin_vector():
    """測試自定義 spin 向量"""
    test_graph = "test_graphs/defined_3nodes.txt"
    cmd = [sys.executable, 'maxcut_main.py', 
           '--graph_file', test_graph, 
           '--algorithm', 'goemans_williamson',
           '--spin_vector', '1,-1,1',
           '--iterations', '1',
           '--disable_module_output']
    
    return run_command(cmd, "自定義 Spin 向量測試 [1,-1,1]")

def performance_comparison():
    """效能比較測試"""
    print("\n🏁 效能比較測試...")
    
    test_graph = "test_graphs/random_10nodes.txt"
    if not os.path.exists(test_graph):
        print(f"⚠️  跳過效能測試，找不到: {test_graph}")
        return
    
    algorithms = ['goemans_williamson', 'rram_psa', 'greedy']
    
    for alg in algorithms:
        cmd = [sys.executable, 'maxcut_main.py', 
               '--graph_file', test_graph, 
               '--algorithm', alg, 
               '--psav_trials', '10' if 'psa' in alg else '20',
               '--iterations', '20' if alg == 'goemans_williamson' else '20',
               '--disable_module_output',
               '--disable_layer_output']
        
        run_command(cmd, f"效能測試 - {alg}")

def print_summary(results):
    """輸出測試總結"""
    print("\n" + "="*60)
    print("📋 測試總結")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{status} {test_name}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n🎯 總計: {success_count}/{total_count} 測試通過")
    
    if success_count == total_count:
        print("🎉 所有測試都通過！系統運作正常。")
    else:
        print("⚠️  部分測試失敗，請檢查錯誤訊息。")

def print_next_steps():
    """輸出後續步驟建議"""
    print("\n" + "="*60)
    print("🎯 後續步驟建議")
    print("="*60)
    
    print("1. 📖 閱讀完整文件:")
    print("   cat README_MaxCut.md")
    
    print("\n2. 🧪 嘗試不同的圖:")
    print("   python maxcut_main.py --graph_file test_graphs/random_20nodes.txt --algorithm rram_psa")
    
    print("\n3. ⚙️  調整硬體參數:")
    print("   python maxcut_main.py --hardware_description SimConfig_MaxCut.ini --graph_file your_graph.txt")
    
    print("\n4. 📊 批次測試:")
    print("   python run_maxcut_tests.py")
    
    print("\n5. 🔬 深入研究 RRAM pSA:")
    print("   python -c \"from MNSIM.Interface.rram_psa import run_rram_psa; help(run_rram_psa)\"")

def main():
    """主函式"""
    print("🚀 MNSIM Max Cut 快速開始")
    print("="*60)
    print("這個腳本會自動測試所有功能並展示使用方式")
    print("預估執行時間: 3-5 分鐘")
    print("="*60)
    
    # 檢查依賴
    deps_ok, gpu_available = check_dependencies()
    if not deps_ok:
        return
    
    # 建立測試圖
    if not create_test_graphs():
        print("❌ 無法建立測試圖，程式終止")
        return
    
    # 測試演算法
    results = test_algorithms(gpu_available)
    
    # 測試自定義 spin 向量
    spin_test = test_custom_spin_vector()
    results["自定義 Spin 向量"] = spin_test
    
    # 效能比較（可選）
    try:
        performance_comparison()
    except Exception as e:
        print(f"⚠️  效能測試跳過: {e}")
    
    # 輸出總結
    print_summary(results)
    print_next_steps()

if __name__ == "__main__":
    main()
