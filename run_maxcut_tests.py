#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import subprocess
import sys

def run_maxcut_test(graph_file, algorithm='goemans_williamson', iterations=50):
    """執行單個 Max Cut 測試"""
    cmd = [
        sys.executable, 'maxcut_main.py',
        '--graph_file', graph_file,
        '--algorithm', algorithm,
        '--iterations', str(iterations),
        '--disable_module_output'
    ]
    
    print(f"測試圖檔: {graph_file}")
    print(f"算法: {algorithm}")
    print(f"迭代次數: {iterations}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("測試成功完成")
            # 提取關鍵結果
            lines = result.stdout.split('\n')
            for line in lines:
                if '最佳分割值' in line or '總模擬時間' in line:
                    print(line)
        else:
            print("測試失敗:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("測試超時 (5分鐘)")
    except Exception as e:
        print(f"測試出錯: {e}")
    
    print("=" * 60)

def main():
    """執行所有測試"""
    graph_dir = "test_graphs"
    
    if not os.path.exists(graph_dir):
        print("找不到測試圖目錄，請先執行 create_sample_graphs.py")
        return
    
    # 取得所有圖檔案
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.txt')]
    
    print(f"找到 {len(graph_files)} 個測試圖檔案")
    print("開始執行測試...")
    print("=" * 60)
    
    for graph_file in sorted(graph_files):
        graph_path = os.path.join(graph_dir, graph_file)
        
        # 根據圖大小調整迭代次數
        if '200nodes' in graph_file:
            iterations = 20  # 大圖用較少迭代
        elif 'complete' in graph_file:
            iterations = 30  # 完全圖中等迭代
        else:
            iterations = 50  # 一般圖較多迭代
        
        run_maxcut_test(graph_path, 'goemans_williamson', iterations)

if __name__ == '__main__':
    main()
