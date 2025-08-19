#!/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import networkx as nx
import os

def create_random_graph(num_nodes: int, edge_prob: float, weight_range: tuple, filename: str):
    """建立隨機圖"""
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    with open(filename, 'w') as f:
        f.write(f"# Random graph with {num_nodes} nodes, edge probability {edge_prob}\n")
        f.write(f"# Format: node1 node2 weight\n")
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            f.write(f"{u} {v} {weight:.3f}\n")
    
    print(f"建立隨機圖: {filename} ({len(G.edges())} 條邊)")
    return filename

def create_complete_graph(num_nodes: int, weight_range: tuple, filename: str):
    """建立完全圖"""
    G = nx.complete_graph(num_nodes)
    
    with open(filename, 'w') as f:
        f.write(f"# Complete graph with {num_nodes} nodes\n")
        f.write(f"# Format: node1 node2 weight\n")
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            f.write(f"{u} {v} {weight:.3f}\n")
    
    print(f"建立完全圖: {filename} ({len(G.edges())} 條邊)")
    return filename

def create_grid_graph(rows: int, cols: int, weight_range: tuple, filename: str):
    """建立網格圖"""
    G = nx.grid_2d_graph(rows, cols)
    
    # 重新標記節點為整數
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    with open(filename, 'w') as f:
        f.write(f"# Grid graph {rows}x{cols}\n")
        f.write(f"# Format: node1 node2 weight\n")
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            f.write(f"{u} {v} {weight:.3f}\n")
    
    print(f"建立網格圖: {filename} ({len(G.edges())} 條邊)")
    return filename

def create_scale_free_graph(num_nodes: int, weight_range: tuple, filename: str):
    """建立無標度圖"""
    G = nx.barabasi_albert_graph(num_nodes, 3)
    
    with open(filename, 'w') as f:
        f.write(f"# Scale-free graph with {num_nodes} nodes\n")
        f.write(f"# Format: node1 node2 weight\n")
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            f.write(f"{u} {v} {weight:.3f}\n")
    
    print(f"建立無標度圖: {filename} ({len(G.edges())} 條邊)")
    return filename

def create_benchmark_graph(filename: str):
    """建立已知最佳解的基準圖"""
    # 簡單的二分圖，最佳分割很明顯
    with open(filename, 'w') as f:
        f.write(f"# Benchmark bipartite-like graph\n")
        f.write(f"# Optimal cut should separate nodes 0-4 from nodes 5-9\n")
        f.write(f"# Format: node1 node2 weight\n")
        
        # 組內連接 (權重較小)
        for i in range(5):
            for j in range(i+1, 5):
                f.write(f"{i} {j} 1.0\n")
        
        for i in range(5, 10):
            for j in range(i+1, 10):
                f.write(f"{i} {j} 1.0\n")
        
        # 組間連接 (權重較大)
        for i in range(5):
            for j in range(5, 10):
                f.write(f"{i} {j} 5.0\n")
    
    print(f"建立基準圖: {filename}")
    return filename

def main():
    """建立各種測試圖"""
    print("建立 Max Cut 測試圖檔案...")
    
    # 建立圖檔案目錄
    graph_dir = "test_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # 建立各種類型的圖
    graphs = []
    
    # 小型測試圖
    graphs.append(create_random_graph(
        10, 0.4, (1.0, 5.0), 
        os.path.join(graph_dir, "random_10nodes.txt")
    ))
    
    graphs.append(create_complete_graph(
        8, (1.0, 3.0), 
        os.path.join(graph_dir, "complete_8nodes.txt")
    ))
    
    graphs.append(create_grid_graph(
        4, 4, (1.0, 4.0), 
        os.path.join(graph_dir, "grid_4x4.txt")
    ))
    
    # 中型測試圖
    graphs.append(create_random_graph(
        50, 0.2, (1.0, 10.0), 
        os.path.join(graph_dir, "random_50nodes.txt")
    ))
    
    graphs.append(create_scale_free_graph(
        30, (1.0, 8.0), 
        os.path.join(graph_dir, "scalefree_30nodes.txt")
    ))
    
    # 基準圖 (已知最佳解)
    graphs.append(create_benchmark_graph(
        os.path.join(graph_dir, "benchmark_bipartite.txt")
    ))
    
    # 大型測試圖 (測試 crossbar 分割)
    graphs.append(create_random_graph(
        200, 0.05, (1.0, 15.0), 
        os.path.join(graph_dir, "random_200nodes.txt")
    ))
    
    print(f"\n總共建立了 {len(graphs)} 個測試圖檔案")
    print("圖檔案位於:", graph_dir)
    
    # 建立測試腳本
    test_script = "run_maxcut_tests.py"
    with open(test_script, 'w') as f:
        f.write("""#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import subprocess
import sys

def run_maxcut_test(graph_file, algorithm='goemans_williamson', iterations=50):
    \"\"\"執行單個 Max Cut 測試\"\"\"
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
            lines = result.stdout.split('\\n')
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
    \"\"\"執行所有測試\"\"\"
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
""")
    
    print(f"\n建立測試腳本: {test_script}")
    print("\n使用方式:")
    print("1. python create_sample_graphs.py  # 建立測試圖")
    print("2. python maxcut_main.py --graph_file test_graphs/random_10nodes.txt  # 單個測試")
    print("3. python run_maxcut_tests.py  # 執行所有測試")

if __name__ == '__main__':
    main()
