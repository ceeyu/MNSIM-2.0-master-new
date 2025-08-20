#!/usr/bin/python
# -*-coding:utf-8-*-
"""
MNSIM Max Cut 執行範例腳本
展示各種使用情境和參數組合
"""

import os
import subprocess
import sys
import time

class MaxCutDemo:
    def __init__(self):
        self.examples = []
        self.setup_examples()
    
    def setup_examples(self):
        """設定所有範例"""
        
        # 範例 1：基本 3 節點測試
        self.examples.append({
            'name': '🔰 基本測試 - 3 節點圖',
            'description': '使用 RRAM pSA 求解簡單的 3 節點 Max Cut 問題',
            'cmd': [
                sys.executable, 'maxcut_main.py',
                '--graph_file', 'test_graphs/defined_3nodes.txt',
                '--algorithm', 'rram_psa',
                '--psav_trials', '5',
                '--psav_cycles', '50'
            ],
            'expected': 'Cut 值應該為 3，分割 {0,2} vs {1}',
            'category': 'basic'
        })
        
        # 範例 2：演算法比較
        algorithms = [
            ('goemans_williamson', '內建啟發式'),
            ('rram_psa', 'RRAM pSA'),
            ('greedy', '貪婪演算法')
        ]
        
        for alg, desc in algorithms:
            self.examples.append({
                'name': f'🔬 演算法比較 - {desc}',
                'description': f'使用 {desc} 求解 10 節點隨機圖',
                'cmd': [
                    sys.executable, 'maxcut_main.py',
                    '--graph_file', 'test_graphs/random_10nodes.txt',
                    '--algorithm', alg,
                    '--psav_trials', '10' if 'psa' in alg else '20',
                    '--iterations', '20',
                    '--disable_module_output'
                ],
                'expected': f'{desc} 的 cut 值和執行時間',
                'category': 'comparison'
            })
        
        # 範例 3：自定義 spin 向量
        self.examples.append({
            'name': '🎯 自定義測試 - Spin 向量',
            'description': '測試特定的 spin 向量 [1,-1,1]',
            'cmd': [
                sys.executable, 'maxcut_main.py',
                '--graph_file', 'test_graphs/defined_3nodes.txt',
                '--algorithm', 'goemans_williamson',
                '--spin_vector', '1,-1,1',
                '--iterations', '1',
                '--disable_hardware_modeling'
            ],
            'expected': '直接計算 spin [1,-1,1] 的 cut 值 = 3',
            'category': 'custom'
        })
        
        # 範例 4：硬體設定比較
        configs = [
            ('SimConfig.ini', '標準設定'),
            ('SimConfig_MaxCut.ini', 'Max Cut 最佳化設定')
        ]
        
        for config, desc in configs:
            if os.path.exists(config):
                self.examples.append({
                    'name': f'⚙️ 硬體設定 - {desc}',
                    'description': f'使用 {desc} 執行硬體建模',
                    'cmd': [
                        sys.executable, 'maxcut_main.py',
                        '--hardware_description', config,
                        '--graph_file', 'test_graphs/random_10nodes.txt',
                        '--algorithm', 'rram_psa',
                        '--psav_trials', '5',
                        '--disable_layer_output'
                    ],
                    'expected': f'{desc} 的硬體效能指標',
                    'category': 'hardware'
                })
        
        # 範例 5：大圖測試
        large_graphs = [
            ('test_graphs/random_50nodes.txt', '50 節點'),
            ('test_graphs/grid_4x4.txt', '4x4 網格'),
            ('test_graphs/complete_8nodes.txt', '8 節點完全圖')
        ]
        
        for graph, desc in large_graphs:
            self.examples.append({
                'name': f'📊 大圖測試 - {desc}',
                'description': f'測試 {desc} 的求解效能',
                'cmd': [
                    sys.executable, 'maxcut_main.py',
                    '--graph_file', graph,
                    '--algorithm', 'rram_psa',
                    '--psav_trials', '10',
                    '--psav_cycles', '100',
                    '--disable_module_output',
                    '--disable_layer_output'
                ],
                'expected': f'{desc} 的 cut 值和執行時間',
                'category': 'performance'
            })
        
        # 範例 6：參數調優
        param_tests = [
            ('低參數', ['--psav_trials', '3', '--psav_cycles', '20']),
            ('中參數', ['--psav_trials', '10', '--psav_cycles', '100']),
            ('高參數', ['--psav_trials', '20', '--psav_cycles', '200'])
        ]
        
        for param_name, params in param_tests:
            self.examples.append({
                'name': f'🔧 參數調優 - {param_name}',
                'description': f'測試 {param_name} 設定的效果',
                'cmd': [
                    sys.executable, 'maxcut_main.py',
                    '--graph_file', 'test_graphs/random_20nodes.txt',
                    '--algorithm', 'rram_psa'
                ] + params + [
                    '--disable_module_output',
                    '--disable_layer_output'
                ],
                'expected': f'{param_name} 的品質與時間權衡',
                'category': 'tuning'
            })
    
    def run_example(self, example, show_output=True):
        """執行單個範例"""
        print(f"\n{'='*80}")
        print(f"🚀 {example['name']}")
        print(f"{'='*80}")
        print(f"📝 說明：{example['description']}")
        print(f"💡 預期：{example['expected']}")
        print(f"🔧 命令：{' '.join(example['cmd'])}")
        print("-" * 80)
        
        # 檢查必要檔案
        graph_file = None
        for i, arg in enumerate(example['cmd']):
            if arg == '--graph_file' and i + 1 < len(example['cmd']):
                graph_file = example['cmd'][i + 1]
                break
        
        if graph_file and not os.path.exists(graph_file):
            print(f"⚠️  跳過：找不到圖檔案 {graph_file}")
            return False
        
        try:
            start_time = time.time()
            result = subprocess.run(
                example['cmd'], 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 分鐘超時
            )
            end_time = time.time()
            
            if result.returncode == 0:
                print("✅ 執行成功")
                if show_output:
                    # 只顯示關鍵結果
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in 
                              ['cut value', 'cut 值', 'maximum cut', 'average cut', 
                               'total time', '總模擬時間', 'rram psa', 'final result']):
                            print(f"📊 {line}")
                else:
                    print("📄 輸出已隱藏（使用 --verbose 顯示完整輸出）")
                
                print(f"⏱️  執行時間：{end_time - start_time:.2f} 秒")
                return True
                
            else:
                print("❌ 執行失敗")
                print("錯誤：", result.stderr[:500])  # 只顯示前 500 字元
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ 執行超時（2分鐘）")
            return False
        except Exception as e:
            print(f"💥 執行出錯：{e}")
            return False
    
    def run_category(self, category, show_output=True):
        """執行特定分類的範例"""
        examples = [ex for ex in self.examples if ex['category'] == category]
        
        print(f"\n{'🎯 執行分類':<20}: {category}")
        print(f"{'📊 範例數量':<20}: {len(examples)}")
        
        results = []
        for example in examples:
            success = self.run_example(example, show_output)
            results.append((example['name'], success))
        
        return results
    
    def run_all(self, categories=None, show_output=False):
        """執行所有或指定分類的範例"""
        if categories is None:
            categories = ['basic', 'comparison', 'custom', 'hardware', 'performance', 'tuning']
        
        all_results = []
        
        for category in categories:
            results = self.run_category(category, show_output)
            all_results.extend(results)
        
        # 輸出總結
        print(f"\n{'='*80}")
        print("📋 執行總結")
        print(f"{'='*80}")
        
        success_count = sum(1 for _, success in all_results if success)
        total_count = len(all_results)
        
        for name, success in all_results:
            status = "✅" if success else "❌"
            print(f"{status} {name}")
        
        print(f"\n🎯 總計：{success_count}/{total_count} 範例成功執行")
        
        if success_count == total_count:
            print("🎉 所有範例都執行成功！")
        else:
            print("⚠️  部分範例執行失敗，請檢查錯誤訊息")

def main():
    """主函式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MNSIM Max Cut 執行範例')
    parser.add_argument('--category', '-c', 
                       choices=['basic', 'comparison', 'custom', 'hardware', 'performance', 'tuning'],
                       help='只執行特定分類的範例')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='顯示完整輸出')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有範例')
    
    args = parser.parse_args()
    
    # 初始化
    demo = MaxCutDemo()
    
    # 列出範例
    if args.list:
        print("📋 可用範例：")
        categories = {}
        for ex in demo.examples:
            if ex['category'] not in categories:
                categories[ex['category']] = []
            categories[ex['category']].append(ex['name'])
        
        for cat, examples in categories.items():
            print(f"\n🏷️  {cat.upper()}:")
            for ex in examples:
                print(f"   • {ex}")
        return
    
    print("🚀 MNSIM Max Cut 執行範例")
    print("="*80)
    print("這個腳本會執行各種 Max Cut 求解範例")
    
    # 確保測試圖存在
    if not os.path.exists('test_graphs'):
        print("📊 建立測試圖...")
        subprocess.run([sys.executable, 'create_sample_graphs.py'], 
                      capture_output=True)
    
    # 執行範例
    if args.category:
        demo.run_category(args.category, args.verbose)
    else:
        # 執行精選範例
        selected_categories = ['basic', 'comparison', 'custom']
        demo.run_all(selected_categories, args.verbose)
        
        print(f"\n💡 提示：")
        print(f"   • 執行所有範例：python {sys.argv[0]} --category performance")
        print(f"   • 顯示完整輸出：python {sys.argv[0]} --verbose")
        print(f"   • 列出所有範例：python {sys.argv[0]} --list")

if __name__ == "__main__":
    main()
