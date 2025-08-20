# 🚀 MNSIM Max Cut - 快速執行指南

## 一鍵開始

```bash
# 自動測試所有功能
python quick_start.py
```

## 手動執行步驟

### 1️⃣ 建立測試圖
```bash
python create_sample_graphs.py
```

### 2️⃣ 執行 Max Cut 求解

**基本用法**：
```bash
# RRAM pSA (推薦)
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa

# 內建啟發式
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm goemans_williamson

# GPU pSA (需要 CUDA)
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm psav_psa
```

**進階參數**：
```bash
# 調整 pSA 參數
python maxcut_main.py --graph_file test_graphs/random_10nodes.txt --algorithm rram_psa --psav_trials 20 --psav_cycles 200

# 使用最佳化硬體設定
python maxcut_main.py --hardware_description SimConfig_MaxCut.ini --graph_file test_graphs/random_20nodes.txt --algorithm rram_psa

# 測試特定 spin 向量
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm goemans_williamson --spin_vector "1,-1,1"
```

## 🎯 3 節點範例

**圖檔內容** (`test_graphs/defined_3nodes.txt`)：
```
0 1 1
1 2 2
```

**執行**：
```bash
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa --psav_trials 5
```

**預期輸出**：
- 最佳 cut 值：3
- 分割方案：{0,2} vs {1}
- 硬體效能指標

## 📊 演算法比較

| 演算法 | 特點 | 適用場景 | 執行時間 |
|--------|------|----------|----------|
| `rram_psa` | RRAM 硬體加速 | 推薦，完整硬體建模 | 中等 |
| `goemans_williamson` | 快速啟發式 | 初步測試 | 快 |
| `psav_psa` | GPU 加速 | 大規模問題 | 快 |
| `greedy` | 簡單貪婪 | 基準比較 | 最快 |

## 🔧 常用參數

- `--psav_trials 10`：執行 10 次試驗
- `--psav_cycles 100`：每次試驗 100 個退火週期
- `--disable_hardware_modeling`：關閉硬體建模（加速）
- `--disable_module_output`：簡化輸出

## 📁 重要檔案

- `maxcut_main.py`：主程式
- `SimConfig.ini`：硬體設定檔
- `test_graphs/`：測試圖目錄
- `README_MaxCut.md`：完整文件

## 🐛 故障排除

**圖檔案找不到**：
```bash
python create_sample_graphs.py  # 重新產生
```

**GPU 錯誤**：
```bash
# 改用 CPU 版本
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa
```

**記憶體不足**：
```bash
# 減少參數
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa --psav_trials 3 --psav_cycles 50
```
