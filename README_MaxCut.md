# MNSIM Max Cut 求解器

基於 RRAM crossbar 的 Max Cut 問題求解系統，整合 MNSIM 硬體模擬器與多種優化演算法。

## 📋 目錄

- [系統架構](#系統架構)
- [安裝與環境](#安裝與環境)
- [快速開始](#快速開始)
- [演算法說明](#演算法說明)
- [檔案結構](#檔案結構)
- [使用範例](#使用範例)
- [參數說明](#參數說明)
- [硬體建模](#硬體建模)

## 🏗️ 系統架構

本系統將傳統的神經網路模擬器 MNSIM 改造為 Max Cut 問題求解器，核心創新：

1. **RRAM 權重映射**：將圖的加權鄰接矩陣直接映射到 RRAM crossbar 的電阻值
2. **多演算法支援**：內建啟發式、GPU pSA、RRAM pSA 等多種求解方法
3. **硬體建模整合**：保留完整的延遲、面積、功耗、能量分析能力

```
圖檔案 → MaxCutInterface → RRAM 量化 → 演算法求解 → 硬體效能分析
   ↓           ↓              ↓           ↓            ↓
 .txt/.csv   權重矩陣      電阻矩陣    Cut 值結果   延遲/功耗/面積
```

## 🔧 安裝與環境

### 基本需求

- Python 3.6+
- NumPy
- NetworkX
- ConfigParser

### GPU pSA 額外需求（可選）

- CUDA 12.2+
- PyCUDA 2022.1+
- 對應的 GPU 驅動

### 安裝步驟

1. 克隆專案：
```bash
git clone <repository-url>
cd MNSIM-2.0-master
```

2. 安裝 Python 依賴：
```bash
pip install numpy networkx configparser
```

3. （可選）安裝 GPU 支援：
```bash
pip install pycuda
```

## 🚀 快速開始

### 1. 建立測試圖

```bash
# 產生多種測試圖檔案
python create_sample_graphs.py
```

### 2. 執行 Max Cut 求解

```bash
# 使用內建啟發式演算法
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm goemans_williamson

# 使用 RRAM pSA 演算法
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa --psav_trials 10 --psav_cycles 100

# 使用 GPU pSA（需要 CUDA 環境）
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm psav_psa --psav_trials 10 --psav_cycles 100
```

### 3. 查看結果

程式會輸出：
- Max Cut 求解結果（cut 值、分割方案）
- 硬體效能指標（延遲、面積、功耗、能量）
- 執行時間統計

## 🧮 演算法說明

### 1. Goemans-Williamson 啟發式 (`goemans_williamson`)
- **原理**：隨機向量投影與閾值分割
- **特點**：快速、適合初步測試
- **硬體**：使用 RRAM crossbar 執行 MVM

### 2. RRAM pSA (`rram_psa`) ⭐ 推薦
- **原理**：基於 RRAM 的機率位元退火
- **特點**：完全整合 MNSIM 硬體模型
- **優勢**：
  - 用 RRAM crossbar 加速局部場計算
  - CPU 執行，無需 GPU 環境
  - 保留完整硬體建模能力

### 3. GPU pSA (`psav_psa`)
- **原理**：GPU 加速的機率位元退火
- **特點**：高效能，適合大規模問題
- **需求**：CUDA 環境

### 4. 貪婪演算法 (`greedy`)
- **原理**：隨機初始化 + 局部改進
- **特點**：簡單快速

## 📁 檔案結構

```
MNSIM-2.0-master/
├── maxcut_main.py                    # 主程式入口
├── create_sample_graphs.py           # 測試圖產生器
├── SimConfig.ini                     # MNSIM 硬體設定檔
├── SimConfig_MaxCut.ini              # Max Cut 最佳化設定檔
├── README_MaxCut.md                  # 本說明文件
├── MNSIM/
│   └── Interface/
│       ├── maxcut_interface.py       # Max Cut 問題介面層
│       ├── rram_psa.py              # RRAM pSA 實作
│       └── psav_adapter.py          # GPU pSA 適配器
├── GPU-pSAv-main/                   # GPU pSA 子模組
│   ├── gpu_MAXCUT.py
│   └── psa_annealing_kernel.cu
└── test_graphs/                     # 測試圖檔案目錄
    ├── defined_3nodes.txt           # 3 節點測試圖
    ├── random_10nodes.txt           # 10 節點隨機圖
    └── ...
```

## 💡 使用範例

### 範例 1：3 節點測試圖

**圖檔案內容** (`test_graphs/defined_3nodes.txt`)：
```
# node1 node2 weight
0 1 1
1 2 2
```

**對應矩陣**：
```
G = [[0, 1, 0],
     [1, 0, 2],
     [0, 2, 0]]
```

**執行命令**：
```bash
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa --psav_trials 5 --psav_cycles 50
```

**預期輸出**：
```
RRAM pSA 初始化完成：3 節點，1 個 crossbar
退火參數：I0_min=0.100, I0_max=10.000, beta=0.500
Trial 1/5
  Cut value: 3, Time: 12.34 ms
...
==================================================
RRAM pSA 結果統計
==================================================
節點數: 3
試驗次數: 5
平均 cut 值: 3.00
最大 cut 值: 3
```

### 範例 2：使用自定義 spin 向量

```bash
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm goemans_williamson --spin_vector "1,-1,1" --iterations 1
```

這會直接測試 spin 向量 [1,-1,1] 的 cut 值。

### 範例 3：大圖測試

```bash
# 先產生大圖
python create_sample_graphs.py

# 使用 RRAM pSA 求解
python maxcut_main.py --graph_file test_graphs/random_50nodes.txt --algorithm rram_psa --psav_trials 20 --psav_cycles 200
```

## ⚙️ 參數說明

### 通用參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--graph_file` | 圖檔案路徑 | 必填 |
| `--algorithm` | 演算法選擇 | `RRAM psa` |
| `--hardware_description` | 硬體設定檔 | `SimConfig.ini` |
| `--disable_hardware_modeling` | 關閉硬體建模 | `False` |

### pSA 專用參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--psav_trials` | 試驗次數 | `50` |
| `--psav_cycles` | 退火週期數 | `200` |
| `--psav_tau` | 每溫度點更新次數 | `1` |
| `--psav_param` | 參數類型 (1或2) | `2` |
| `--psav_thread` | GPU 執行緒數 | `32` |
| `--psav_gpu` | GPU 裝置 ID | `0` |

### 圖檔案格式

支援兩種格式：

1. **邊清單格式** (.txt)：
```
# 註解行
node1 node2 weight
0 1 1.5
1 2 2.0
```

2. **矩陣格式** (.csv)：
```
0,1.5,0
1.5,0,2.0
0,2.0,0
```

## 🔬 硬體建模

### RRAM 設定 (SimConfig.ini)

關鍵參數：
```ini
[Device level]
Device_Level = 8                    # 8 個電阻等級
Device_Resistance = 1e6,5e5,2.5e5,1.25e5,6.25e4,3.125e4,1.56e4,1e4
Read_Voltage = 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35

[Crossbar level]
Xbar_Size = 128,128                 # Crossbar 尺寸
```

### 權重映射流程

1. **正規化**：圖權重 → [0,1]
2. **量化**：[0,1] → {0,1,2,...,7}
3. **電阻映射**：等級 → 實際電阻值
4. **電導計算**：G = 1/R
5. **MVM 運算**：I = G × V

### 硬體指標輸出

程式會自動計算並輸出：
- **延遲**：crossbar 讀取延遲、ADC/DAC 延遲
- **面積**：crossbar 面積、周邊電路面積
- **功耗**：讀取功耗、寫入功耗
- **能量**：總能量消耗

## 🛠️ 進階用法

### 自定義硬體設定

1. 複製 `SimConfig.ini` 為 `MyConfig.ini`
2. 修改關鍵參數
3. 執行：
```bash
python maxcut_main.py --hardware_description MyConfig.ini --graph_file your_graph.txt --algorithm rram_psa
```

### 批次測試

```bash
# 執行所有測試圖
python run_maxcut_tests.py
```

### 效能比較

```bash
# 比較不同演算法
for alg in goemans_williamson rram_psa psav_psa; do
    echo "Testing $alg"
    python maxcut_main.py --graph_file test_graphs/random_20nodes.txt --algorithm $alg --psav_trials 10
done
```

## 🐛 故障排除

### 常見問題

1. **找不到圖檔案**
   - 確認檔案路徑正確
   - 執行 `python create_sample_graphs.py` 產生測試圖

2. **GPU pSA 執行失敗**
   - 檢查 CUDA 環境：`nvidia-smi`
   - 確認 PyCUDA 安裝：`python -c "import pycuda.autoinit"`

3. **記憶體不足**
   - 減少 trials 和 cycles 參數
   - 使用較小的測試圖

4. **硬體建模錯誤**
   - 檢查 `SimConfig.ini` 語法
   - 使用 `--disable_hardware_modeling` 跳過硬體分析

### 除錯模式

```bash
# 開啟詳細輸出
python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa --psav_trials 1 --psav_cycles 10
```

