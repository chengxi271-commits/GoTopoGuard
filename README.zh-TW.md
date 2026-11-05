# GoTopoGuard

### 面向神經圍棋引擎罕見結構性失誤的拓撲感知風險監測

**PRICAI 2026** · [Yixi Chen](https://orcid.org/0009-0004-7084-0395) · 中山大學

[English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

[程式碼與資料](https://github.com/chengxi271-commits/GoTopoGuard/releases/tag/v0.2.0) ·
[重現指南](docs/REPRODUCIBILITY.md) · [特徵定義](docs/FEATURES.md) · [引用方式](#引用)

**當棋盤結構變得不尋常時，強大的圍棋引擎會遺漏甚麼？**
GoTopoGuard 從棋子連通塊、氣、封閉空域等棋盤圖結構出發，研究這類局面的風險訊號。
這些訊號用於 TopoRisk 外部監測器和搜尋預算校準，無需重新訓練底層圍棋引擎。

本儲存庫提供 PRICAI 論文的實作與實驗紀錄，包括拓撲特徵擷取、基準建構、風險偵測與提前預警、
搜尋預算校準，以及預算、網路和跨引擎分析。透過發布的特徵與結果資料包，可以直接使用原實驗測量值，
不必先下載引擎或重新執行搜尋。

## 快速開始

需要 Python 3.10 或以上版本；建議使用 Python 3.12 執行本次發布所附的分析環境。

```bash
git clone https://github.com/chengxi271-commits/GoTopoGuard.git
cd GoTopoGuard
git checkout v0.2.0
python -m pip install ".[reproduce]"
python examples/minimal_features.py
python scripts/reproduce.py --suite summary
```

最後一條指令將論文報告的數值與儲存庫中的紀錄核對，並在 `reproduced/summary/` 下產生 JSON 和 CSV。
這一步不需要 GPU 或額外下載。若只使用拓撲擷取器，執行 `pip install .` 即可；擷取器沒有第三方執行期依賴。

### 使用拓撲特徵

```python
from gotopoguard import extract_topology_features, parse_board

board = parse_board([".....", ".BBB.", ".B.B.", ".BBB.", "....."])
features = extract_topology_features(board)
print(features["enclosed_empty_region_count"])  # 1
print(features["black_max_liberty_count"])      # 13
```

`B`、`W` 和 `.` 分別表示黑子、白子和空點。API 亦提供連通塊成員、精確的氣集合、空域邊界和割點。
各欄位的含義見[特徵定義](docs/FEATURES.md)。

## 重現實驗分析

下載 **44.8 MB** 的實測特徵與結果資料包，然後選擇分析任務：

```bash
python scripts/fetch_artifacts.py
python scripts/reproduce.py --suite guard-s09
python scripts/reproduce.py --suite detection
python scripts/reproduce.py --suite early-warning
python scripts/reproduce.py --suite controls
```

這些指令基於保存的測量值重新擬合模型或計算統計量，不啟動圍棋引擎。歷史結果與新產生的結果分開保存。
本次發布環境中，部分偵測和對照指標重新擬合後存在差異；原實驗的完整依賴鎖定檔案尚未恢復。
[重現說明](docs/REPRODUCIBILITY.md)介紹了這項界限，以及操作點、自然資料流校準和配對顯著性檢驗等其他任務。

需要對自己的棋盤評分時，可以擬合原實作中的全資料診斷模型：

```bash
python scripts/train_models.py
python scripts/score_board.py examples/board19.json
```

範例使用人工構造的 19×19 棋盤，不是基準測試案例。這些新擬合的模型用於推論，其訓練集輸出不作為留出評估結果。

## 主要結果

以下數值來自 PRICAI 論文：

| 實驗 | 結果 |
|---|---|
| B3 棋盤 + 拓撲特徵，家族留出偵測 | AUROC **0.9577**，AUPRC **0.9044** |
| B5 加入搜尋與對稱變換特徵 | AUPRC **0.9250** |
| 20 局留出對抗棋譜的提前預警 | 全部觸發預警；平均最早取樣提前量 **80 手** |
| 1,144 個壓力局面上的 S09 與均勻 1024 visits | 一致率 **0.8470 vs. 0.8427**；平均 visits **875 vs. 1,024** |

嚴格對照模型的整體偵測表現與加入拓撲特徵的模型相當或更好；S09 與均勻搜尋的差異未達統計顯著性
（McNemar *p* = 0.4996）。這些結果支持診斷性監測與搜尋策略校準，並不證明已經實現實戰防禦。

## 儲存庫導覽

| 路徑 | 內容 |
|---|---|
| `src/gotopoguard/` | 獨立特徵 API 與命令列介面 |
| `experiments/scripts/` | 原始 R 系列資料建構、模型訓練評估和引擎查詢腳本 |
| `analysis/` | 嚴格對照、機制留出與配對統計 |
| `experiments/configs/` | 論文實驗設定與固定的上游版本 |
| `artifacts/` | 精簡歷史結果與可下載資料包清單 |
| `scripts/` | 下載、重現、訓練與評分入口 |
| `tests/` | 棋盤、回放、模型與評估測試 |

如需重建輸入或自行執行引擎，請閱讀[資料與模型](docs/DATA_AND_MODELS.md)及[引擎設定](docs/ENGINES.md)。
外部二進位檔案、模型權重和第三方 SGF 棋譜集合從上游取得，不在本儲存庫鏡像分發。

## 引用

論文已獲 PRICAI 2026 錄用。正式出版資訊核實後會更新以下暫用引用：

```bibtex
@misc{chen2026gotopoguard,
  author = {Chen, Yixi},
  title = {{GoTopoGuard}: Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines},
  year = {2026},
  note = {Accepted at PRICAI 2026},
  url = {https://github.com/chengxi271-commits/GoTopoGuard}
}
```

使用實作時，請同時註明版本號或提交。軟件引用資訊見 [`CITATION.cff`](CITATION.cff)。

## 授權與聯絡

原創程式碼採用 [Apache-2.0](LICENSE)。引擎補丁遵循相應上游授權條款，詳見[第三方聲明](docs/THIRD_PARTY_NOTICES.md)。
論文及外部資料、模型適用各自條款。

歡迎透過 [Issues](https://github.com/chengxi271-commits/GoTopoGuard/issues) 提問或回報可重現的問題。
研究交流：[chengxi271@gmail.com](mailto:chengxi271@gmail.com)。
