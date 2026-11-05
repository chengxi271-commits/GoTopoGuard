# GoTopoGuard

### 面向神经围棋引擎罕见结构性失误的拓扑感知风险监测

**PRICAI 2026** · [Yixi Chen](https://orcid.org/0009-0004-7084-0395) · 中山大学

[English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

[代码与数据](https://github.com/chengxi271-commits/GoTopoGuard/releases/tag/v0.2.0) ·
[复现指南](docs/REPRODUCIBILITY.md) · [特征定义](docs/FEATURES.md) · [引用方式](#引用)

**当棋盘结构变得不寻常时，强大的围棋引擎会遗漏什么？**
GoTopoGuard 从棋子连通块、气、封闭空域等棋盘图结构出发，研究这类局面的风险信号。
这些信号用于 TopoRisk 外部监测器和搜索预算校准，无需重新训练底层围棋引擎。

本仓库提供 PRICAI 论文的实现与实验记录，包括拓扑特征提取、基准构建、风险检测与提前预警、
搜索预算校准，以及预算、网络和跨引擎分析。通过发布的特征与结果包，可以直接使用原实验测量值，
不必先下载引擎或重跑搜索。

## 快速开始

需要 Python 3.10 或以上版本；建议使用 Python 3.12 运行本次发布所附的分析环境。

```bash
git clone https://github.com/chengxi271-commits/GoTopoGuard.git
cd GoTopoGuard
git checkout v0.2.0
python -m pip install ".[reproduce]"
python examples/minimal_features.py
python scripts/reproduce.py --suite summary
```

最后一条命令将论文报告的数值与仓库中的记录核对，并在 `reproduced/summary/` 下生成 JSON 和 CSV。
这一步不需要 GPU 或额外下载。若只使用拓扑提取器，执行 `pip install .` 即可；提取器没有第三方运行时依赖。

### 使用拓扑特征

```python
from gotopoguard import extract_topology_features, parse_board

board = parse_board([".....", ".BBB.", ".B.B.", ".BBB.", "....."])
features = extract_topology_features(board)
print(features["enclosed_empty_region_count"])  # 1
print(features["black_max_liberty_count"])      # 13
```

`B`、`W` 和 `.` 分别表示黑子、白子和空点。API 还提供连通块成员、精确的气集合、空域边界和割点。
各字段的含义见[特征定义](docs/FEATURES.md)。

## 复现实验分析

下载 **44.8 MB** 的实测特征与结果包，然后选择分析任务：

```bash
python scripts/fetch_artifacts.py
python scripts/reproduce.py --suite guard-s09
python scripts/reproduce.py --suite detection
python scripts/reproduce.py --suite early-warning
python scripts/reproduce.py --suite controls
```

这些命令基于保存的测量值重新拟合模型或计算统计量，不启动围棋引擎。历史结果与新生成结果分别保存。
本次发布环境中，部分检测和对照指标重新拟合后存在差异；原实验的完整依赖锁定文件尚未恢复。
[复现说明](docs/REPRODUCIBILITY.md)介绍了这一边界，以及操作点、自然流校准和配对显著性检验等其他任务。

需要对自己的棋盘评分时，可以拟合原实现中的全数据诊断模型：

```bash
python scripts/train_models.py
python scripts/score_board.py examples/board19.json
```

示例使用人工构造的 19×19 棋盘，不是基准测试案例。这些新拟合的模型用于推断，其训练集输出不作为留出评估结果。

## 主要结果

以下数值来自 PRICAI 论文：

| 实验 | 结果 |
|---|---|
| B3 棋盘 + 拓扑特征，家族留出检测 | AUROC **0.9577**，AUPRC **0.9044** |
| B5 加入搜索与对称变换特征 | AUPRC **0.9250** |
| 20 局留出对抗棋谱的提前预警 | 全部触发预警；平均最早采样提前量 **80 手** |
| 1,144 个压力局面上的 S09 与均匀 1024 visits | 一致率 **0.8470 vs. 0.8427**；平均 visits **875 vs. 1,024** |

严格对照模型的总体检测表现与加入拓扑特征的模型相当或更好；S09 与均匀搜索的差异未达到统计显著性
（McNemar *p* = 0.4996）。这些结果支持诊断性监测与搜索策略校准，并不证明已经实现实战防御。

## 仓库导航

| 路径 | 内容 |
|---|---|
| `src/gotopoguard/` | 独立特征 API 与命令行接口 |
| `experiments/scripts/` | 原始 R 系列数据构建、模型训练评估和引擎查询脚本 |
| `analysis/` | 严格对照、机制留出与配对统计 |
| `experiments/configs/` | 论文实验设置与固定的上游版本 |
| `artifacts/` | 精简历史结果与可下载数据包清单 |
| `scripts/` | 下载、复现、训练与评分入口 |
| `tests/` | 棋盘、回放、模型与评估测试 |

如需重建输入或自行运行引擎，请阅读[数据与模型](docs/DATA_AND_MODELS.md)及[引擎配置](docs/ENGINES.md)。
外部二进制文件、模型权重和第三方 SGF 棋谱集合从上游获取，不在本仓库镜像分发。

## 引用

论文已被 PRICAI 2026 录用。正式出版信息核实后会更新以下临时引用：

```bibtex
@misc{chen2026gotopoguard,
  author = {Chen, Yixi},
  title = {{GoTopoGuard}: Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines},
  year = {2026},
  note = {Accepted at PRICAI 2026},
  url = {https://github.com/chengxi271-commits/GoTopoGuard}
}
```

使用实现时，请同时注明版本号或提交。软件引用信息见 [`CITATION.cff`](CITATION.cff)。

## 许可证与联系

原创代码采用 [Apache-2.0](LICENSE)。引擎补丁遵循对应上游许可证，详见[第三方声明](docs/THIRD_PARTY_NOTICES.md)。
论文及外部数据、模型适用各自条款。

欢迎通过 [Issues](https://github.com/chengxi271-commits/GoTopoGuard/issues) 提问或报告可复现的问题。
研究交流：[chengxi271@gmail.com](mailto:chengxi271@gmail.com)。
