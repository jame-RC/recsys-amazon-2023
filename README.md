# 🌌 基于 Amazon Reviews 2023 的多模型 Top-K 商品序列推荐系统

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-shadow)](https://www.python.org/)
[![PyTorch 1.12](https://img.shields.io/badge/pytorch-1.12.1-orange.svg?style=flat-shadow)](https://pytorch.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--v4--flash-green.svg?style=flat-shadow)](https://api-docs.deepseek.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-shadow)](https://opensource.org/licenses/MIT)

本仓库提供了一个用于 **Amazon Reviews 2023** 数据集上 Top-K 商品序列推荐系统的现代化工业级实现方案。项目涵盖了从**全局热度、传统协同过滤、矩阵分解、深度自注意力 Transformer 序列模型 (SASRec)**，到**大语言模型重排序 (LLM Zero-Shot & Few-Shot)** 的全系列算法对比，并包含一个**直接在浏览器双击即可运行的无 CORS 限制可视化交互演示大屏**。

> [!NOTE]
> 🏆 **核心技术成果**：
> 1. **大语言模型常识先验红利**：在最庞大、稀疏的 `CDs_and_Vinyl` 唱片数据集上，**LLM Zero-Shot 取得了 0.0373 的 NDCG@10 成绩，超越传统最强协同矩阵模型 BPR (0.0281) 达 32.7%！**
> 2. **SASRec 极致性能跃升**：彻底修复了 SASRec 的 Padding 偏移漏洞和提取索引 Bug，重构了**全 GPU 向量化批量预测**，将评估速度**暴提升 11.5 倍以上**（5万样本仅耗时 12 秒），显存开销非常安全。

---

## 📊 实验成绩横向大比拼 (Test NDCG@10)

以下表格汇总了所有 6 种算法在 Amazon 三大垂直品类（`Industrial_and_Scientific`、`Musical_Instruments`、`CDs_and_Vinyl`）上的真实评测结果：

| 模型架构 (Model) | Industrial | Musical | CDs | 三品类平均 | 特性优势 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Popularity** (全局热度基线) | 0.0081 | 0.0128 | 0.0009 | 0.0073 | 极速响应，无需个性化 |
| **ItemCF** (物品协同过滤) | 0.0050 | 0.0023 | — | 0.0037 | 捕获物品显式共现 |
| **BPR** 🏆 (矩阵分解) | 0.0155 | **0.0211** | 0.0281 | 0.0216 | 隐向量关联，成对排序损失极强 |
| **SASRec** (Ours) (Transformer序列) | 0.0086 | 0.0095 | 0.0017 | 0.0066 | 多头自注意力捕获时序兴趣 |
| **LLM Zero-Shot** (大模型零样本) | 0.0154 | 0.0083 | **0.0373** | 0.0203 | 品类人设先验，语义匹配极其强悍 |
| **LLM Few-Shot** (大模型少样本) | **0.0187** | 0.0085 | 0.0357 | **0.0210** | 相似范例上下文引导，示范效应强 |

---

## 📂 项目模块划分

```
├── data/                  # 数据下载和分类库存目录
├── src/
│   ├── data/              # 数据预处理与加载
│   │   ├── dataset.py     # 规范化 Left-Last-Out 数据划分与 Padding 数据集
│   │   └── vocab.py       # ID 双向映射词表
│   ├── models/            # 推荐模型实现
│   │   ├── base.py        # 统一规范推荐基类 (BaseRecommender)
│   │   ├── pop.py         # Popularity
│   │   ├── item_cf.py     # ItemCF
│   │   ├── bpr.py         # BPR (Bayesian Personalized Ranking)
│   │   ├── sasrec.py      # 极致重构优化的自注意力序列网络 (SASRec)
│   │   └── llm_ranker.py  # LLM 重排序器 (对接 DeepSeek)
│   ├── evaluation/        # 评估指标
│   │   ├── metrics.py     # 高性能向量化 NDCG@10, Hit@10, MRR@10 实现
│   │   └── evaluator.py   # 全流程流式评测评估器
│   └── llm/               # LLM 模块
│       ├── prompt.py      # 适配品类专业知识与格式强控制的 System/User Prompts
│       └── api.py         # 支持高并发、指数退避重试的 DeepSeek 接口层
├── scripts/
│   ├── train.py           # 传统模型快速单步训练
│   ├── evaluate.py        # 传统模型单步评估
│   ├── evaluate_llm_sampled.py  # 大模型高并发多线程重排序随机评测
│   ├── update_llm_metrics.py    # 一键数据同步回写大屏与 MD 报告
│   ├── convert_report.py  # markdown 自动向高级 Word 文档格式 (.docx) 编译
│   └── run_all.py         # 批量传统实验流跑通
├── results/               # 指标 JSON 库及大屏数据
│   └── dashboard_data.js  # 可视化大屏的数据载体
├── dashboard.html         # 🌟 零跨域阻碍可视化大屏主页
├── 实验报告.md             # 学术级 Markdown 实验报告
└── 实验报告.docx           # 完美排版、中英文对照的 Word 实验报告
```

---

## ⚡ 极速起步与运行指南

> [!WARNING]
> 本项目大模型重排序评测基于 **DeepSeek API** 真实请求完成。请确保在运行前已配置环境变量 `DEEPSEEK_API_KEY` 或通过命令行参数传入正确的 Sk 密钥。

### 1. 依赖环境安装
```bash
pip install -r requirements.txt
```

### 2. 运行传统模型批量流水线
直接运行以下指令，全自动完成数据集下载、切分并依次运行 Popularity, ItemCF, BPR, SASRec 的训练与评估：
```bash
python scripts/run_all.py
```

### 3. 运行 500 用户量级的大模型真实评测 (以 Industrial 类别 Few-Shot 为例)
```bash
python scripts/evaluate_llm_sampled.py --category Industrial_and_Scientific --mode fewshot --sample-size 500 --max-workers 15 --api-key YOUR_DEEPSEEK_KEY
```

### 4. 一键回写同步并编译 Word 报告
当所有的评测完成后，运行以下指令直接更新大屏指标数据，并重新将 Markdown 自动转换为排版精美、表格严整的 Word 报告：
```bash
# 1. 动态写回指标
python scripts/update_llm_metrics.py

# 2. 编译生成全新 Word 文档
python scripts/convert_report.py
```

---

## 📺 现代化可视化大屏 (Dashboard)

我们精心设计了一个高品质的可视化演示大屏，让算法对比和重排序机制直观可见。

> [!IMPORTANT]
> **🚀 免 CORS 跨域痛点启动**：
> 针对浏览器直接加载本地 JSON 文件会触发安全沙箱限制（CORS Error）的痛点，我们巧妙地将评测数据和测试集用户画像序列化打包写入 [results/dashboard_data.js](file:///c:/Users/Jame.RC/Desktop/推荐系统/results/dashboard_data.js) 全局变量中。
> **您无需在本地搭建任何 Web 服务器，直接在 Windows 中双击 [dashboard.html](file:///c:/Users/Jame.RC/Desktop/推荐系统/dashboard.html) 即可立刻在任意浏览器中完美进行流畅动画交互！**

### 🌟 交互界面特色亮点：
1.  **📊 多维度 KPI 卡片与柱状图**：动态呈现 `NDCG@10`, `Hit@10`, `MRR@10` 核心评测指标。柱状高度支持在三个行业品类之间流畅、平滑切换。
2.  **🎯 交互式推荐命中演示**：支持从下拉菜单中任意挑选 5 个具有不同历史行为密度的真实测试集用户，动态绘制其**时序卡片历史足迹**，并呈现第 $N$ 次真实交互商品。
3.  **💫 呼吸高亮特效**：侧边一字排开 4 种模型推荐的 Top-10 列表。当模型推荐商品中包含了用户的真实目标（Hit）时，商品卡片将立刻触发 **霓虹绿光边界 + 脉冲呼吸动画 + "🎯 HIT 命中!" 徽章**，视觉表现极强，完美诠释了 “Hit@10” 的学术原理。
