# CLAUDE.md — 推荐系统项目开发指南

## 📋 项目概述 (Project Overview)

推荐系统课程大作业：基于 **Amazon Reviews 2023** 数据集的 Top-K 商品序列推荐系统。

*   **核心任务**：根据用户历史交互序列（`history`），精确预测下一个最可能交互的商品（`target`），主要衡量指标为 **NDCG@10**。
*   **数据集**：Amazon Reviews 2023，采用 5-Core 过滤版本，包含三个垂直行业品类：`Industrial_and_Scientific`、`Musical_Instruments` 和 `CDs_and_Vinyl`。
*   **数据划分**：采用 **Leave-Last-Out** 规范。前 $N-2$ 次交互作为**训练集**，第 $N-1$ 次交互作为**验证集**，最后一次 $N$ 交互作为**测试集**。

---

## 🛠️ 技术栈与环境 (Tech Stack & Environment)

*   **Python 运行环境**：Python 3.10.6 (Conda 环境: `recsys`)
    *   环境 Python 路径：`D:\environment\miniconda3\envs\recsys\python.exe`
*   **核心依赖库**：PyTorch 1.12.1+cu113, Pandas, NumPy, scikit-learn, python-docx, tqdm
*   **评测大模型**：DeepSeek-v4-flash API

---

## ⚡ 核心命令指南 (Core Commands)

在开发过程中，请确保使用 recsys 环境中的 Python 解释器运行脚本：

```bash
# 定义 python 路径变量 (Windows Powershell 规范)
$PYTHON="D:\environment\miniconda3\envs\recsys\python.exe"

# 1. 运行传统模型批量流水线 (Popularity, ItemCF, BPR, SASRec)
& $PYTHON scripts/run_all.py

# 2. 训练单个传统模型 (例如 SASRec)
& $PYTHON scripts/train.py --category Industrial_and_Scientific --model sasrec

# 3. 评估传统模型 (例如 BPR)
& $PYTHON scripts/evaluate.py --category Industrial_and_Scientific --model bpr

# 4. 评估大语言模型重排序 (LLM Ranker)
# 需要传入 deepseek 密钥，支持 --sample-size (推荐 500 进行全量验证)
& $PYTHON scripts/evaluate_llm_sampled.py --sample-size 500 --max-workers 15

# 5. 一键同步评测指标至大屏数据及 MD 实验报告
& $PYTHON scripts/update_llm_metrics.py

# 6. 将 Markdown 实验报告编译生成完美的交付级 Word 报告 (.docx)
& $PYTHON scripts/convert_report.py
```

---

## 📐 核心开发规范与 Bug 防御 (Conventions & Defensive Coding)

### 1. SASRec 序列填充与索引对齐约定 [CRITICAL]
*   **填充方向**：统一使用 **右填充 (Right Padding)**。即较短历史序列的尾部用 `0` 补齐，其后接真实的交互。
*   **序列截断**：`max_len = 50`。
*   **目标特征提取**：训练与评估在提取最后一项商品的隐藏状态时，必须精确使用 `seq_lens - 1` 的动态索引定位（原 Bug 版使用了固定索引导致在右填充下提取到了 padding 占位符 `0` 处的全零隐藏层向量，造成模型崩溃不收敛）。
*   **防御**：严禁在 `src/models/sasrec.py` 和 `src/data/dataset.py` 中擅自改动填充方向。

### 2. LLM 大模型评测与网络防抖约定
*   **多线程并发限制**：在 Windows 端为防止 Socket 句柄耗尽或 TCP 并发瓶颈引发 Python 静默退出，多线程并发度 `max_workers` **不得超过 15**。
*   **网络容错**：接口调用必须包裹在包含指数退避重试（至少 3 次重试上限）的 `try-except` 中，防止 DeepSeek 429 频控报错阻断评测。
*   **强格式控制**：利用 System Prompt 人设先验与 Strict Output Formatting 约束，强力规避 DeepSeek 吐出冗余文本，确保解析错误率为 0.0%。

### 3. 代码架构 conventions
*   **模型基类**：所有推荐模型必须继承自 `src.models.base.BaseRecommender`。
*   **核心接口**：
    *   `fit(train_data)`：训练接口。
    *   `recommend(history, top_k)`：单个用户序列推荐接口，返回包含 Top-K 商品 ID 的 Python List。
    *   `recommend_batch(histories, top_k)` [RECOMMENDED]：批量向量化推荐接口，必须利用 PyTorch 张量操作在 GPU 端并发计算，严禁写串行循环评估（速度相差 11 倍以上）。
*   **评估规范**：统一使用 `src.evaluation.evaluator.Evaluator` 进行 `NDCG@10`、`Hit@10` 和 `MRR@10` 的计算。

---

## 📺 可视化大屏与免 CORS 约定 (Visual Dashboard & CORS Avoidance)

*   **大屏文件**：根目录下 [dashboard.html](file:///c:/Users/Jame.RC/Desktop/推荐系统/dashboard.html)
*   **数据载体**：[results/dashboard_data.js](file:///c:/Users/Jame.RC/Desktop/推荐系统/results/dashboard_data.js)
*   **免 CORS 策略**：为了规避浏览器沙箱加载本地 JSON 导致的跨域限制（CORS Error），数据必须以 JS 全局变量 `DASHBOARD_DATA` 写入 `dashboard_data.js`，网页直接通过 `<script>` 引入。
*   **启动大屏**：**无需搭建任何本地服务器，直接在 Windows 中双击 `dashboard.html` 即可在任意浏览器中完美交互运行！**
