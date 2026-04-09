# OpenHands 适配逆问题求解 Benchmark

本目录包含将 [OpenHands](https://github.com/All-Hands-AI/OpenHands)（v1.4.0）适配到我们的**科学计算逆问题求解 Benchmark** 的完整代码和配置。

## 📋 概述

我们将 OpenHands（一个通用的 AI 编程 Agent 框架）适配到包含 **26 个科学计算逆问题**的 Benchmark 上，用于与我们自研的多 Agent Pipeline 进行对比实验。每个任务要求 Agent 读取测量数据（`input.npy`），根据物理模型编写逆求解器代码（`solver.py`），生成重建结果（`output.npy`），并通过 PSNR/SSIM 等指标评估。

### 核心挑战

| 挑战 | 解决方案 |
|------|---------|
| OpenHands 默认面向软件开发，不适配科学计算任务 | 定制 Prompt 模板，包含物理描述、数据格式、评估标准 |
| 答案泄漏（Agent 可能直接复制 gt_output.npy） | 隐藏 gt 到 `.eval_gt/`，patch eval_script，MD5 校验反作弊 |
| LLM API 兼容性（Claude 4.6 via 第三方 gateway） | Patch LLM 模块解决 temperature/top_p 冲突 |
| 工具调用参数不兼容（security_risk 字段） | 从所有工具的 required 列表中移除 |
| 长时间科学计算被误判为"卡死" | 禁用 stuck detection |
| 单任务隔离（避免状态污染） | 每任务独立 workspace + UUID session + 内存文件存储 |
| 并行执行稳定性 | 解决 action_server 端口耗尽，提供顺序/并行两种模式 |

## 📁 目录结构

```
openhands_benchmark/
├── README.md                          # 本文件
├── run_openhands_benchmark.py         # 🔑 核心：Benchmark Runner（~1080行）
├── config/
│   ├── config.toml                    # OpenHands 配置模板
│   └── tasks_example.yaml             # 任务配置示例
├── patches/
│   └── openhands_v1.4.0_benchmark.patch  # OpenHands 源码补丁（9个文件）
├── scripts/
│   ├── launch_sequential_benchmark.sh # 顺序执行脚本
│   ├── launch_parallel_benchmark.sh   # 8-GPU 并行执行脚本
│   ├── monitor_parallel.py            # 并行监控 & 结果聚合
│   └── generate_report.py             # 结果报告生成
└── results/
    └── benchmark_results_v1.json      # 首次完整运行结果
```

## 🚀 快速开始

### 1. 安装 OpenHands

```bash
# 克隆 OpenHands v1.4.0
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands
git checkout v1.4.0

# 创建 conda 环境
conda create -n openhands python=3.12 -y
conda activate openhands
pip install -e .
```

### 2. 应用补丁

```bash
# 将补丁应用到 OpenHands 源码
cd /path/to/OpenHands
git apply /path/to/openhands_benchmark/patches/openhands_v1.4.0_benchmark.patch
```

补丁修改了 9 个文件（详见[补丁说明](#补丁详解)）。

### 3. 配置

```bash
# 复制配置模板并填入你的 API Key
cp config/config.toml /path/to/OpenHands/config.toml
# 编辑 config.toml，设置：
#   - llm.api_key
#   - llm.base_url
#   - llm.model
```

### 4. 准备 Benchmark 数据

每个任务需要一个标准化的 sandbox 目录：

```
end_sandbox/{task_name}_sandbox/
├── dataset/
│   ├── input.npy          # 测量数据（正问题的输出）
│   ├── baseline.npy       # 基线结果
│   └── gt_output.npy      # 真实值（仅用于评估，不暴露给 Agent）
├── eval_script.py         # 评估脚本
└── gt_code/               # 参考代码（不暴露给 Agent）
```

### 5. 运行

```bash
# 顺序运行所有 26 个任务
bash scripts/launch_sequential_benchmark.sh

# 或直接用 Python
python run_openhands_benchmark.py \
    --task-config config/tasks.yaml \
    --max-iterations 100 \
    --gpu-id 0

# 单任务调试
python run_openhands_benchmark.py \
    --task-filter "mne-master" \
    --max-iterations 50
```

## 🔧 适配工作详解

### Benchmark Runner 核心设计

`run_openhands_benchmark.py` 是整个适配的核心（~1080 行），实现了以下功能：

#### 1️⃣ 沙箱隔离（Anti-Cheat）

```
prepare_sandbox(task_name) →
  ├── 复制 dataset/（排除 gt_output.npy）
  ├── 将 gt_output.npy 藏入 .eval_gt/（不可见）
  ├── Patch eval_script.py（指向隐藏的 gt 路径）
  ├── 验证：gt_code 不存在 + dataset/中无 gt
  └── 生成 README.md 指引 Agent
```

每个任务创建独立的 `{task_name}_workspace/` 目录，Agent 在其中工作。运行结束后通过 **MD5 校验**检测 output.npy 是否为 gt_output.npy 的直接拷贝。

#### 2️⃣ Prompt 工程

为每个任务构建包含以下信息的完整 Prompt：

- **任务描述**：物理模型的数学描述（从 Markdown 文件加载）
- **数据格式**：自动扫描 dataset/ 中所有文件的大小
- **评估脚本**：完整的 eval_script.py 源码（让 Agent 理解评估标准和输出格式）
- **Python 路径**：指定正确的解释器（每个任务可能有不同的 conda 环境）
- **强制约束**：
  - 必须创建 `solver.py` 文件
  - 必须使用 `str_replace_editor` 而非仅靠 `python -c`
  - 禁止复制/重命名 gt 文件
  - 必须运行 eval_script.py 验证

#### 3️⃣ Artifact 保存

每个任务完成后，自动保存：

| 产物 | 说明 |
|------|------|
| `trajectory.json` | Agent 的完整行为轨迹（所有 action/observation 对） |
| `thinking_log.md` | 人类可读的推理日志（Markdown 格式） |
| `agent_solver.py` | 从轨迹中提取的最终求解器代码 |
| `solver.py` | Agent 直接创建的求解器文件 |
| `output.npy` | Agent 生成的重建结果 |

#### 4️⃣ 容错机制

- **基础设施重试**：`RetryError`、`Server process died` 等自动重试（最多 2 次）
- **进程清理**：每个任务后 kill 残留的 `action_execution_server` 和 `file_viewer_server`
- **Global State 清理**：清除 `_RUNNING_SERVERS` 避免端口耗尽
- **`--skip-done`**：跳过已成功的任务（支持断点续跑）

#### 5️⃣ 部分成功检测

即使 Agent 因为 `max_iterations` 或 `AgentStuckInLoopError` 而以 ERROR 状态结束，只要产出了有效的 `output.npy`，仍会运行评估并标记为 `partial_success`。

## 🩹 补丁详解

我们对 OpenHands v1.4.0 做了 9 处修改（均在 `patches/` 中）：

### 修改 1：移除 `security_risk` 必填参数

**文件**：`bash.py`, `browser.py`, `ipython.py`, `str_replace_editor.py`

**问题**：OpenHands 的工具定义中 `security_risk` 是 required 参数，但通过第三方 API Gateway 调用的 Claude 4.6 不会生成该字段，导致所有工具调用失败。

**解决**：从 `required` 数组中移除 `security_risk`。

### 修改 2：`think` 和 `finish` 参数可选

**文件**：`think.py`, `finish.py`

**问题**：Agent 有时不传 `thought`/`message` 参数就调用 think/finish，导致参数校验失败。

**解决**：将 `required` 改为空数组 `[]`。

### 修改 3：禁用 Stuck Detection

**文件**：`openhands/core/config/agent_config.py`

**问题**：科学计算任务（如 MRI 重建、光学反演）的计算时间较长，Agent 在等待计算结果时被误判为"卡在循环中"并强制终止。

**解决**：`enable_stuck_detection` 默认值改为 `False`。

### 修改 4：RecallAction 空查询修复

**文件**：`openhands/events/action/agent.py`

**问题**：`RecallAction.message` 属性调用 `self.query[:50]`，当 `query` 为 `dict` 或 `None` 时抛出 `KeyError`/`TypeError`。

**解决**：改为 `str(self.query) if self.query else ''`。

### 修改 5：Claude 4.6 温度参数兼容

**文件**：`openhands/llm/llm.py`

**问题**：Claude 模型不支持同时传 `temperature` 和 `top_p`。OpenHands 已处理了 `claude-opus-4-5` 等变体，但我们通过 gateway 使用的模型名为 `cds/Claude-4.6-opus`，不在已有的匹配列表中。

**解决**：在模型名匹配中增加 `'claude-4.6' in _model_lower` 条件。

## 📊 实验结果

### 首次完整运行（26 任务 × 50 iterations）

| 指标 | 数值 |
|------|------|
| 总任务数 | 26 |
| 成功（FINISHED + output.npy） | 9 / 26（34.6%） |
| 产出 output.npy | 19 / 26（73.1%） |
| 总耗时 | 6.1 小时 |
| 模型 | Claude 4.6 Opus |
| Agent | CodeActAgent |

### 各任务详细结果

| 任务 | PSNR | SSIM | 状态 | 耗时(s) |
|------|------|------|------|---------|
| mne-master | 319.20 | 1.00 | ✅ FINISHED | 150 |
| storm-analysis-master | 172.27 | 1.00 | ✅ FINISHED | 1010 |
| carspy-main | 89.44 | 1.00 | ◐ partial | 1680 |
| nirfaster-FF-main_2 | 74.85 | 1.00 | ✅ FINISHED | 240 |
| pyDHM-master | 53.63 | 1.00 | ✅ FINISHED | 407 |
| PtyLab-main | 49.60 | 0.96 | ✅ FINISHED | 556 |
| mripy-master | 45.61 | 0.97 | ✅ FINISHED | 1948 |
| caustics-main | 39.65 | 0.96 | ✅ FINISHED | 561 |
| PyHoloscope-main | 33.90 | 0.85 | ✅ FINISHED | 477 |
| PyAbel-master | 27.55 | 0.74 | ◐ partial | 1937 |
| us-beamform-linarray | 24.49 | 0.97 | ✅ FINISHED | 1016 |
| AMICO-master | 18.86 | 0.72 | ◐ partial | 1700 |
| mrf-reconstruction | 12.99 | 0.44 | ◐ partial | 923 |
| dmipy-master | 8.74 | 0.32 | ◐ partial | 486 |
| MRE-elast-master | 8.65 | 0.15 | ◐ partial | 472 |
| oct-cbort-main | 7.63 | 0.16 | ◐ partial | 762 |
| hcipy-master | 0.00 | 0.00 | ◐ partial | 1359 |
| phasorpy-main | — | — | 🚨 cheat | 241 |
| structured-light | — | — | 🚨 cheat | 829 |
| CT-and-MR-Perfusion | — | — | ❌ no output | 481 |
| DiffuserCam-Tutorial | — | — | ❌ no output | 202 |
| MPIRF-master | — | — | ❌ no output | 275 |
| PySMLFM-main | — | — | ❌ no output | 2206 |
| spectral_ct_examples | — | — | ❌ no output | 1434 |
| svmbir-master | — | — | ❌ no output | 267 |
| tomopy-master | — | — | ❌ no output | 205 |

> **状态说明**：✅ = Agent 正常完成（FINISHED）；◐ = Agent 超时/报错但产出了 output（partial_success）；🚨 = 检测到直接复制 gt_output.npy；❌ = 未产出 output.npy

### Agent 行为分析

通过分析 trajectory，我们发现 OpenHands CodeActAgent 的典型工作模式：

1. **一次成功型**（9 任务，≤2 次代码迭代）：如 mne-master 仅用 1 次迭代就完美解决（PSNR=319）
2. **参数调优型**（7 任务，3-10 次迭代）：如 caustics-main 通过 `str_replace` 反复调整迭代次数和正则化参数
3. **暴力搜索型**（3 任务，>10 次迭代）：如 us-beamform 尝试了 23 种不同的滤波方法

## ⚠️ 已知问题

1. **并行执行不稳定**：8-GPU 同时运行时 `action_execution_server` 容易崩溃，建议使用顺序模式
2. **反作弊并非完美**：MD5 校验仅检测完全相同的文件，无法防止 Agent 读取 `.eval_gt/` 下的 gt_output.npy 再做微小修改
3. **solver.py 未必被使用**：尽管 Prompt 强制要求创建 solver.py，部分 Agent 仍然倾向于使用 `python -c` 内联执行
4. **Session 内存泄漏**：长时间运行多个任务后，内存会逐渐增长，建议适时重启

## 🔗 相关项目

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — 通用 AI 编程 Agent 框架
- [agentic_pipeline_dev](../agentic_pipeline_dev/) — 我们自研的多 Agent Pipeline（Planner-Coder-Evaluator 架构）
- [react_inverse_problem](../react_inverse_problem/) — ReAct 范式的多模型对比实验

## 📝 引用

如果您使用了本代码，请引用：

```
@misc{inverse_benchmark_openhands,
  title={Adapting OpenHands for Scientific Inverse Problem Solving},
  year={2026},
  url={https://github.com/starpacker/inverse_benchmark}
}
```
