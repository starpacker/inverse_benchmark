# Agentic Pipeline — 综合技术文档

> **面向科学反问题自动代码生成的多智能体系统**
> 
> 版本: 1.1 | 最后更新: 2026-02-21

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构与数据流](#2-架构与数据流)
3. [核心组件](#3-核心组件)
   - 3.1 [入口点: main_flow.py](#31-入口点-main_flowpy)
   - 3.2 [工作流引擎: workflow_base.py](#32-工作流引擎-workflow_basepy)
   - 3.3 [执行报告器: reporting.py](#33-执行报告器-reportingpy)
4. [智能体系统](#4-智能体系统)
   - 4.1 [规划智能体 (Planner Agent)](#41-规划智能体-planner-agent)
   - 4.2 [架构智能体 (Architect Agent)](#42-架构智能体-architect-agent)
   - 4.3 [编码智能体 (Coder Agent)](#43-编码智能体-coder-agent)
   - 4.4 [评审智能体 (Judge Agent)](#44-评审智能体-judge-agent)
5. [持久化技能系统](#5-持久化技能系统)
   - 5.1 [三层知识架构](#51-三层知识架构)
   - 5.2 [存储层: storage.py](#52-存储层-storagepy)
   - 5.3 [知识管理器: manager.py](#53-知识管理器-managerpy)
   - 5.4 [教师（知识蒸馏）: teacher.py](#54-教师知识蒸馏-teacherpy)
   - 5.5 [进化管理器: evolution_manager.py](#55-进化管理器-evolution_managerpy)
6. [工具组件](#6-工具组件)
   - 6.1 [代码编辑器: code_editor.py](#61-代码编辑器-code_editorpy)
   - 6.2 [技能管理命令行工具: manage_skills.py](#62-技能管理命令行工具-manage_skillspy)
7. [配置系统](#7-配置系统)
   - 7.1 [任务配置](#71-任务配置)
   - 7.2 [LLM 配置](#72-llm-配置)
8. [工作流生命周期（端到端）](#8-工作流生命周期端到端)
9. [关键设计模式与决策](#9-关键设计模式与决策)
10. [轨迹数据格式](#10-轨迹数据格式)

---

## 1. 系统概述

**Agentic Pipeline** 是一个多智能体 AI 系统，旨在自动生成、调试和优化用于**科学反问题**的 Python 代码——涵盖显微成像重建、地震反演、InSAR 相位解缠、引力透镜效应、医学影像等多个领域。

### 核心能力

| 能力 | 描述 |
|------|------|
| **多智能体协调** | 四个专业化 LLM 智能体（规划器、架构师、编码器、评审官）在结构化流水线中协作 |
| **迭代式自修复** | 失败代码由评审智能体分析，并由编码智能体选择性修复，最多重试 N 次 |
| **持久化知识系统** | 三层知识库（核心/经验/实例）跨任务积累专业知识 |
| **基于 AST 的代码编辑** | 使用 Python AST 进行函数级精准代码替换，保持文件结构不变 |
| **46+ 科学任务** | 预配置的基准任务，涵盖光学、地震学、医学影像等领域 |
| **多 LLM 支持** | 兼容 11+ 种 LLM 后端（Gemini、GPT-5.2、Claude Opus 4.5、Qwen、DeepSeek、Grok 等） |

### 系统全局视图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENTIC PIPELINE                             │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌────────┐    ┌───────────────┐  │
│  │  规划器   │───>│  架构师   │───>│ 编码器  │───>│ 执行+评审循环  │  │
│  │  Agent   │    │   Agent   │    │ Agent  │    │    Loop       │  │
│  └──────────┘    └───────────┘    └────────┘    └───────┬───────┘  │
│       ▲                                                  │          │
│       │              ┌──────────────────┐               │          │
│       └──────────────│    技能系统       │◄──────────────┘          │
│                      │ (核心/经验/实例)  │                           │
│                      └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构与数据流

### 高层数据流

```
配置文件 (YAML)
    │
    ▼
main_flow.py ──── 对每个任务 ────┐
    │                             │
    ▼                             ▼
创建 Workflow ────────> InverseProblemWorkflow.run()
(传入 skill_manager      │
 引用，不预注入知识)      ├─── 0. Phase 0 准备（沙箱 + 数据生成 + 评估脚本）
                          ├─── 1. 规划器 → 数学方案（Critic 审查）
                          │       ↑ 每个 Agent 调用前通过
                          │         _build_context_with_memory()
                          │         独立检索并注入分层知识
                          ├─── 2. 架构师 → 代码骨架
                          ├─── 3. 编码器 → 完整实现
                          │       (imports → init → methods → main)
                          ├─── 4. 执行 (subprocess) + 评估
                          ├─── 5. 评审官 → 通过/失败 + 诊断
                          │       │
                          │       ├─ 通过 → 蒸馏知识（实例+经验）
                          │       └─ 失败 → 修复工单 → ticket 调度
                          │              │  → Coder/Architect/Planner
                          │              └─ 最终失败 → 仅蒸馏经验
                          │
                          ▼
                    报告生成 (JSON)
```

### 目录结构

```
agentic_pipeline/
├── main_flow.py              # 入口点，批量编排
├── workflow_base.py          # 核心工作流引擎（规划器→架构师→编码器→评审官循环）
├── reporting.py              # 执行报告生成
├── agents/
│   ├── planner.py            # 数学规划智能体
│   ├── architect.py          # 代码骨架生成智能体
│   ├── coder.py              # 函数实现智能体
│   └── judge.py              # 错误诊断与修复分配智能体
├── persistent_skill_system/
│   ├── storage.py            # SQLite 存储后端
│   ├── manager.py            # 知识检索与蒸馏编排器
│   ├── teacher.py            # 轨迹 → 知识提取（基于 LLM）
│   ├── evolution_manager.py  # 离线知识进化（DBSCAN + LLM 归纳）
│   ├── skills_new.db         # SQLite 数据库文件
│   └── trajectories/         # JSON 轨迹文件
│       └── *.json
├── config/
│   ├── config_task.yaml      # 任务定义（27 个任务）
│   ├── config_task_2.yaml    # 额外任务定义（26 个任务）
│   └── config_llm.yaml       # LLM 模型配置
├── utils/
│   └── code_editor.py        # 基于 AST 的代码编辑工具
└── scripts/
    └── manage_skills.py      # 技能/轨迹检查命令行工具
```

---

## 3. 核心组件

### 3.1 入口点: `main_flow.py`

**用途**: 编排整个流水线——加载配置、初始化 LLM 客户端、检索知识，并将每个任务送入工作流引擎运行。

#### 关键函数

| 函数 | 描述 |
|------|------|
| `create_llm_client(model_name, config)` | 从 config_llm.yaml 配置创建 OpenAI 兼容客户端 |
| `load_task_description(gt_code_path)` | 读取参考代码文件，通过 LLM 生成结构化任务描述 |
| `run_single_task(task_config, ...)` | 执行单个任务：知识注入 → 工作流运行 → 轨迹保存 → 蒸馏 |
| `main()` | 入口函数：解析配置，遍历任务，生成最终报告 |

#### 执行流程

```python
# 简化的 main() 流程:
1. 加载 config_task.yaml → 任务列表
2. 加载 config_llm.yaml → LLM 配置
3. 初始化 SkillManager（持久化知识系统）
4. 对每个任务:
   a. 加载任务描述（来自 gt_code_path）
   b. 创建 InverseProblemWorkflow 实例（传入 skill_manager 引用）
   c. workflow.run() → (success, result_code)
      ※ 知识检索与注入在 run() 内部按 Agent 独立进行
        （通过 _build_context_with_memory），不在此处统一注入
   d. 保存轨迹到技能系统
   e. 蒸馏知识（成功→实例+经验；失败→仅经验）
   f. 更新已使用知识的信用分
5. 生成执行报告 (JSON)
```

#### 知识注入格式

检索到的知识以结构化提示片段的形式注入任务描述：

```markdown
### 💡 相关经验模式（策略）
1. **[模式名称]**
   - 条件: 当 [情境]...
   - 动作: [推荐方法]...
   - 理由: [为什么有效]...

### 📝 参考示例（Few-Shot）
#### 示例（方案）: [task_name] 的方案
[完整方案文本...]
#### 示例（代码）: [task_name] 的解决方案代码
[完整代码...]
```

---

### 3.2 工作流引擎: `workflow_base.py`

**用途**: 核心引擎，编排四智能体流水线，带有迭代重试逻辑。

#### 类: `InverseProblemBase` / `InverseProblemWorkflow`

```python
# workflow_base.py
class InverseProblemBase:
    def __init__(self, task_name, task_desc, gt_code_path, python_path,
                 working_dir, client, model_name, root_output_dir, 
                 skill_manager=None)

# main_flow.py
class InverseProblemWorkflow(InverseProblemBase):
    # 继承基类，实现 run() 主循环
```

#### 关键属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `task_name` | str | 当前任务名称 |
| `task_desc` | str | 完整任务描述 |
| `sandbox_dir` | str | 代码执行沙箱目录 |
| `python_path` | str | Python 解释器路径（conda 环境） |
| `max_retries` | int | 最大修复尝试次数（默认: 10） |
| `retry_count` | int | 当前重试计数 |
| `current_plan` | str | 规划器的最新方案 |
| `current_skeleton` | str | 架构师的最新骨架代码 |
| `current_code` | str | 编码器的最新完整代码 |
| `trajectory` | list | 完整执行历史（所有步骤） |
| `failure_history` | list | 失败诊断列表 |
| `used_knowledge_ids` | list | 使用的知识条目 ID |

#### 核心方法: `run()`

```python
def run(self) -> Tuple[bool, str]:
    # 阶段 1: 规划
    plan = self.planner.generate_plan(self.task_desc)
    
    # 阶段 2: 架构设计
    skeleton = self.architect.generate_skeleton(self.task_desc, plan)
    
    # 阶段 3: 逐函数实现
    code = skeleton
    for func in extract_functions(skeleton):
        code = self.coder.implement_function(func, code, plan, self.task_desc)
    
    # 阶段 4: 执行 + 评审循环
    for retry in range(self.max_retries):
        success, output = self.execute(code)
        if success:
            return True, code
        
        # 评审官分析失败
        diagnosis = self.judge.analyze(code, output, self.failure_history)
        
        # 定向修复
        code = self.coder.fix_function(diagnosis.target, code, diagnosis)
    
    return False, code
```

#### 执行方法

代码作为子进程执行，带超时限制：

```python
def execute(self, code: str) -> Tuple[bool, str]:
    # 将代码写入 working_folder/sim_code.py（或任务特定文件）
    # 运行: subprocess.run([python_path, code_file], 
    #                      cwd=working_folder, timeout=300)
    # 捕获 stdout + stderr
    # 检查返回码 == 0 且输出文件存在
```

#### 轨迹记录

每次智能体调用都记录为一个步骤：

```python
step = {
    "step_id": N,
    "iteration": retry_count,
    "role": "Planner|Architect|Coder|Judge",
    "timestamp": time.time(),
    "input": { ... },   # 智能体接收的输入
    "output": { ... },  # 智能体产生的输出
    "retrieval_key": "..."  # 用于后续知识检索
}
self.trajectory.append(step)
```

---

### 3.3 执行报告器: `reporting.py`

**用途**: 生成 JSON 报告，汇总批量执行结果。

#### 类: `ExecutionReporter`

```python
class ExecutionReporter:
    def __init__(self, root_output_dir: str)
    def add_result(self, task_name, workflow, success, elapsed)
    def generate_report(self) -> str  # 返回 JSON 报告路径
```

#### 报告结构

```json
{
  "meta": {
    "timestamp": "2026-02-20 23:45:00",
    "total_duration_seconds": 3600.0,
    "total_tasks": 10,
    "success_count": 7,
    "failure_count": 3,
    "success_rate": 70.0
  },
  "knowledge_generation_summary": {
    "instances": 15,
    "experiences": 8,
    "core": 2
  },
  "tasks": [
    {
      "task_name": "sim",
      "outcome": "Success",
      "loops_used": 1,
      "elapsed_seconds": 120.5,
      "generated_knowledge": {"instances": 2, "experiences": 1, "core": 0},
      "used_knowledge_count": 3,
      "error_summary": null
    }
  ]
}
```

---

## 4. 智能体系统

系统采用**六个专业化 LLM 智能体**和**两个辅助智能体**，在结构化流水线中协作。

| 智能体 | 文件 | 角色 | 调用阶段 |
|--------|------|------|---------|
| PlannerAgent | `agents/planner_agent.py` | 生成数学/算法方案 | 阶段 1 |
| CriticAgent | `agents/planner_agent.py` | 审查方案质量（PASS/REJECT） | 阶段 1（内循环） |
| ArchitectAgent | `agents/architect_agent.py` | 方案→代码骨架 | 阶段 2 |
| CoderAgent | `agents/coder_agent.py` | 逐函数实现 + 修复 | 阶段 3 + 4 |
| JudgeAgent | `agents/judge_agent.py` | 错误诊断 + ticket 分配 | 阶段 4 |
| DataGenAgent | `agents/sandbox_manager.py` | 生成测试数据脚本 | 阶段 0 |
| EvalGenAgent | `agents/sandbox_manager.py` | 生成评估脚本 | 阶段 0 |

### 4.0 Critic 智能体 (Critic Agent)

**文件**: `agents/planner_agent.py`（与 PlannerAgent 同文件）

**角色**: 对 Planner 生成的方案进行质量审查，决策 PASS 或 REJECT。与 Planner 形成最多 3 轮的审查内循环。

#### 审查循环

```
Planner 生成方案
    │
    ▼
Critic 审查 ──── PASS ───→ 进入 Architect
    │
    └── REJECT + 反馈 ──→ Planner 修改方案（最多 3 轮）
```

#### 输出
- `verdict`: "PASS" 或 "REJECT"
- `feedback`: 具体的改进建议（仅 REJECT 时）

---

### 4.1 规划智能体 (Planner Agent)

**文件**: `agents/planner_agent.py`

**角色**: 为给定的反问题生成详细的数学和算法方案。

#### 输入
- 任务描述（通过 `_build_context_with_memory` 注入分层知识：核心约束 + 经验模式 + few-shot 实例）
- 先前反馈（如果失败后重新规划，或 Critic REJECT）

#### 输出结构
规划器生成结构化的 Markdown 文档：

```markdown
## 1. [问题建模]
- 正向模型: 物理过程的数学描述
- 反问题: 带目标函数的优化问题建模
- 变量定义表

## 2. [提议策略]
- 算法选择及理由
- 为什么选择此策略（稳定性、效率等）

## 3. [分步方案]
- 步骤 1: 数据预处理（详细子步骤）
- 步骤 2: 正向算子实现
- 步骤 3: 求解器/算法细节（完整迭代循环与伪代码）
- 步骤 4: 损失函数与优化器（超参数、评估指标）
```

#### 提示工程
- 使用**经验模式**作为策略提示（条件→动作→理由）
- 使用**成功方案的 few-shot 示例**来自相似任务
- 要求数学严谨性，使用 LaTeX 标记

---

### 4.2 架构智能体 (Architect Agent)

**文件**: `agents/architect.py`

**角色**: 将方案转换为 Python 代码骨架，包含类结构、方法签名、类型标注和 TODO 占位符。

#### 输入
- 任务描述
- 规划器的方案
- 先前骨架（如果修改中）
- 反馈（如果修复中）

#### 输出
完整的 Python 文件：

```python
import numpy as np
# ... 所有必需的导入

class InverseSolver:
    def __init__(self, param1=default1, ...):
        self.param1 = param1
        # ... 所有参数和运行时状态
    
    def preprocess(self, data):
        """解释该方法的详细文档字符串。"""
        # TODO: 实现预处理
        pass
    
    def forward(self, x):
        """应用正向模型。"""
        # TODO: 实现正向算子
        pass
    
    def solve(self, data):
        """主求解方法。"""
        # TODO: 实现求解循环
        pass
    
    def evaluate(self, result, expected):
        """计算评估指标。"""
        # TODO: 实现评估
        pass

if __name__ == "__main__":
    data = np.load("dataset/input.npy")
    solver = InverseSolver(...)
    result = solver.solve(data)
    np.save("output.npy", result)
```

#### 设计约定
- 单一 `InverseSolver` 类封装整个流程
- 标准接口: `preprocess()` → `solve()` → `evaluate()`
- `__main__` 代码块用于独立执行
- 所有超参数作为构造函数参数并带有默认值

---

### 4.3 编码智能体 (Coder Agent)

**文件**: `agents/coder.py`

**角色**: 在骨架中逐个实现函数，每次实现一个。这是调用最频繁的智能体。

#### 实现顺序

```
1. imports（替换导入部分）
2. __init__（构造函数）
3. _helper_method_1
4. _helper_method_2
5. ...（所有私有/内部方法）
6. preprocess
7. forward
8. solve
9. evaluate
10. __main__ 代码块
```

#### 每个函数的输入
- 目标函数名称和类型（function/imports/main_block）
- 完整骨架代码（提供上下文）
- 当前完整代码（含已实现的函数）
- 方案（提供算法细节）
- 任务描述（提供领域知识）
- 可用包列表
- 反馈（如果修复特定函数）

#### 输出
编码器返回**带有目标函数已实现的完整文件**。系统使用 `CodeEditor.replace_function()`（基于 AST）精确替换工作代码中的目标函数。

#### 修复模式
当评审官识别出故障函数时，编码器接收：
- 错误消息和堆栈追踪
- 评审官的诊断（根因分析）
- 需要修复的特定函数（`fix_target`）
- 先前失败历史（避免重复错误）

---

### 4.4 评审智能体 (Judge Agent)

**文件**: `agents/judge.py`

**角色**: 分析执行失败，识别根本原因，并向编码器分配定向修复"工单"。

#### 输入
- 当前代码
- 执行输出（stdout + stderr）
- 先前失败历史（检测循环）

#### 输出: 修复工单

```python
{
    "ticket_assigned_to": "Coder",
    "analysis": "_split_bregman 方法在收敛检查中存在偏差错误。
                 残差比较使用了 abs(residual) 
                 而非 abs(residual - prev_residual)。",
    "fix_target": "_split_bregman",     # 需修复的特定函数
    "fix_type": "function",              # function | imports | main_block
    "severity": "high",
    "suggested_approach": "将第 ~180 行的收敛检查替换为
                          相对变化: abs(r - r_prev) / max(r_prev, 1e-10)"
}
```

#### 诊断策略
1. 解析错误追踪，识别出错的行/函数
2. 分析错误类型（SyntaxError、RuntimeError、ValueError 等）
3. 与方案交叉对比，检查算法正确性
4. 检查先前失败历史，避免重复建议相同修复
5. 将修复分配给尽可能具体的函数

---

## 5. 持久化技能系统

技能系统是一个**三层知识库**，跨任务积累专业知识，使系统能够从过去的成功和失败中学习。

### 5.1 三层知识架构

```
┌─────────────────────────────────────────────┐
│          核心知识 (Core)（顶层）              │
│  从多个经验中通过 DBSCAN + LLM 归纳出       │
│  的通用原则                                  │
│  示例: "对于基于 ADMM 的成像，始终应         │
│  离线预计算 FFT 分母"                        │
├─────────────────────────────────────────────┤
│        经验知识 (Experience)（中层）          │
│  从成功和失败轨迹中提取的                    │
│  条件-动作-理由模式                          │
│  示例: "实现 TV 去噪时                       │
│  → 使用索引数组处理边界"                     │
├─────────────────────────────────────────────┤
│        实例知识 (Instance)（底层）            │
│  来自特定任务执行的                          │
│  具体智能体产物（few-shot）                  │
│  示例: "sim" 任务的完整方案/代码              │
└─────────────────────────────────────────────┘
```

| 层级 | 粒度 | 来源 | 用途 |
|------|------|------|------|
| **核心 (Core)** | 通用原则 | DBSCAN 聚类经验 → LLM 归纳 | 始终包含在提示中 |
| **经验 (Experience)** | 条件→动作→理由 | LLM 从成功+失败轨迹提取 | 按相似度 Top-K 检索 |
| **实例 (Instance)** | 完整产物（方案/代码） | 从轨迹直接提取 | 作为 few-shot 示例 Top-K 检索 |

---

### 5.2 存储层: `storage.py`

**文件**: `persistent_skill_system/storage.py`

**用途**: 基于 SQLite 的持久化存储，带向量相似度搜索。

#### 数据库结构

**表: `knowledge_items`**（v1.1 更新）

| 列名 | 类型 | 描述 |
|------|------|------|
| `id` | TEXT (主键) | UUID 标识符 |
| `name` | TEXT | 人类可读名称 |
| `type` | TEXT | `instance` / `experience` / `core` |
| `content` | TEXT | JSON 内容（结构因类型而异） |
| `embedding` | BLOB | float32 向量（384 维，all-MiniLM-L6-v2） |
| `tags` | TEXT | JSON 标签列表 |
| `source_trajectories` | TEXT | JSON 来源实验 ID 列表 |
| `credit_score` | REAL | 质量评分（默认 1.0，基于使用结果更新） |
| `usage_count` | INTEGER | 被检索使用的次数 |
| `agent_scope` | TEXT | `Planner` / `Architect` / `Coder` / `Judge` / `General` |
| `artifact_type` | TEXT | `plan` / `skeleton` / `code` / `feedback` / `experience_pattern` / `principle` |
| `status` | TEXT | `active` / `archived` / `hypothesis` / `verified` / `deprecated` |
| `preconditions` | TEXT | JSON 前置条件列表（Core Knowledge 冲突检测） |
| `contributed_to_ck_ids` | TEXT | JSON Core Knowledge ID 列表（Experience 追溯） |
| `source_experience_ids` | TEXT | JSON Experience ID 列表（Core Knowledge 来源） |
| `create_time` | INTEGER | Unix 时间戳 |
| `update_time` | INTEGER | Unix 时间戳 |
| `version` | INTEGER | 版本计数器 |

**表: `skills`**（旧表，保留向后兼容）

| 列名 | 类型 | 描述 |
|------|------|------|
| `name` | TEXT (主键) | 技能名称 |
| `principle` | TEXT | 技能原则描述 |
| `when_to_apply` | TEXT | 适用条件 |
| `source` | TEXT | JSON 来源列表 |
| `embedding` | BLOB | float32 向量 |
| `create_time` | INTEGER | Unix 时间戳 |
| `update_time` | INTEGER | Unix 时间戳 |
| `version` | INTEGER | 版本计数器 |

> **注**: 轨迹不再通过数据库持久化。蒸馏完成后，仅提取的 Instance 和 Experience 被存储到 `knowledge_items` 表中，原始轨迹 JSON 文件保存在 `persistent_skill_system/trajectories/` 目录但不在数据库中索引。

#### 关键方法

```python
class SkillStorage:
    def __init__(self, db_path: str)
    
    # 知识条目操作（knowledge_items 表）
    def add_knowledge_item(self, item_data: Dict) -> bool
    def get_knowledge_items(self, k_type=None, agent_scope=None, 
                            limit=None) -> List[Dict]
    def get_knowledge_by_ids(self, ids: List[str]) -> List[Dict]
    def search_knowledge(self, query_embedding, k_type=None, 
                         agent_scope=None, top_k=5) -> List[Dict]
    def update_knowledge_usage(self, item_id: str, success: bool)
    
    # 旧 skills 表操作（向后兼容）
    def add_skill(self, skill_data: Dict) -> bool
    def update_skill(self, name: str, update_data: Dict) -> bool
    def get_all_skills(self) -> List[Dict]
    def get_relevant_skills(self, query_embedding, top_k=5) -> List[Dict]
    def merge_skill(self, new_skill_data, similarity_threshold=0.85) -> bool
```

#### 向量相似度搜索

搜索使用**余弦相似度**结合**信用评分加权**：

```python
def search_knowledge(self, query_embedding, k_type=None, 
                     agent_scope=None, top_k=5):
    items = self.get_knowledge_items(k_type, agent_scope)
    
    for item in items:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        
        # 信用评分归一化到 0-1
        norm_credit = item['credit_score'] / max_credit
        
        # 加权评分（两因子）:
        final_score = similarity * 0.7 + norm_credit * 0.3
    
    # 去重: 同名知识最多返回 1 条（保证多样性）
    # 返回加权评分最高的 top_k 条
```

---

### 5.3 知识管理器: `manager.py`

**文件**: `persistent_skill_system/manager.py`

**用途**: 知识检索、嵌入生成、轨迹保存和蒸馏触发的高层编排器。

#### 类: `SkillManager`

```python
class SkillManager:
    def __init__(self, db_path: str, client=None, model_name: str = "gpt-4")
    # 内部持有:
    #   self.storage = SkillStorage(db_path)
    #   self.teacher = SkillTeacher(client, model_name)
    #   self.embedder = None  # SentenceTransformer (懒加载)
    #   self.target_dim = 384  # 默认嵌入维度
```

#### 关键方法

| 方法 | 描述 |
|------|------|
| `get_embedding(text)` | 生成嵌入向量（优先 SentenceTransformer，回退到确定性哈希伪嵌入） |
| `retrieve_knowledge(task_desc, agent_role, top_k)` | 为特定智能体分层检索相关知识 |
| `format_knowledge_for_prompt(knowledge)` | 将分层知识格式化为可注入提示的结构化文本 |
| `distill_and_store(trajectory)` | 从轨迹蒸馏知识（实例+经验）并存储到数据库 |
| `update_scores(knowledge_ids, success)` | 批量更新已使用知识的信用分 |
| `get_knowledge_details(knowledge_ids)` | 按 ID 批量获取知识详情 |
| `_deinstantiate(text)` | 将具体值替换为占位符（URL→`{url}`、路径→`{path}`、UUID→`{uuid}`、大数字→`{number}`） |

#### 嵌入生成

```python
def get_embedding(self, text: str) -> List[float]:
    # 1. 优先尝试本地 SentenceTransformer (all-MiniLM-L6-v2, 384维)
    if self.embedder and self.target_dim == 384:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()
    
    # 2. 回退: 确定性伪嵌入（基于 SHA-256 哈希种子的随机向量）
    #    特殊处理: 包含特定关键词的文本生成相近向量（模拟语义相似性）
    hash_val = hashlib.sha256(text.encode('utf-8')).hexdigest()
    seed = int(hash_val, 16)
    random.seed(seed)
    vec = [random.uniform(-1, 1) for _ in range(self.target_dim)]
    # 添加轻微噪声确保不完全相同
    return vec
```

#### 检索流程

```python
def retrieve_knowledge(self, task_desc, agent_role='General', top_k=3):
    # 清除已注入的技能头部，避免重复
    clean_desc = task_desc.split("### 🧠 RELEVANT SKILLS")[0]
    clean_desc = clean_desc.split("### 🛡️ CORE KNOWLEDGE")[0]
    embedding = self.get_embedding(clean_desc)
    
    results = {
        "core": [],       # 核心知识: Top-5 全局约束
        "experience": [],  # 经验模式: Top-K 策略
        "instance": []     # 实例: Top-2 Agent 特定 few-shot
    }
    
    # 1. Core Knowledge（全局约束，始终检索）
    results['core'] = self.storage.search_knowledge(
        embedding, k_type='core', top_k=5)
    
    # 2. Experience（条件-动作-理由模式）
    results['experience'] = self.storage.search_knowledge(
        embedding, k_type='experience', top_k=top_k)
    
    # 3. Instance（按 Agent 角色过滤的 few-shot 示例）
    if agent_role in ['Planner', 'Architect', 'Coder', 'Judge']:
        results['instance'] = self.storage.search_knowledge(
            embedding, k_type='instance', 
            agent_scope=agent_role, top_k=2)
    
    return results
```

---

### 5.4 教师（知识蒸馏）: `teacher.py`

**文件**: `persistent_skill_system/teacher.py`

**用途**: 使用 LLM 分析**成功和失败轨迹**，在两个层面提取可复用知识。关键设计：Instance（实例）仅从成功轨迹提取（质量门控），Experience（经验）从成功和失败轨迹中均提取。

#### 类: `SkillTeacher`

```python
class SkillTeacher:
    def __init__(self, llm_client, model_name)
```

#### 蒸馏流水线

```
任意轨迹（成功或失败）
    │
    ├─── outcome == 'success' ───┐
    │                             ▼
    │               ┌───────────────────────────────────┐
    │               │  实例提取（仅成功轨迹）             │
    │               │  基于规则提取具体产物:               │
    │               │    - final_plan → Planner 实例      │
    │               │    - 最终 skeleton → Architect 实例  │
    │               │    - final_code → Coder 实例        │
    │               │    - Judge 反馈样本（最多2个）       │
    │               │  带嵌入存储到 DB                     │
    │               └───────────────────────────────────┘
    │
    ▼ （无论成败均执行）
┌───────────────────────────────────┐
│  经验提取（成功 + 失败轨迹）       │
│  LLM 分析完整轨迹:                │
│    成功: "哪些模式促成了成功？"     │
│    失败: "什么导致了失败？如何避免？"│
│  提取为 条件→动作→理由 模式        │
│  去实例化（_deinstantiate）后存储  │
└───────────────────────────────────┘
```

> **设计理由**: Instance 是作为 few-shot 示例注入 Agent 的具体产物（plan/code/skeleton），如果从失败轨迹中提取，会将**错误代码**作为参考示例"毒化"后续任务。而 Experience 是抽象的策略模式，失败中的"教训"（如"当使用 ADMM 时，如果不归一化输入会导致 NaN"）具有极高价值。

#### 实例提取

实例提取是**基于规则的**，从轨迹的顶层字段（`final_plan`、`final_skeleton`、`final_code`）和步骤记录中直接提取产物，**仅对成功轨迹执行**：

```python
def analyze_trajectory_layered(self, trajectory):
    results = {"instances": [], "experiences": []}
    
    # 仅成功轨迹提取实例（质量门控）
    if trajectory['outcome'] == 'success':
        # A. Planner Instance — 从 trajectory['final_plan']
        if trajectory.get('final_plan'):
            results['instances'].append({
                "name": f"Plan for {task_name}",
                "content": trajectory['final_plan'],
                "agent_scope": "Planner",
                "artifact_type": "plan"
            })
        
        # B. Architect Instance — 优先 trajectory['final_skeleton']
        #    回退: 遍历 steps 找最后一个 Architect 步骤的 output['skeleton']
        
        # C. Coder Instance — 从 trajectory['final_code']
        if trajectory.get('final_code'):
            results['instances'].append({
                "name": f"Solution Code for {task_name}",
                "content": trajectory['final_code'],
                "agent_scope": "Coder",
                "artifact_type": "code"
            })
        
        # D. Judge Instance — 遍历 steps 找最近 2 个 Judge 步骤
        #    提取 output['full_judgement_analysis'] 作为诊断示例
    
    # 经验提取（成功+失败均执行，见下方）
    ...
```

#### 经验提取（基于 LLM）

教师提示 LLM 对轨迹进行整体分析：

```
提示: "分析此成功轨迹，提取 2-5 个可复用的经验模式。
每个模式应包含:
- name: 简短描述性名称
- condition: 何时应用此模式？
- action: 使用什么具体实现方法？
- rationale: 为什么有效？
- domain: 适用于什么问题领域？"
```

LLM 返回结构化 JSON 模式，例如：

```json
{
    "name": "使用索引数组的 TV 去噪器边界处理",
    "condition": "当实现 Chambolle 对偶算法用于 TV 去噪且需要 Neumann 边界条件时",
    "action": "预计算移位索引数组 (ir, il) 以实现复制边界条件",
    "rationale": "基于索引的边界处理避免了昂贵的填充操作，保持实现向量化",
    "domain": "image_reconstruction"
}
```

---

### 5.5 进化管理器: `evolution_manager.py`

**文件**: `persistent_skill_system/evolution_manager.py`

**用途**: 定期将经验级知识通过聚类和基于 LLM 的归纳整合为更高级的核心原则。

#### 进化流水线

```
所有经验条目（嵌入向量）
        │
        ▼
┌───────────────────────────────────┐
│  1. DBSCAN 聚类                    │
│     eps=0.3, min_samples=3        │
│     将相似经验分组到簇中           │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  2. LLM 归纳                      │
│     对每个簇:                      │
│     "给定这 N 个相似经验，         │
│      归纳出一个通用原则"           │
│     生成候选核心条目               │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  3. 对抗性验证                     │
│     LLM 将候选原则与已有核心       │
│     知识进行对照检查:              │
│     → 创建 (CREATE)（新原则）      │
│     → 合并 (MERGE)（涵盖已有）     │
│     → 丢弃 (DISCARD)（冗余）      │
└───────────────────────────────────┘
        │
        ▼
  更新后的核心知识层
```

#### 信用评分系统

| 事件 | 信用变化 | 理由 |
|------|---------|------|
| 知识被使用 → 任务成功 | +0.1 | 正向强化 |
| 知识被使用 → 任务失败 | -0.2 | 更强的负向信号 |
| 低信用 < -0.5 | 归档/删除 | 剪枝无用知识 |

#### DBSCAN 参数

| 参数 | 值 | 描述 |
|------|-----|------|
| `eps` | 0.3 | 聚类邻居间的最大距离 |
| `min_samples` | 3 | 形成一个簇的最少经验数 |
| `metric` | cosine | 嵌入向量上的距离度量 |

---

## 6. 工具组件

### 6.1 代码编辑器: `code_editor.py`

**文件**: `utils/code_editor.py`

**用途**: 基于 AST 的精准代码编辑——替换单个函数/方法而不影响文件其余部分。

#### 类: `CodeEditor`

关键操作：

| 方法 | 描述 |
|------|------|
| `extract_functions(code)` | 解析代码 → 返回函数/方法名称及其类型的列表 |
| `replace_function(original_code, func_name, new_code)` | AST 解析两个文件 → 定位目标函数 → 拼接替换 |
| `replace_imports(original_code, new_imports)` | 替换文件顶部的导入块 |
| `replace_main_block(original_code, new_main)` | 替换 `if __name__ == "__main__"` 代码块 |

#### AST 替换工作原理

```python
def replace_function(self, original_code, func_name, new_full_code):
    # 1. 将 original_code 解析为 AST
    original_tree = ast.parse(original_code)
    
    # 2. 将 new_full_code 解析为 AST
    new_tree = ast.parse(new_full_code)
    
    # 3. 在两棵树中找到目标函数（按名称）
    old_node = find_function_node(original_tree, func_name)
    new_node = find_function_node(new_tree, func_name)
    
    # 4. 获取新函数的源代码行
    new_source = ast.get_source_segment(new_full_code, new_node)
    
    # 5. 用新源代码行替换旧源代码行
    lines = original_code.splitlines()
    lines[old_node.lineno-1 : old_node.end_lineno] = new_source.splitlines()
    
    return "\n".join(lines)
```

该方法确保：
- 仅修改目标函数
- 保留周围代码的缩进和格式
- 替换方法时保持类结构完整

---

### 6.2 技能管理命令行工具: `manage_skills.py`

**文件**: `scripts/manage_skills.py`

**用途**: 用于检查和管理技能数据库的命令行工具。

#### 命令

```bash
# 列出所有知识条目
python manage_skills.py list --layer experience

# 显示特定条目的详情
python manage_skills.py show <knowledge_id>

# 列出轨迹
python manage_skills.py trajectories --outcome success

# 手动触发进化
python manage_skills.py evolve

# 导出知识为 JSON
python manage_skills.py export --output skills_dump.json

# 数据库统计信息
python manage_skills.py stats
```

---

## 7. 配置系统

### 7.1 任务配置

**文件**: `config/config_task.yaml`, `config/config_task_2.yaml`

每个任务的定义格式：

```yaml
task_name:
  gt_code_path: "/path/to/ground_truth/sim_code.py"
  working_folder: "/path/to/task/working/directory"
  code_filename: "sim_code.py"          # 输出文件名
  conda_env: "base"                      # Python 环境
  max_retries: 3                         # 最大修复迭代次数
```

#### 任务分类（2 个配置文件共 46 个任务）

| 类别 | 示例任务 | 数量 |
|------|---------|------|
| **光学成像** | 显微反卷积、相位恢复、全息成像 | ~8 |
| **地震学** | 全波形反演、地震层析成像、速度模型 | ~6 |
| **医学影像** | CT 重建、MRI、PET 成像 | ~5 |
| **遥感** | InSAR 相位解缠、SAR 成像、DEM 重建 | ~5 |
| **信号处理** | 光谱解混、压缩感知、盲反卷积 | ~7 |
| **物理** | 引力透镜、量子态层析、电磁 | ~5 |
| **其他** | 扩散 MRI、流场估计、地球物理反演 | ~10 |

### 7.2 LLM 配置

**文件**: `config/config_llm.yaml`

```yaml
models:
  gemini_25_pro:
    model_name: "gemini-2.5-pro-exp-03-25"
    api_key: "${GEMINI_API_KEY}"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
    temperature: 1.0
    max_tokens: 65536
    
  gpt_52:
    model_name: "gpt-5.2"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    temperature: 0.8
    max_tokens: 32768

  claude_opus:
    model_name: "claude-opus-4-5-20250918"
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com/v1/"
    temperature: 1.0
    max_tokens: 65536
    
  # ... 11+ 种模型配置，包括:
  # deepseek_r1, qwen3_235b, grok_3, o3, o4_mini, 
  # gemini_25_flash, claude_sonnet 等
```

所有模型使用 **OpenAI 兼容 API 格式**，允许统一的客户端创建：

```python
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
response = client.chat.completions.create(
    model=config["model_name"],
    messages=[...],
    temperature=config.get("temperature", 1.0),
    max_tokens=config.get("max_tokens", 32768)
)
```

---

## 8. 工作流生命周期（端到端）

### 单个任务的完整生命周期

```
┌─ 阶段 0: 准备 ────────────────────────────────────────────────────┐
│  1. 加载任务配置（gt_code_path、python_path 等）                   │
│  2. 通过 LLM 生成任务描述（从参考代码）                            │
│  3. 创建 InverseProblemWorkflow 实例（传入 skill_manager 引用）    │
│  4. _setup_sandbox(): 初始化沙箱目录，复制 gt_code                │
│  5. _phase_0_preparation():                                        │
│       a. DataGenAgent 生成 data_gen.py → 创建 input/gt_output/    │
│          baseline .npy 文件（最多3次重试）                         │
│       b. EvalGenAgent 生成 eval_script.py → 验证 baseline 指标    │
│       c. _load_data_shapes() → 记录输入输出维度信息                │
│  ※ 知识不在此处注入，而是在每个 Agent 调用前按需检索               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 1: 规划（含 Critic 审查循环）──────────────────────────────┐
│  _build_context_with_memory() → 注入 Core/Experience/Instance     │
│  规划智能体接收: task_desc + 注入知识 + 先前反馈（如有）          │
│  输出: 包含算法细节的数学方案                                      │
│  Critic 审查方案 → PASS 则继续 / REJECT 则返回 Planner（≤3轮）   │
│  记录: 轨迹步骤 (role=Planner)                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 2: 架构设计 ────────────────────────────────────────────────┐
│  架构智能体接收: task_desc + 方案                                  │
│  输出: Python 骨架（类 + 方法签名 + TODO）                        │
│  记录: 轨迹步骤 (role=Architect)                                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 3: 实现 ────────────────────────────────────────────────────┐
│  对骨架中的每个函数（按依赖顺序）:                                │
│    编码智能体接收: target_func + 骨架 + 方案 + task_desc          │
│    输出: 包含目标函数已实现的完整文件                              │
│    CodeEditor: AST 替换工作代码中的目标函数                       │
│    记录: 轨迹步骤 (role=Coder, target=func_name)                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 4: 执行 + 评审循环（Ticket 调度）──────────────────────────┐
│  iteration = 0                                                     │
│  当 iteration < max_retries(=10) 时:                               │
│    1. 语法检查: py_compile 验证（失败→Coder 全文重写，≤5次）      │
│    2. 通过 subprocess 执行（超时=600秒）                           │
│    3. 运行 eval_script.py 计算指标                                 │
│    4. 成功判定: PSNR >= max(baseline_psnr * 0.8, 20.0)            │
│         → 跳出循环，success=True                                   │
│    5. 否则: Judge 分析错误，生成修复工单:                          │
│         ticket_assigned_to 可以是:                                  │
│           - "Coder" → 编码器修复特定函数（最常见）                 │
│           - "Architect" → 重新生成骨架 + 重新实现                  │
│           - "Planner" → 重新规划 + 重新设计 + 重新实现             │
│         _reset_downstream_state(ticket): 按 ticket 级别重置        │
│           Planner ticket → 清空 skeleton + code                    │
│           Architect ticket → 清空 code                             │
│         记录: 轨迹步骤 (role=Judge)                                │
│         iteration += 1                                             │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 5: 后处理 ──────────────────────────────────────────────────┐
│  1. 将完整轨迹保存为 JSON 文件（trajectories/ 目录）              │
│  2. 调用 distill_and_store(trajectory) 蒸馏知识                   │
│  3. 如果成功:                                                      │
│       a. 教师提取实例（方案、骨架、代码、Judge反馈产物）           │
│       b. 教师提取经验（条件→动作→理由）                           │
│       c. 将所有新知识带嵌入存储                                    │
│       d. 更新已使用知识的信用分 (+0.1)                             │
│  4. 如果失败:                                                      │
│       a. 教师提取经验（从失败中学习教训，跳过实例提取）            │
│       b. 将经验知识带嵌入存储                                      │
│       c. 更新已使用知识的信用分 (-0.2)                             │
│  5. 将结果添加到 ExecutionReporter                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键设计模式与决策

### 9.1 逐函数实现

系统不是让编码器一次性编写整个解决方案，而是每次实现一个函数。这样做的好处：
- 让每次 LLM 调用聚焦且可控
- 编码器可以看到先前已实现的函数作为上下文
- 调试更容易（错误可追溯到特定函数）
- 支持基于 AST 的精准修复

### 9.2 基于 AST 的代码编辑 vs. 完全重新生成

系统使用 Python 的 `ast` 模块替换单个函数，而非重新生成整个文件。优势：
- 保留不需要更改的正常工作代码
- 避免不必要的重写导致的回归 bug
- 更快的迭代周期（只重新生成出问题的部分）

### 9.3 三层知识层级

知识系统模拟人类学习：
- **实例** = "我以前做过这个"（情景记忆 / few-shot 示例）
- **经验** = "我知道这个模式"（程序性知识 / 策略）
- **核心** = "这是一个普遍真理"（陈述性知识 / 原则）

### 9.4 基于信用的知识质量控制

知识条目基于下游任务结果获得/失去信用。这创造了一个自然选择机制，有用的知识得以留存，无用的知识被剪枝。

### 9.5 核心知识的对抗性验证

新的核心原则在插入之前会与已有原则进行对照检查，防止知识膨胀和冗余。LLM 同时充当提议者（归纳）和批评者（验证）。

### 9.6 加权检索评分

知识检索结合两个信号：
- **语义相似度**（70%）：该知识与当前任务的相关程度？（余弦相似度）
- **信用分数**（30%）：该知识在过去任务中导致成功的频率？（归一化到 0-1）

公式：`final_score = similarity * 0.7 + normalized_credit * 0.3`

此外，搜索结果进行**同名去重**（每个名称最多返回 1 条），确保检索结果的多样性。

### 9.7 失败历史追踪

评审官可以访问当前任务的所有先前失败诊断，防止重复建议相同的修复方案，并支持逐步升级的修复策略。

---

## 10. 轨迹数据格式

### 文件位置
`persistent_skill_system/trajectories/<task_name>_<timestamp>.json`

### 数据结构

```json
{
  "trajectory_id": "sim_20260220_233944",
  "task_name": "sim",
  "task_description": "实现一个模拟...",
  "outcome": "success",
  "total_iterations": 2,
  "used_knowledge_ids": ["uuid-1", "uuid-2"],
  "steps": [
    {
      "step_id": 1,
      "iteration": 0,
      "role": "Planner",
      "timestamp": 1740067184.5,
      "input": {
        "task_description": "...",
        "feedback": null
      },
      "output": {
        "plan": "## 1. 问题建模\n..."
      },
      "retrieval_key": "sim 任务的方案"
    },
    {
      "step_id": 2,
      "iteration": 0,
      "role": "Architect",
      "timestamp": 1740067220.1,
      "input": {
        "task_description": "...",
        "plan": "..."
      },
      "output": {
        "skeleton": "import numpy as np\n..."
      },
      "retrieval_key": "sim 任务的架构"
    },
    {
      "step_id": 3,
      "iteration": 0,
      "role": "Coder",
      "timestamp": 1740067245.3,
      "input": {
        "target": "__init__",
        "target_type": "function",
        "skeleton": "...",
        "current_code": "...",
        "plan": "..."
      },
      "output": {
        "target": "__init__",
        "code": "def __init__(self, ...):\n    ..."
      },
      "retrieval_key": "sim 的 init 实现"
    },
    {
      "step_id": 10,
      "iteration": 1,
      "role": "Judge",
      "timestamp": 1740067400.0,
      "input": {
        "code": "...",
        "execution_output": "Traceback...",
        "failure_history": [...]
      },
      "output": {
        "ticket_assigned_to": "Coder",
        "fix_target": "solve",
        "fix_type": "function",
        "analysis": "迭代边界存在偏差错误...",
        "suggested_approach": "将 range(N) 改为 range(N+1)..."
      },
      "retrieval_key": "sim 迭代错误的评审诊断"
    }
  ]
}
```

### 轨迹统计（来自现有数据）

系统在 `persistent_skill_system/trajectories/` 中维护轨迹文件。每个成功轨迹通常包含：
- 1 个规划器步骤
- 1 个架构师步骤
- 5-15 个编码器步骤（每个函数一个）
- 0-3 个评审官步骤（取决于重试次数）
- 0-3 个额外的编码器修复步骤

---

## 附录 A: 运行系统

```bash
# 基本执行
cd agentic_pipeline
python main_flow.py --config config/config_task.yaml --model gemini_25_pro

# 指定特定任务
python main_flow.py --config config/config_task.yaml --model gpt_52 --tasks sim,deconv

# 检查技能数据库
python scripts/manage_skills.py stats
python scripts/manage_skills.py list --layer experience

# 触发知识进化
python scripts/manage_skills.py evolve
```

## 附录 B: 添加新任务

1. 创建参考 Python 解决方案文件
2. 在 `config/config_task.yaml` 中添加条目：
   ```yaml
   my_new_task:
     gt_code_path: "/path/to/ground_truth.py"
     working_folder: "/path/to/working/dir"
     code_filename: "sim_code.py"
     conda_env: "my_env"
     max_retries: 3
   ```
3. 确保工作目录中包含所需的输入数据文件
4. 运行: `python main_flow.py --tasks my_new_task`

## 附录 C: 添加新的 LLM

1. 在 `config/config_llm.yaml` 中添加配置：
   ```yaml
   my_new_model:
     model_name: "model-identifier"
     api_key: "${MY_API_KEY}"
     base_url: "https://api.provider.com/v1"
     temperature: 1.0
     max_tokens: 32768
   ```
2. 确保提供方支持 OpenAI 兼容的 chat completions API
3. 运行: `python main_flow.py --model my_new_model`

---

*本文档由 agentic_pipeline 代码库的源码分析生成并手动维护，反映系统当前最新状态。*
