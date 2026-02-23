# 运行任务 (Run Tasks)
## 运行单个任务 (例如: DiffuserCam-Tutorial-master)
```bash
export TASK_NAMES="DiffuserCam-Tutorial-master" && python3 agentic_pipeline/run_task.py
```

## 运行多个任务 (逗号分隔)
```bash
export TASK_NAMES="Task1,Task2" && python3 agentic_pipeline/run_task.py
```

## 运行所有任务
```bash
export TASK_NAMES="" && python3 agentic_pipeline/run_task.py
```

# 测试模式 (Test Only Mode - Read-only Skills)
## 运行单个测试任务
```bash
export TASK_NAMES="DiffuserCam-Tutorial-master" && python3 agentic_pipeline/run_test_only.py
```

## 运行所有测试任务
```bash
export TASK_NAMES="" && python3 agentic_pipeline/run_test_only.py
```

# 知识库管理 (Knowledge Base Management)

## 1. 查看与编辑 Core Knowledge (交互式工具)
使用此工具可以查看当前数据库中所有的 Core Knowledge，并支持删除或修改内容。
```bash
python3 agentic_pipeline/manage_knowledge.py
```

## 2. 导出所有知识 (Experience, Instance, Core Knowledge)
将数据库中的所有知识导出为人类可读的 Markdown 报告（生成在 `/home/yjh/knowledge_export_YYYYMMDD_HHMMSS`）。
```bash
python3 agentic_pipeline/export_knowledge_report.py
```

## 3. 启动离线进化 (Offline Evolution Daemon)
启动后台守护进程，自动扫描新增的 Experience，通过聚类和归纳生成新的 Core Knowledge。
（建议在 screen 或 tmux 会话中运行，或者直接在终端前台运行）
```bash
python3 agentic_pipeline/run_evolution.py
```
*注：该进程每 5 秒检查一次是否有新的 Experience 加入。*
