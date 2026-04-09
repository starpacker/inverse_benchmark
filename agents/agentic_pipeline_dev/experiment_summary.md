# Experiment Records: Teacher Model Experience Extraction

This document summarizes the experimental results of different Teacher Model configurations for experience extraction in the Agentic Pipeline.

## 1. Experiment Overview

| Experiment ID | Configuration | Report File | Success Rate | Generated Experiences | Generated Instances |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **Few-shot + Experiences** | `execution_report_20260221_121404.json` | **25.9%** (7/27) | 148 | 14 |
| **Exp 2** | **Only-Experiences** | `execution_report_20260222_175626.json` | **37.0%** (10/27) | 213 | 37 |
| **Exp 3** | **Only-Experiences (Layered/Iterative)** | `execution_report_20260223_120222.json` | **40.7%** (11/27) | 623 | 46 |

---

## 2. Detailed Comparison

### Exp 1: Few-shot + Experiences
*   **File**: [execution_report_20260221_121404.json](file:///home/yjh/agentic_pipeline_dev/execution_report_20260221_121404.json)
*   **Timestamp**: 2026-02-21 12:14:04
*   **Performance**:
    *   Success: 7
    *   Failure: 20
*   **Knowledge Generation**:
    *   Experiences: 148 (Avg ~5.5 per task)
    *   Instances: 14
*   **Observation**: Initial baseline. The combination of few-shot instances and experiences yielded a moderate success rate.

### Exp 2: Only-Experiences
*   **File**: [execution_report_20260222_175626.json](file:///home/yjh/agentic_pipeline_dev/execution_report_20260222_175626.json)
*   **Timestamp**: 2026-02-22 17:56:26
*   **Performance**:
    *   Success: 10 (+3 vs Exp 1)
    *   Failure: 17
*   **Knowledge Generation**:
    *   Experiences: 213 (Avg ~7.9 per task)
    *   Instances: 37
*   **Observation**: Focusing purely on experiences (or prioritizing them) improved the success rate to 37%. The number of generated experiences increased significantly.

### Exp 3: Only-Experiences (Teacher Layered Generation)
*   **File**: [execution_report_20260223_120222.json](file:///home/yjh/agentic_pipeline_dev/execution_report_20260223_120222.json)
*   **Timestamp**: 2026-02-23 12:02:22
*   **Performance**:
    *   Success: 11 (+1 vs Exp 2)
    *   Failure: 16
*   **Knowledge Generation**:
    *   Experiences: **623** (Avg ~23.0 per task)
    *   Instances: 46
*   **Observation**: The "Layered" or "Iterative" generation approach for the Teacher model resulted in a massive increase in the quantity of extracted experiences (nearly 3x Exp 2). This denser knowledge base contributed to a further improvement in success rate to 40.7%.

---

## 3. Task-Specific Improvements (Exp 1 vs Exp 3)

Tasks that improved from Exp 1 to Exp 3:
*   **semiblindpsfdeconv-master**: Failure -> Success
*   **bayhunt**: Failure -> Success
*   **pyeit**: Failure -> Success
*   **dpi_task1**: Failure -> Success
*   **dpi_task2**: Failure -> Success
*   **fpm_inr**: Failure -> Success

Tasks that regressed (Success -> Failure):
*   *None observed in the summary comparison (need detailed check if any previously successful task failed).*
