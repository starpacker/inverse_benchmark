# Persistent Skill System Design Document

## 1. Overview
The Persistent Skill System is a module designed to accumulate cross-task experience in the form of **layered reusable knowledge**. It operates in an out-of-the-loop manner, distilling knowledge after task completion and injecting relevant knowledge before task execution.

## 2. Architecture
The system consists of four main components:
- **Teacher Model (`teacher.py`)**: Analyzes task trajectories to extract structured **Instances** (rule-based artifact extraction) and **Experiences** (LLM-based pattern distillation).
- **Skill Storage (`storage.py`)**: Manages SQLite database and vector embeddings for knowledge persistence and retrieval. Supports the new `knowledge_items` table with layered types.
- **Skill Manager (`manager.py`)**: Orchestrates the entire process, including layered retrieval, distillation, de-instantiation, credit scoring, and prompt formatting.
- **Evolution Manager (`evolution_manager.py`)**: Offline process that clusters Experiences using DBSCAN and induces **Core Knowledge** principles via LLM-based Generator/Critic loop.

## 3. Knowledge Layers

### 3.1 Instances (Low-Level)
- **What**: Direct artifacts from successful trajectories (plans, skeletons, code, feedback).
- **Extraction**: Rule-based, per agent role. Only extracted from successful outcomes.
- **Agent Scopes**: `Planner` (plans), `Architect` (skeletons), `Coder` (code), `Judge` (feedback).
- **Use**: Few-shot examples injected into agent-specific prompts.

### 3.2 Experiences (Mid-Level)
- **What**: Condition-Action-Rationale patterns distilled from trajectories.
- **Extraction**: LLM-based analysis of trajectory steps (both success and failure).
- **Structure**: `{condition, action, rationale}` with tags.
- **Use**: Strategy suggestions injected as "Relevant Experience Patterns".

### 3.3 Core Knowledge (High-Level)
- **What**: Universal principles and constraints generalized from clusters of Experiences.
- **Extraction**: Offline evolutionary loop (DBSCAN clustering → LLM induction → Adversarial verification).
- **Structure**: `{principle, checklist, preconditions}`.
- **Status Lifecycle**: `hypothesis` → `verified` (via credit scoring) → potentially `deprecated`.
- **Use**: Non-negotiable constraints injected at highest priority.

## 4. Data Schema

### 4.1 Knowledge Items Table (`knowledge_items`)
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `name` | TEXT | Human-readable name |
| `type` | TEXT | `instance`, `experience`, `core` |
| `content` | TEXT | JSON content (structure varies by type) |
| `embedding` | BLOB | float32 vector (384-dim, all-MiniLM-L6-v2) |
| `tags` | TEXT | JSON list of tags |
| `source_trajectories` | TEXT | JSON list of source experiment IDs |
| `credit_score` | REAL | Quality score (default 1.0, updated on usage) |
| `usage_count` | INTEGER | Times retrieved and used |
| `agent_scope` | TEXT | `Planner`, `Architect`, `Coder`, `Judge`, `General` |
| `artifact_type` | TEXT | `plan`, `skeleton`, `code`, `feedback`, `experience_pattern`, `principle` |
| `status` | TEXT | `active`, `archived`, `hypothesis`, `verified`, `deprecated` |
| `preconditions` | TEXT | JSON list (for Core Knowledge conflict detection) |
| `contributed_to_ck_ids` | TEXT | JSON list of Core Knowledge IDs (for Experiences) |
| `source_experience_ids` | TEXT | JSON list of Experience IDs (for Core Knowledge) |
| `create_time` | INTEGER | Unix timestamp |
| `update_time` | INTEGER | Unix timestamp |
| `version` | INTEGER | Incremental version |

### 4.2 Legacy Skills Table (`skills`)
Retained for backward compatibility. New code uses `knowledge_items` exclusively.

## 5. Key Workflows

### 5.1 Knowledge Distillation (Post-Task)
1. **Trigger**: Task completion (Success or Failure).
2. **Input**: Trajectory (Task Description, Steps, Outcome, Final Artifacts).
3. **Process**:
   - **Instance Extraction** (Rule-based, success only):
     - Extract `final_plan` → Planner Instance
     - Extract `final_skeleton` → Architect Instance
     - Extract `final_code` → Coder Instance
     - Extract Judge feedback (up to 2) → Judge Instances
   - **Experience Extraction** (LLM-based, success and failure):
     - Teacher analyzes trajectory steps to identify Condition-Action-Rationale patterns
   - `SkillManager` de-instantiates specific values in Experience text
   - `SkillManager` generates embeddings and stores items
4. **Note**: Raw trajectories are NOT persisted. Only distilled knowledge is stored.

### 5.2 Knowledge Injection (Pre-Task)
1. **Trigger**: Task initialization.
2. **Input**: New Task Description + Agent Role.
3. **Process**:
   - Generate embedding for task description.
   - Retrieve layered knowledge:
     - **Core**: Top-5 global constraints (non-negotiable)
     - **Experience**: Top-K patterns (strategies)
     - **Instance**: Top-2 agent-specific few-shots
   - Format into structured prompt sections (🛡️ Core → 💡 Experience → 📝 Instance).
   - Retrieval scoring: `Score = Similarity × 0.7 + Normalized_Credit × 0.3`

### 5.3 Credit Score Updates (Post-Task Feedback)
- **Success**: `credit_score += 0.1`
- **Failure**: `credit_score -= 0.2`
- **Auto-Archive**: Items with `credit_score < -0.5` are marked `archived`.

### 5.4 Offline Evolution Loop
1. Fetch all active Experiences.
2. Cluster using DBSCAN (eps=0.5, min_samples=2, euclidean on normalized embeddings).
3. For each cluster (≥2 items): LLM Generator induces Core Knowledge candidate.
4. LLM Critic verifies against existing Core Knowledge:
   - **Create**: Novel principle → insert as `hypothesis`.
   - **Merge**: Refine existing → update & version up.
   - **Discard**: Redundant or conflicting.
5. Link source Experiences to created/merged Core Knowledge.

## 6. De-instantiation Rules
To ensure generalization, specific instance details are replaced with placeholders:
- **URLs**: `https://api.example.com/v1/...` → `{url}`
- **File Paths**: `/home/user/data.txt` → `{path}`
- **UUIDs**: `123e4567-e89b-12d3-a456-426614174000` → `{uuid}`
- **Numbers (≥3 digits)**: `12345` → `{number}`

## 7. Testing
- **Unit Tests**: `tests/test_skills.py` covers:
  - Legacy storage operations (add/merge skills)
  - Knowledge item CRUD operations
  - Layered distillation pipeline
  - Knowledge retrieval and formatting
  - Credit score updates
  - De-instantiation logic
- **Integration**: Verified via `agentic_pipeline/run_task.py` execution flow.

## 8. Dependencies
- `sqlite3`: Storage backend.
- `numpy`: Vector operations.
- `sentence-transformers`: Embedding generation (`all-MiniLM-L6-v2`, 384-dim).
- `sklearn`: DBSCAN clustering for evolution loop.
- `openai`: LLM interface for Teacher Model and Evolution Manager.
