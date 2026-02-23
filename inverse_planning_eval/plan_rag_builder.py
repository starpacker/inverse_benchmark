import os
import json
import yaml
import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inverse_rag_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("InverseRAG_Builder")

# --- 数据模型定义 ---

class InverseProblemTask(BaseModel):
    """
    数据容器：用于存储从文件读取的一对数据
    """
    task_id: str  # 文件名 (无后缀)
    description: str
    golden_plan: Dict[str, Any]

# --- 工具函数 ---

def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_llm_env(config_path, model_key):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    config = load_yaml(config_path)
    if model_key not in config['models']:
        raise ValueError(f"Model {model_key} not found in {config_path}")
    
    model_conf = config['models'][model_key]
    os.environ["OPENAI_API_KEY"] = model_conf['api_key']
    os.environ["OPENAI_BASE_URL"] = model_conf['base_url']
    logger.info(f"LLM Environment configured for [{model_key}]")
    return model_conf.get('model_name', model_key)

# --- 本地数据加载器 ---

class LocalDataLoader:
    def __init__(self, desc_dir: str, plan_dir: str):
        self.desc_dir = Path(desc_dir)
        self.plan_dir = Path(plan_dir)
    
    def load_all_tasks(self) -> List[InverseProblemTask]:
        """
        遍历 task_descriptions 目录，并在 golden_plans 中寻找同名文件。
        假设: Description 是 .txt 或 .md, Plan 是 .json
        """
        tasks = []
        if not self.desc_dir.exists() or not self.plan_dir.exists():
            raise FileNotFoundError(f"Data directories not found: {self.desc_dir} or {self.plan_dir}")

        # 获取所有描述文件
        desc_files = list(self.desc_dir.glob("*.*"))
        logger.info(f"Found {len(desc_files)} files in description directory.")

        for desc_file in desc_files:
            if desc_file.suffix not in ['.txt', '.md']:
                continue
                
            task_id = desc_file.stem # 获取文件名作为 ID (例如 'MRI_Recon')
            
            # 寻找对应的 JSON Plan
            plan_file = self.plan_dir / f"{task_id}.json"
            
            if not plan_file.exists():
                logger.warning(f"Missing Golden Plan for task: {task_id}. Skipping.")
                continue
            
            try:
                # 读取描述
                with open(desc_file, 'r', encoding='utf-8') as f:
                    description = f.read().strip()
                
                # 读取 Plan
                with open(plan_file, 'r', encoding='utf-8') as f:
                    plan_json = json.load(f)
                
                tasks.append(InverseProblemTask(
                    task_id=task_id,
                    description=description,
                    golden_plan=plan_json
                ))
            except Exception as e:
                logger.error(f"Error loading task {task_id}: {e}")
        
        logger.info(f"Successfully loaded {len(tasks)} matched tasks.")
        return tasks

# --- 核心：反问题算法抽象引擎 (Inverse Problem Abstraction) ---

class AbstractionEngine:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name
        
        # --- 针对计算成像/反问题的定制 Prompt ---
        self.SYSTEM_PROMPT = """You are a Principal Computational Imaging Scientist and Applied Mathematician.
Your specialty is formulating physics-based Inverse Problems.

Goal: Abstract a specific imaging/sensing task into its mathematical formulation.

RULES:
1. IGNORE specific hardware brands or commercial names.
2. IDENTIFY the Forward Model (e.g., Radon Transform, Helmholtz Equation, Fourier Transform).
3. IDENTIFY the Mathematical Structure (e.g., Linear/Non-linear System, Convex/Non-convex, Ill-posedness).
4. SPECIFY the Priors/Regularizers (e.g., Total Variation (TV), Sparsity in Wavelet domain, Deep Image Prior).
5. SUGGEST the Solver Class (e.g., ADMM, Gradient Descent, Plug-and-Play (PnP), Langevin Dynamics).

Output Format: A concise, dense mathematical paragraph describing the physics and the optimization problem.
"""

        # --- 针对反问题的 Few-Shot Examples ---
        self.USER_PROMPT_TEMPLATE = """
Transform the following task description into a Mathematical/Algorithmic Abstraction.

---
[Example 1]
Input: "We need to reconstruct high-quality images from undersampled MRI data acquired with spiral trajectories to shorten scan time. The images are expected to have piecewise smooth features."
Output: "Compressed Sensing (CS) Inverse Problem. The forward operator $A$ represents the Non-uniform Fast Fourier Transform (NUFFT) corresponding to the spiral trajectory. The problem is ill-posed and under-determined. Modeled as minimizing $||Ax - y||_2^2 + \lambda TV(x)$, where $y$ is k-space data. Solvable using Primal-Dual Hybrid Gradient (PDHG) or ADMM algorithms exploiting the sparsity of the gradient map."

[Example 2]
Input: "Reconstruct the image of a black hole from sparse interferometric data collected by telescopes around the globe. The atmosphere causes phase errors."
Output: "Sparse Aperture Synthesis / Radio Interferometry. Non-convex optimization problem involving Phase Closure constraints to handle atmospheric phase corruption. Forward model relates the image Fourier components (visibilities) to sparse UV-plane sampling. Optimization objectives include Maximum Entropy (MEM) or Regularized Maximum Likelihood (RML) with TSV (Total Squared Variation) regularization."
---

[Current Task]
Input: "{description}"
Output:
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_abstraction(self, description: str) -> str:
        """同步调用 LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(description=description)}
                ],
                temperature=0.0, # 必须为 0，保证严谨性和复现性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error abstracting task: {e}")
            raise e

# --- 向量数据库构建器 ---

class AlgorithmicRAGBuilder:
    def __init__(self, persist_dir: str = "./inverse_rag_db", collection_name: str = "inverse_algo_idx"):
        # 初始化持久化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # 创建或获取集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.openai_client = OpenAI()
        logger.info(f"ChromaDB initialized at {persist_dir}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def process_and_index(self, tasks: List[InverseProblemTask], abstraction_engine: AbstractionEngine):
        """
        线性处理流程 (无并发)，稳定可靠。
        """
        logger.info("Starting processing pipeline (Sequential Mode)...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        # 使用 tqdm 显示进度
        for task in tqdm(tasks, desc="Processing Tasks"):
            try:
                # 1. 核心：生成算法抽象
                # 这里会调用 LLM，因为是单线程，速度取决于 API 响应时间
                abstract_desc = abstraction_engine.generate_abstraction(task.description)
                
                # 2. 向量化 (基于抽象描述)
                vec = self.get_embedding(abstract_desc)
                
                # 3. 准备存储数据
                # 将 Plan 序列化存入 Metadata
                plan_str = json.dumps(task.golden_plan, ensure_ascii=False)
                
                ids.append(task.task_id)
                embeddings.append(vec)
                documents.append(abstract_desc) # 存入 Document 字段方便查看抽象结果
                metadatas.append({
                    "original_desc": task.description,
                    "golden_plan": plan_str,
                    "abstraction": abstract_desc
                })
                
                # 为了防止处理大量数据时内存溢出或程序崩溃，可以每处理一个就 log 一下
                logger.debug(f"Processed task: {task.task_id}")

            except Exception as e:
                logger.error(f"Failed to process task {task.task_id}: {e}")
                continue

        # 4. 批量写入数据库 (最后一次性写入，或者分批写入)
        if ids:
            logger.info(f"Upserting {len(ids)} vectors to ChromaDB...")
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info("Indexing Complete.")
        else:
            logger.warning("No data to index.")

    def inspect(self):
        """简单的验证函数"""
        print(f"\n--- Database Inspection ({self.collection.count()} entries) ---")
        if self.collection.count() > 0:
            peek = self.collection.peek(limit=1)
            print(f"ID: {peek['ids'][0]}")
            print(f"Abstraction: {peek['documents'][0]}")

# --- 主程序 ---

def main():
    # 1. 路径配置 (根据你的要求)
    DESC_DIR = "/data/yjh/task_descriptions"
    PLAN_DIR = "/data/yjh/golden_plans"
    CONFIG_PATH = "config2.yaml"
    
    # 2. 环境初始化
    try:
        model_name = setup_llm_env(CONFIG_PATH, "gpt-4-turbo") # 建议使用理解物理公式能力强的模型
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return

    # 3. 加载数据
    loader = LocalDataLoader(DESC_DIR, PLAN_DIR)
    try:
        tasks = loader.load_all_tasks()
        if not tasks:
            logger.error("No valid tasks matched. Exiting.")
            return
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # 4. 初始化引擎
    abstraction_engine = AbstractionEngine(model_name=model_name)
    rag_builder = AlgorithmicRAGBuilder(persist_dir="./inverse_rag_db")

    # 5. 执行构建
    rag_builder.process_and_index(tasks, abstraction_engine)
    
    # 6. 验证
    rag_builder.inspect()

if __name__ == "__main__":
    main()