"""
Paper2Executable - Database ORM Models

SQLAlchemy models for the global paper index.
Implements the schema:
    papers_table = {
        "arxiv_id", "title", "abstract", "topics",
        "github_url", "data_status", "metrics", "last_verified"
    }
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    DateTime,
    Boolean,
    Integer,
    Enum as SAEnum,
    create_engine,
)

# JSON type with fallback for older SQLAlchemy
try:
    from sqlalchemy import JSON
except ImportError:
    from sqlalchemy import PickleType as JSON

from sqlalchemy.orm import Session

# Compat: declarative_base for SQLAlchemy < 2.0
try:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass
except ImportError:
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()


class Paper(Base):
    """Core paper record — one row per unique paper."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Identity ---
    arxiv_id = Column(String(32), unique=True, nullable=True, index=True)
    title = Column(Text, nullable=False, index=True)
    abstract = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)  # JSON-serialized list

    # --- Classification ---
    topics = Column(JSON, nullable=True)  # ["computational_imaging", "inverse_problems", ...]
    topic_confirmed = Column(Boolean, default=False)  # LLM-confirmed relevance

    # --- Code Discovery ---
    github_url = Column(String(512), nullable=True)
    github_discovered_by = Column(
        String(32), nullable=True
    )  # "local_parse" | "pwc_api" | "web_search"
    github_verified = Column(Boolean, default=False)  # LLM-verified link
    has_code = Column(Boolean, nullable=True)  # None = unknown

    # --- Data & Execution ---
    data_status = Column(
        String(32), nullable=True
    )  # "real_data" | "mock_only" | "partial" | None
    conda_env_name = Column(String(128), nullable=True)  # Created conda environment
    run_gt_status = Column(
        String(32), nullable=True
    )  # "success" | "failed" | "timeout" | "oom" | None
    evaluate_status = Column(String(32), nullable=True)

    # --- Metrics (sandbox-measured) ---
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    inference_time = Column(Float, nullable=True)  # seconds

    # --- Pipeline State ---
    pipeline_stage = Column(
        String(32), default="discovered"
    )  # discovered | code_found | consolidated | validated | failed
    failure_reason = Column(Text, nullable=True)
    debug_attempts = Column(Integer, default=0)

    # --- Provenance ---
    source = Column(String(64), nullable=True)  # "baseline_import" | "arxiv_crawl" | "manual"
    workspace_path = Column(String(512), nullable=True)  # Path to staged workspace
    output_path = Column(String(512), nullable=True)  # Path to final output

    # --- Timestamps ---
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_verified = Column(DateTime, nullable=True)

    def to_schema_dict(self) -> dict:
        """Export in the canonical paper2executable schema format."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "topics": self.topics or [],
            "github_url": self.github_url,
            "data_status": self.data_status,
            "metrics": {
                "psnr": self.psnr,
                "ssim": self.ssim,
                "rmse": self.rmse,
                "mse": self.mse,
                "inference_time": self.inference_time,
            },
            "pipeline_stage": self.pipeline_stage,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
        }

    def __repr__(self) -> str:
        return (
            f"<Paper(id={self.id}, arxiv_id={self.arxiv_id!r}, "
            f"title={self.title[:50]!r}..., stage={self.pipeline_stage!r})>"
        )


class DiscoveryLog(Base):
    """Audit trail for code discovery attempts."""

    __tablename__ = "discovery_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, nullable=False, index=True)

    level = Column(String(32), nullable=False)  # "local_parse" | "pwc_api" | "web_search"
    query_used = Column(Text, nullable=True)
    raw_result = Column(Text, nullable=True)  # JSON blob of raw API/search response
    candidate_url = Column(String(512), nullable=True)
    llm_verdict = Column(String(16), nullable=True)  # "match" | "none" | "uncertain"
    llm_reasoning = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class ExecutionLog(Base):
    """Record of sandbox execution attempts."""

    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, nullable=False, index=True)

    step = Column(String(64), nullable=False)  # "data_extract" | "code_consolidate" | "run_gt" | "evaluate" | "debug"
    attempt = Column(Integer, default=1)
    status = Column(String(32), nullable=False)  # "success" | "failed" | "timeout" | "oom"
    error_message = Column(Text, nullable=True)
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # OpenHands specifics
    openhands_session_id = Column(String(128), nullable=True)
    openhands_iterations = Column(Integer, nullable=True)
    trajectory_path = Column(String(512), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables(db_path: str = "database/papers.db") -> None:
    """Create all tables in the database."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine
