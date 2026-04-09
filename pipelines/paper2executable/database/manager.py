"""
Paper2Executable - Database Manager

CRUD operations for the paper index database.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Paper, DiscoveryLog, ExecutionLog


class DatabaseManager:
    """Manages all database operations for the paper2executable pipeline."""

    def __init__(self, db_path: str = "database/papers.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self._Session()

    # ── Paper CRUD ──────────────────────────────────────────────

    def add_paper(self, **kwargs) -> Paper:
        """Insert a new paper record. Returns the created Paper."""
        with self.get_session() as session:
            paper = Paper(**kwargs)
            session.add(paper)
            session.commit()
            session.refresh(paper)
            return paper

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """Lookup by ArXiv ID (primary dedup key)."""
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

    def get_paper_by_id(self, paper_id: int) -> Optional[Paper]:
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.id == paper_id).first()

    def get_paper_by_title(self, title: str) -> Optional[Paper]:
        """Exact title lookup."""
        with self.get_session() as session:
            return session.query(Paper).filter(Paper.title == title).first()

    def search_papers_by_title(self, title_fragment: str) -> list[Paper]:
        """Fuzzy title search using SQL LIKE."""
        with self.get_session() as session:
            return (
                session.query(Paper)
                .filter(Paper.title.ilike(f"%{title_fragment}%"))
                .all()
            )

    def get_papers_by_stage(self, stage: str) -> list[Paper]:
        """Get all papers at a given pipeline stage."""
        with self.get_session() as session:
            return (
                session.query(Paper)
                .filter(Paper.pipeline_stage == stage)
                .all()
            )

    def get_all_papers(self) -> list[Paper]:
        with self.get_session() as session:
            return session.query(Paper).all()

    def update_paper(self, paper_id: int, **kwargs) -> Optional[Paper]:
        """Update fields on an existing paper."""
        with self.get_session() as session:
            paper = session.query(Paper).filter(Paper.id == paper_id).first()
            if paper is None:
                return None
            for key, value in kwargs.items():
                if hasattr(paper, key):
                    setattr(paper, key, value)
            paper.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(paper)
            return paper

    def paper_exists(self, arxiv_id: Optional[str] = None, title: Optional[str] = None) -> bool:
        """Check if a paper already exists by ArXiv ID or exact title."""
        with self.get_session() as session:
            conditions = []
            if arxiv_id:
                conditions.append(Paper.arxiv_id == arxiv_id)
            if title:
                conditions.append(Paper.title == title)
            if not conditions:
                return False
            return session.query(Paper).filter(or_(*conditions)).first() is not None

    def count_papers(self, stage: Optional[str] = None) -> int:
        """Count papers, optionally filtered by stage."""
        with self.get_session() as session:
            q = session.query(func.count(Paper.id))
            if stage:
                q = q.filter(Paper.pipeline_stage == stage)
            return q.scalar()

    # ── Discovery Log ──────────────────────────────────────────

    def log_discovery(self, paper_id: int, **kwargs) -> DiscoveryLog:
        """Record a code discovery attempt."""
        with self.get_session() as session:
            entry = DiscoveryLog(paper_id=paper_id, **kwargs)
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def get_discovery_logs(self, paper_id: int) -> list[DiscoveryLog]:
        with self.get_session() as session:
            return (
                session.query(DiscoveryLog)
                .filter(DiscoveryLog.paper_id == paper_id)
                .order_by(DiscoveryLog.created_at)
                .all()
            )

    # ── Execution Log ──────────────────────────────────────────

    def log_execution(self, paper_id: int, **kwargs) -> ExecutionLog:
        """Record an execution attempt."""
        with self.get_session() as session:
            entry = ExecutionLog(paper_id=paper_id, **kwargs)
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def get_execution_logs(self, paper_id: int) -> list[ExecutionLog]:
        with self.get_session() as session:
            return (
                session.query(ExecutionLog)
                .filter(ExecutionLog.paper_id == paper_id)
                .order_by(ExecutionLog.created_at)
                .all()
            )

    # ── Stats ──────────────────────────────────────────────────

    def get_pipeline_stats(self) -> dict:
        """Get aggregate pipeline statistics."""
        with self.get_session() as session:
            total = session.query(func.count(Paper.id)).scalar()
            stages = {}
            for stage_name in [
                "discovered", "code_found", "consolidated",
                "validated", "failed"
            ]:
                stages[stage_name] = (
                    session.query(func.count(Paper.id))
                    .filter(Paper.pipeline_stage == stage_name)
                    .scalar()
                )
            validated_with_metrics = (
                session.query(func.count(Paper.id))
                .filter(Paper.pipeline_stage == "validated", Paper.psnr.isnot(None))
                .scalar()
            )
            avg_psnr = (
                session.query(func.avg(Paper.psnr))
                .filter(Paper.psnr.isnot(None))
                .scalar()
            )
            return {
                "total": total,
                "stages": stages,
                "validated_with_metrics": validated_with_metrics,
                "avg_psnr": round(avg_psnr, 2) if avg_psnr else None,
            }

    # ── Export ─────────────────────────────────────────────────

    def export_all(self, output_path: str = "papers_export.json") -> str:
        """Export all papers in the canonical schema format."""
        papers = self.get_all_papers()
        data = [p.to_schema_dict() for p in papers]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path
