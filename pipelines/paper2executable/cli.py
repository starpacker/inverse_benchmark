"""
Paper2Executable CLI

Command-line interface for the pipeline.
"""

import asyncio
import json
import click
import yaml
from pathlib import Path

from orchestrator import PipelineOrchestrator
from utils.logging_utils import setup_logging

# Load config
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Initialize logging
setup_logging(level=config["general"]["log_level"])
orchestrator = PipelineOrchestrator(config)


@click.group()
def cli():
    """Paper2Executable: Scientific Paper to Code Pipeline."""
    pass


@cli.command()
def init_baseline():
    """Import existing benchmark dataset into the database."""
    click.echo("Initializing baseline data...")
    stats = orchestrator.import_baseline()
    click.echo(f"Import complete: {stats}")


@cli.command()
@click.option("--arxiv-id", required=True, help="ArXiv ID (e.g., 2403.12345)")
def process_paper(arxiv_id):
    """Process a single paper by ArXiv ID (original OpenHands flow)."""
    click.echo(f"Processing paper: {arxiv_id}")
    asyncio.run(orchestrator.run_pipeline(arxiv_id=arxiv_id))


@cli.command()
@click.option("--arxiv-id", required=True, help="ArXiv ID (e.g., 2108.10257)")
@click.option("--skip-env", is_flag=True, help="Skip conda env creation")
def run(arxiv_id, skip_env):
    """End-to-end pipeline: ArXiv ID → working-folder + gt-code + env + task-desc + report."""
    click.echo(f"[E2E] Processing: {arxiv_id}")
    result = orchestrator.process_paper_e2e(
        arxiv_id=arxiv_id,
        skip_env_create=skip_env,
    )
    click.echo(f"\n{'='*60}")
    click.echo(f"Status: {result.get('status', '?')}")
    if result.get("outputs"):
        for k, v in result["outputs"].items():
            click.echo(f"  {k}: {v}")
    # Dump full result to stdout as JSON
    click.echo(f"\nFull result JSON:")
    click.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@cli.command()
@click.argument("papers", nargs=-1, required=True)
@click.option("--skip-env", is_flag=True, help="Skip conda env creation")
def run_batch(papers, skip_env):
    """Process multiple papers. Each arg: ARXIV_ID.

    Example: python cli.py run-batch 2108.10257 2209.14687
    """
    entries = []
    for p in papers:
        arxiv_id = p.strip()
        entries.append({"arxiv_id": arxiv_id, "repo_url": ""})

    click.echo(f"[Batch] Processing {len(entries)} papers...")
    results = orchestrator.process_batch(entries, skip_env_create=skip_env)

    # Save batch results
    out_path = Path(config["general"]["workspace_base"]) / "batch_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"\nBatch results saved → {out_path}")


@cli.command()
def status():
    """Show pipeline status and statistics."""
    stats = orchestrator.db.get_pipeline_stats()
    click.echo(f"Pipeline Statistics:")
    click.echo(f"  Total papers: {stats['total']}")
    for stage, count in stats.get("stages", {}).items():
        click.echo(f"  {stage}: {count}")
    if stats.get("avg_psnr"):
        click.echo(f"  Avg PSNR: {stats['avg_psnr']:.2f} dB")


@cli.command()
@click.option("--topic", default="computational imaging", help="Topic or keywords to search on arXiv")
@click.option("--max-results", default=5, help="Maximum number of papers to fetch")
def discover(topic, max_results):
    """Search arXiv for papers matching TOPIC and process them through the pipeline."""
    click.echo(f"Searching arXiv for topic '{topic}' (max {max_results})...")
    import arxiv

    # use default sorting
    search = arxiv.Search(query=topic, max_results=max_results)
    entries = []
    for paper in search.results():
        arxiv_id = paper.get_short_id()
        title = paper.title
        click.echo(f"  found {arxiv_id}: {title}")
        entries.append({"arxiv_id": arxiv_id, "repo_url": ""})

    if not entries:
        click.echo("No papers found, exiting.")
        return

    click.echo(f"Processing {len(entries)} papers through pipeline...")
    results = orchestrator.process_batch(entries)
    out_path = Path(config["general"]["workspace_base"]) / "discover_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"Discovery results saved → {out_path}")


@cli.command()
@click.option("--topics", "-t", multiple=True, default=None,
              help="Topics to search (can specify multiple: -t 'inverse problem' -t 'image reconstruction')")
@click.option("--max-per-topic", default=30, type=int, help="Max papers per topic (default: 30)")
@click.option("--days-back", default=60, type=int, help="Only papers from last N days (default: 60)")
@click.option("--require-code/--no-require-code", default=True, help="Only papers with code")
@click.option("--require-python/--no-require-python", default=True, help="Only Python-based papers")
@click.option("--skip-pwc", is_flag=True, help="Skip PapersWithCode API check (faster)")
@click.option("--top-n", default=20, type=int, help="Show top N papers")
@click.option("--auto-queue", is_flag=True, help="Automatically queue top papers for processing")
@click.option("--queue-count", default=5, type=int, help="Number of papers to auto-queue (default: 5)")
def collect(topics, max_per_topic, days_back, require_code, require_python, skip_pwc, top_n, auto_queue, queue_count):
    """Collect relevant papers from ArXiv based on topics.
    
    Default topics: computational imaging, inverse problem, image reconstruction,
    phase retrieval, ptychography, super resolution, etc.
    
    Examples:
        python cli.py collect
        python cli.py collect -t "ptychography" -t "phase retrieval" --days-back 90
        python cli.py collect --auto-queue --queue-count 3
    """
    from tools.arxiv_collector import ArxivCollector
    from database.manager import DatabaseManager
    
    click.echo("=" * 70)
    click.echo("ArXiv Paper Collector - Computational Imaging / Inverse Problems")
    click.echo("=" * 70)
    
    # Initialize
    db = DatabaseManager(config["general"]["workspace_base"] + "/database/papers.db")
    
    collector = ArxivCollector(
        topics=list(topics) if topics else None,  # None = use defaults
        db=db,
        output_dir=config["general"]["workspace_base"] + "/collected_papers",
        days_back=days_back,
        require_code=require_code,
        require_python=require_python,
    )
    
    click.echo(f"Topics: {collector.topics}")
    click.echo(f"Filters: code={require_code}, python={require_python}, days={days_back}, skip_pwc={skip_pwc}")
    click.echo("-" * 70)
    
    # Collect
    papers = collector.collect(max_per_topic=max_per_topic, skip_pwc_check=skip_pwc)
    
    if not papers:
        click.echo("No papers found matching criteria.")
        return
    
    # Save results
    output_path = collector.save_results(papers)
    
    # Print summary
    collector.print_summary(papers, top_n=top_n)
    
    # Auto-queue if requested
    if auto_queue and papers:
        queue_papers = papers[:queue_count]
        click.echo(f"\n{'='*70}")
        click.echo(f"Auto-queuing top {len(queue_papers)} papers for processing...")
        click.echo(f"{'='*70}")
        
        entries = [{"arxiv_id": p.arxiv_id, "repo_url": p.github_url} for p in queue_papers]
        results = orchestrator.process_batch(entries, skip_env_create=True)
        
        # Summary
        success = sum(1 for r in results if r.get("status") == "completed")
        click.echo(f"\nProcessing complete: {success}/{len(results)} successful")
        
        # Save queue results
        queue_out = Path(config["general"]["workspace_base"]) / "auto_queue_results.json"
        with open(queue_out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        click.echo(f"Queue results saved → {queue_out}")


@cli.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.option("--top-n", default=5, type=int, help="Process top N papers from the collection")
@click.option("--skip-env", is_flag=True, help="Skip conda env creation")
def process_collected(json_file, top_n, skip_env):
    """Process papers from a collected JSON file.
    
    Example:
        python cli.py process-collected /data/.../collected_20260303_120000.json --top-n 10
    """
    with open(json_file) as f:
        data = json.load(f)
    
    papers = data.get("papers", [])
    if not papers:
        click.echo("No papers in the collection file.")
        return
    
    # Sort by relevance and take top N
    papers = sorted(papers, key=lambda p: p.get("relevance_score", 0), reverse=True)[:top_n]
    
    click.echo(f"Processing top {len(papers)} papers from {json_file}")
    for i, p in enumerate(papers, 1):
        click.echo(f"  {i}. [{p.get('relevance_score', 0):.1f}] {p['arxiv_id']}: {p['title'][:50]}...")
    
    entries = [{"arxiv_id": p["arxiv_id"], "repo_url": p.get("github_url", "")} for p in papers]
    results = orchestrator.process_batch(entries, skip_env_create=skip_env)
    
    success = sum(1 for r in results if r.get("status") == "completed")
    click.echo(f"\nProcessing complete: {success}/{len(results)} successful")


if __name__ == "__main__":
    cli()
