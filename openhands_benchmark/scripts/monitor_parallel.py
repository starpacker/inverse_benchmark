#!/usr/bin/env python3
"""
Monitor and aggregate OpenHands parallel benchmark results.

Usage:
    python monitor_parallel.py                  # show status
    python monitor_parallel.py --aggregate      # merge all GPU results into one report
    python monitor_parallel.py --live           # live monitoring loop
"""
import os
import sys
import json
import glob
import time
import subprocess
from datetime import datetime
from pathlib import Path

RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", "/data/openhands_results")
LOG_DIR = f"{RESULTS_DIR}/logs"
TRAJ_DIR = f"{RESULTS_DIR}/trajectories"
PID_FILE = f"{RESULTS_DIR}/benchmark_pids.txt"


def check_running():
    """Check which GPU workers are still running."""
    result = subprocess.run(
        ["pgrep", "-f", "run_openhands_benchmark_v2"],
        capture_output=True, text=True
    )
    running_pids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

    gpu_status = {}
    if os.path.isfile(PID_FILE):
        with open(PID_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if ':' in line:
                    gid, pid = line.split(':')
                    gpu_status[int(gid)] = {
                        'pid': int(pid),
                        'running': pid in running_pids
                    }
    return gpu_status


def parse_log(gpu_id):
    """Parse a GPU log file for status."""
    log_path = os.path.join(LOG_DIR, f"gpu{gpu_id}.log")
    if not os.path.isfile(log_path):
        return None

    info = {
        'tasks_started': 0,
        'tasks_completed': 0,
        'successes': 0,
        'failures': 0,
        'current_task': None,
        'last_line': '',
    }

    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if '▶ Task:' in line:
                info['tasks_started'] += 1
                info['current_task'] = line.split('▶ Task:')[1].strip()
            elif '[✓]' in line or '[⚠]' in line:
                info['tasks_completed'] += 1
                if '[✓]' in line:
                    info['successes'] += 1
            elif '[✗]' in line:
                info['tasks_completed'] += 1
                info['failures'] += 1
            if line:
                info['last_line'] = line

    return info


def get_result_files():
    """Get all per-GPU result JSON files."""
    pattern = os.path.join(RESULTS_DIR, "openhands_results_gpu*.json")
    # Also match timestamped ones
    pattern2 = os.path.join(RESULTS_DIR, "openhands_results_gpu*_*.json")
    files = glob.glob(pattern) + glob.glob(pattern2)
    return sorted(set(files))


def show_status():
    """Show current benchmark status."""
    print("═" * 65)
    print("  OpenHands Parallel Benchmark — Status")
    print("═" * 65)

    gpu_status = check_running()
    total_running = sum(1 for v in gpu_status.values() if v['running'])
    total_done = sum(1 for v in gpu_status.values() if not v['running'])

    print(f"\n  Workers: {total_running} running, {total_done} finished")
    print()

    all_results = []
    for gpu_id in range(8):
        log_info = parse_log(gpu_id)
        status = gpu_status.get(gpu_id, {})
        is_running = status.get('running', False)
        pid = status.get('pid', '?')

        state_str = "🟢 RUNNING" if is_running else "⚫ DONE"
        if not log_info:
            state_str = "⭕ NOT STARTED"

        print(f"  GPU {gpu_id} [{state_str}] (PID {pid})")
        if log_info:
            print(f"    Tasks: {log_info['tasks_completed']}/{log_info['tasks_started']} "
                  f"(✓{log_info['successes']} ✗{log_info['failures']})")
            if is_running and log_info['current_task']:
                print(f"    Current: {log_info['current_task']}")
        print()

    # Aggregate from result files
    result_files = get_result_files()
    if result_files:
        print("─" * 65)
        print("  Result files found:")
        for f in result_files:
            print(f"    {os.path.basename(f)}")


def aggregate_results():
    """Merge all per-GPU results into a single report."""
    result_files = get_result_files()
    if not result_files:
        # Also try non-GPU-prefixed ones
        result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "openhands_results_*.json")))

    if not result_files:
        print("No result files found to aggregate.")
        return

    all_results = []
    for f in result_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
                results = data.get('results', [])
                all_results.extend(results)
                print(f"  Loaded {len(results)} results from {os.path.basename(f)}")
        except Exception as e:
            print(f"  ⚠ Error reading {f}: {e}")

    if not all_results:
        print("No results to aggregate.")
        return

    # Deduplicate by task_name (keep latest)
    seen = {}
    for r in all_results:
        seen[r['task_name']] = r
    results = sorted(seen.values(), key=lambda r: r['task_name'])

    # Build aggregate report
    successes = [r for r in results if r.get('success')]
    outputs = [r for r in results if r.get('output_exists')]
    failures = [r for r in results if not r.get('success')]

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "framework": "OpenHands",
        "agent": "CodeActAgent",
        "model": "cds/Claude-4.6-opus",
        "max_iterations": 50,
        "total_tasks": len(results),
        "successful": len(successes),
        "output_produced": len(outputs),
        "failed": len(failures),
        "success_rate_pct": round(100 * len(successes) / len(results), 1) if results else 0,
        "total_time_sec": round(sum(r.get('elapsed_sec', 0) for r in results), 1),
        "avg_time_sec": round(sum(r.get('elapsed_sec', 0) for r in results) / len(results), 1) if results else 0,
        "avg_iterations": round(sum(r.get('iterations', 0) for r in results) / len(results), 1) if results else 0,
        "results": results,
    }

    # Print summary
    print()
    print("═" * 65)
    print("  AGGREGATED BENCHMARK RESULTS")
    print("═" * 65)
    print(f"  Tasks        : {report['total_tasks']}")
    print(f"  Success      : {report['successful']} ({report['success_rate_pct']}%)")
    print(f"  Output.npy   : {report['output_produced']} produced")
    print(f"  Failed       : {report['failed']}")
    print(f"  Total Time   : {report['total_time_sec']:.0f}s ({report['total_time_sec']/60:.1f} min)")
    print(f"  Avg Time     : {report['avg_time_sec']:.0f}s per task")
    print(f"  Avg Iterations: {report['avg_iterations']:.1f}")
    print()

    # Per-task detail
    print("  Per-task results:")
    print(f"  {'Task':<40s} {'Status':>8s} {'Time':>8s} {'Iter':>5s} {'Eval':>20s}")
    print("  " + "─" * 85)
    for r in results:
        icon = "✓" if r.get('success') else ("⚠" if r.get('output_exists') else "✗")
        eval_str = ""
        if r.get('eval_result'):
            ev = r['eval_result']
            if isinstance(ev, dict):
                # Pick first 2 metrics
                parts = [f"{k}={v}" for k, v in list(ev.items())[:2]]
                eval_str = ", ".join(parts)
            else:
                eval_str = str(ev)[:20]
        time_str = f"{r.get('elapsed_sec', 0):.0f}s"
        iter_str = str(r.get('iterations', '?'))
        print(f"  {r['task_name']:<40s} {icon:>8s} {time_str:>8s} {iter_str:>5s} {eval_str:>20s}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "openhands_benchmark_FINAL.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Aggregate saved: {out_path}")

    # Also save as CSV for easy comparison
    csv_path = os.path.join(RESULTS_DIR, "openhands_benchmark_FINAL.csv")
    with open(csv_path, 'w') as f:
        headers = ['task_name', 'success', 'output_exists', 'elapsed_sec', 'iterations',
                   'effective_actions', 'error_events', 'agent_state', 'eval_result', 'error']
        f.write(','.join(headers) + '\n')
        for r in results:
            row = [
                r.get('task_name', ''),
                str(r.get('success', False)),
                str(r.get('output_exists', False)),
                str(r.get('elapsed_sec', '')),
                str(r.get('iterations', '')),
                str(r.get('effective_actions', '')),
                str(r.get('error_events', '')),
                str(r.get('agent_state', '')),
                json.dumps(r.get('eval_result', '')) if r.get('eval_result') else '',
                str(r.get('error', '')).replace(',', ';'),
            ]
            f.write(','.join(row) + '\n')
    print(f"  📄 CSV saved: {csv_path}")


def live_monitor():
    """Live monitoring loop."""
    try:
        while True:
            os.system('clear')
            show_status()

            gpu_status = check_running()
            if gpu_status and all(not v['running'] for v in gpu_status.values()):
                print("\n  ✅ All workers finished!")
                aggregate_results()
                break

            print(f"\n  [Refreshing every 30s... Ctrl+C to stop]")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n  Monitoring stopped. Workers continue in background.")


if __name__ == "__main__":
    if '--aggregate' in sys.argv:
        aggregate_results()
    elif '--live' in sys.argv:
        live_monitor()
    else:
        show_status()
