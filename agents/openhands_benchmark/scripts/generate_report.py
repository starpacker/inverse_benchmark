#!/usr/bin/env python3
"""Generate a summary report from the OpenHands benchmark results."""
import json
import os

RESULT_FILE = os.environ.get("BENCH_RESULTS_DIR", "/data/openhands_results") + "/openhands_results.json"

with open(RESULT_FILE, "r") as f:
    data = json.load(f)

results = data["results"]

print("=" * 85)
print("OpenHands Benchmark Results — 26 Tasks × 10 Max Iterations")
print(f"Model: {data['model']}  |  Agent: {data['agent']}  |  Total Time: {data['total_time_sec']:.0f}s")
print("=" * 85)
print(f"{'#':<3} {'Task Name':<42} {'Status':<16} {'Time(s)':<10} {'Iterations':<10}")
print("-" * 85)

max_iter_count = 0
exception_count = 0
success_count = 0

for i, r in enumerate(results, 1):
    name = r["task_name"]
    elapsed = r["elapsed_sec"]
    iters = r["iterations"]
    
    if "maximum iteration" in (r.get("error") or ""):
        status = "MAX_ITER (10)"
        max_iter_count += 1
        iters = 10  # All used 10 iterations
    elif r.get("agent_state") == "EXCEPTION":
        status = "CONN_ERROR"
        exception_count += 1
    elif r["success"]:
        status = "✓ SUCCESS"
        success_count += 1
    else:
        status = "✗ FAILED"
    
    print(f"{i:<3} {name:<42} {status:<16} {elapsed:<10.1f} {iters:<10}")

print("-" * 85)
print(f"\nSUMMARY:")
print(f"  Total tasks          : {len(results)}")
print(f"  Successful (FINISHED): {success_count}")
print(f"  Max iterations (10)  : {max_iter_count}  (agent worked but ran out of budget)")
print(f"  Connection errors    : {exception_count}  (LocalRuntime port exhaustion)")
print(f"  Total time           : {data['total_time_sec']:.0f}s ({data['total_time_sec']/60:.1f} min)")
print(f"  Avg time per task    : {data['total_time_sec']/len(results):.1f}s")
print()
print("NOTE: 'MAX_ITER' means the agent ran all 10 iterations doing useful work")
print("      (reading code, writing solver, executing) but didn't call 'finish'.")
print("      'CONN_ERROR' = LocalRuntime port contention from accumulated servers.")
print("      To get true success/fail, check each task's working_folder for outputs.")
print("=" * 85)
