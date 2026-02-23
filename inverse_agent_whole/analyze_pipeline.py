import json
import sys
import datetime
from collections import defaultdict

def parse_timestamp(ts_str):
    return datetime.datetime.fromisoformat(ts_str)

def analyze_log(file_path):
    print(f"Analyzing log file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            events = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: File not found.")
        return

    runs = []
    current_run = None
    
    for event in events:
        if event['type'] == 'start':
            if current_run:
                runs.append(current_run)
            current_run = {
                'start_time': parse_timestamp(event['timestamp']),
                'events': [],
                'phases': defaultdict(float),
                'timeline': []
            }
        
        if current_run:
            current_run['events'].append(event)

    if current_run:
        runs.append(current_run)

    if not runs:
        print("No 'start' events found in log.")
        return

    # Analyze each run
    for run_idx, run in enumerate(runs):
        print(f"\n{'='*20} Run {run_idx+1} {'='*20}")
        print(f"Start Time: {run['start_time']}")
        
        events = run['events']
        if not events:
            continue
            
        last_ts = parse_timestamp(events[0]['timestamp'])
        
        # State Machine
        current_phase = "Initialization"
        phase_start_ts = last_ts
        
        # We'll use specific events to switch phases
        # Phase 0: Start -> Phase 0 -> Sandbox
        # Phase 1: phase1 -> phase1_critic
        # Phase 2: phase2 -> execution
        # Phase 3: phase3 -> phase3_result
        
        for i in range(1, len(events)):
            evt = events[i]
            ts = parse_timestamp(evt['timestamp'])
            evt_type = evt['type']
            content = str(evt.get('content', ''))
            
            # Determine if phase changed
            new_phase = None
            
            if evt_type == 'phase0':
                if current_phase != 'Phase 0 (Dataset)': new_phase = 'Phase 0 (Dataset)'
            elif evt_type == 'phase1':
                new_phase = 'Phase 1 (Planning)'
            elif evt_type == 'phase2':
                new_phase = 'Phase 2 (Coding)'
            elif evt_type == 'phase3':
                new_phase = 'Phase 3 (Verification)'
            elif evt_type == 'elo':
                new_phase = 'ELO Evaluation'
            elif evt_type == 'success':
                new_phase = 'Success'
            elif evt_type == 'fail':
                new_phase = 'Failed'
                
            # Special handling for loop_start to detect phase if missed
            if evt_type == 'loop_start':
                if 'PLANNING' in content: new_phase = 'Phase 1 (Planning)'
                elif 'CODING' in content: new_phase = 'Phase 2 (Coding)'
                elif 'VERIFYING' in content: new_phase = 'Phase 3 (Verification)'

            if new_phase and new_phase != current_phase:
                duration = (ts - phase_start_ts).total_seconds()
                run['phases'][current_phase] += duration
                run['timeline'].append({
                    'phase': current_phase,
                    'duration': duration,
                    'start': phase_start_ts.strftime("%H:%M:%S"),
                    'end': ts.strftime("%H:%M:%S")
                })
                
                current_phase = new_phase
                phase_start_ts = ts
            
            last_ts = ts
            
        # Add last phase
        duration = (last_ts - phase_start_ts).total_seconds()
        run['phases'][current_phase] += duration
        run['timeline'].append({
            'phase': current_phase,
            'duration': duration,
            'start': phase_start_ts.strftime("%H:%M:%S"),
            'end': last_ts.strftime("%H:%M:%S")
        })
        
        total_duration = (last_ts - run['start_time']).total_seconds()
        print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
        
        # Print Summary Table
        print(f"\n{'-'*50}")
        print(f"{'Phase':<25} | {'Time (s)':<10} | {'%':<5}")
        print(f"{'-'*50}")
        
        # Order of phases for display
        ordered_phases = [
            'Initialization', 'Phase 0 (Dataset)', 
            'Phase 1 (Planning)', 'Phase 2 (Coding)', 
            'Phase 3 (Verification)', 'ELO Evaluation'
        ]
        # Add any other phases encountered
        for p in run['phases']:
            if p not in ordered_phases:
                ordered_phases.append(p)
                
        for phase in ordered_phases:
            if phase in run['phases']:
                dur = run['phases'][phase]
                pct = (dur / total_duration) * 100 if total_duration > 0 else 0
                print(f"{phase:<25} | {dur:<10.2f} | {pct:<5.1f}%")
        print(f"{'-'*50}")
        
        # Bottleneck Identification
        sorted_phases = sorted(run['phases'].items(), key=lambda x: x[1], reverse=True)
        bottleneck_phase, bottleneck_time = sorted_phases[0]
        print(f"\n🛑 Primary Bottleneck: {bottleneck_phase} ({bottleneck_time:.2f}s)")
        
        # Check for multiple iterations
        iterations = len([e for e in events if e['type'] == 'loop_start'])
        print(f"🔄 Total Iterations: {iterations}")
        
        # Check final status
        last_evt = events[-1]
        status = "Unknown"
        if any(e['type'] == 'success' for e in events):
            status = "✅ Solved"
        elif any(e['type'] == 'fail' for e in events):
            status = "❌ Failed"
        print(f"🏁 Final Status: {status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pipeline.py <log_file>")
        sys.exit(1)
    analyze_log(sys.argv[1])
