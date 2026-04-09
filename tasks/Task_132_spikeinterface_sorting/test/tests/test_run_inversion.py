import sys
import os
import dill
import numpy as np
import traceback
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Inject the evaluate_results function (Reference B)
# ============================================================
def evaluate_results(
    sorting,
    sorting_gt,
    recording_cmr,
    recording_raw,
    sorter_used: str,
    results_dir: str,
    bin_size_ms: float = 1.0,
) -> dict:
    import spikeinterface.full as si

    os.makedirs(results_dir, exist_ok=True)

    print("\n[4/7] Comparing with ground truth...")
    t0 = time.time()

    comp = si.compare_sorter_to_ground_truth(
        sorting_gt, sorting,
        exhaustive_gt=True,
        match_score=0.5,
    )

    perf = comp.get_performance(method='by_unit', output='pandas')
    print(f"\n  Performance by GT unit:")
    print(perf.to_string())

    accuracies = perf['accuracy'].values.astype(float)
    precisions = perf['precision'].values.astype(float)
    recalls = perf['recall'].values.astype(float)

    print(f"\n  Mean accuracy:  {np.mean(accuracies):.4f}")
    print(f"  Mean precision: {np.mean(precisions):.4f}")
    print(f"  Mean recall:    {np.mean(recalls):.4f}")
    print(f"  Well-detected (acc>0.8): {int(np.sum(accuracies > 0.8))}/{len(accuracies)}")
    print(f"  Comparison took {time.time() - t0:.1f}s")

    hungarian_match = comp.hungarian_match_12

    print("\n[5/7] Extracting waveforms and computing template correlation...")
    t0 = time.time()

    templates = None
    has_templates = False
    template_cc = None

    try:
        sorting_analyzer = si.create_sorting_analyzer(
            sorting, recording_cmr,
            format='memory',
            sparse=False,
        )
        sorting_analyzer.compute('random_spikes', max_spikes_per_unit=500, seed=42)
        sorting_analyzer.compute('waveforms', ms_before=1.0, ms_after=2.0)
        sorting_analyzer.compute('templates')
        ext = sorting_analyzer.get_extension('templates')
        templates = ext.get_data()
        print(f"  Extracted templates shape: {templates.shape}")
        has_templates = True
    except Exception as e:
        print(f"  Template extraction failed: {e}")
        traceback.print_exc()

    templates_gt = None
    if has_templates and templates is not None:
        try:
            sa_gt = si.create_sorting_analyzer(
                sorting_gt, recording_cmr,
                format='memory',
                sparse=False,
            )
            sa_gt.compute('random_spikes', max_spikes_per_unit=500, seed=42)
            sa_gt.compute('waveforms', ms_before=1.0, ms_after=2.0)
            sa_gt.compute('templates')
            templates_gt = sa_gt.get_extension('templates').get_data()
        except Exception as e3:
            print(f"  GT template extraction failed: {e3}")
            traceback.print_exc()

        if templates_gt is not None and templates is not None:
            matched_ccs = []
            for gt_idx, gt_uid in enumerate(sorting_gt.unit_ids):
                matched_uid = hungarian_match.get(gt_uid, -1)
                if matched_uid is not None and matched_uid != -1:
                    det_idx = list(sorting.unit_ids).index(matched_uid)
                    gt_template = templates_gt[gt_idx].flatten()
                    det_template = templates[det_idx].flatten()
                    cc = np.corrcoef(gt_template, det_template)[0, 1]
                    matched_ccs.append(float(cc))
            if matched_ccs:
                template_cc = float(np.mean(matched_ccs))
                print(f"  Mean template correlation: {template_cc:.4f}")

    print(f"  Waveform analysis took {time.time() - t0:.1f}s")

    print("\n[6/7] Saving metrics and arrays...")

    metrics = {
        "task_name": "spikeinterface_sorting",
        "task_number": 132,
        "inverse_problem": "spike_sorting_blind_source_separation",
        "sorter_used": sorter_used,
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "median_accuracy": float(np.median(accuracies)),
        "num_gt_units": int(sorting_gt.get_num_units()),
        "num_detected_units": int(sorting.get_num_units()),
        "num_well_detected": int(np.sum(accuracies > 0.8)),
        "num_false_positive_units": max(0, int(sorting.get_num_units()) - int(np.sum(accuracies > 0.0))),
        "per_unit_accuracy": {str(uid): float(acc) for uid, acc in zip(sorting_gt.unit_ids, accuracies)},
        "per_unit_precision": {str(uid): float(prec) for uid, prec in zip(sorting_gt.unit_ids, precisions)},
        "per_unit_recall": {str(uid): float(rec) for uid, rec in zip(sorting_gt.unit_ids, recalls)},
    }

    if template_cc is not None:
        metrics["mean_template_correlation"] = template_cc

    metrics["primary_metric_name"] = "accuracy"
    metrics["primary_metric_value"] = float(np.mean(accuracies))

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")

    sampling_frequency = recording_raw.get_sampling_frequency()
    num_samples = recording_raw.get_num_samples()
    bin_size_samples = int(bin_size_ms * sampling_frequency / 1000.0)
    n_bins = int(np.ceil(num_samples / bin_size_samples))

    gt_raster = np.zeros((sorting_gt.get_num_units(), n_bins), dtype=np.float32)
    for i, uid in enumerate(sorting_gt.unit_ids):
        spike_train = sorting_gt.get_unit_spike_train(uid)
        bins = spike_train // bin_size_samples
        bins = bins[bins < n_bins]
        gt_raster[i, bins] = 1.0

    recon_raster = np.zeros((sorting_gt.get_num_units(), n_bins), dtype=np.float32)
    for i, uid in enumerate(sorting_gt.unit_ids):
        matched_uid = hungarian_match.get(uid, -1)
        if matched_uid is not None and matched_uid != -1:
            spike_train = sorting.get_unit_spike_train(matched_uid)
            bins = spike_train // bin_size_samples
            bins = bins[bins < n_bins]
            recon_raster[i, bins] = 1.0

    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_raster)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_raster)
    print(f"  Saved ground_truth.npy shape={gt_raster.shape}")
    print(f"  Saved reconstruction.npy shape={recon_raster.shape}")

    print("\n[7/7] Creating visualization...")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax_raster_gt = fig.add_subplot(gs[0, 0])
    t_max_raster = 5.0
    sample_max = int(t_max_raster * sampling_frequency)
    for i, uid in enumerate(sorting_gt.unit_ids):
        spike_train = sorting_gt.get_unit_spike_train(uid)
        spike_times = spike_train[spike_train < sample_max] / sampling_frequency
        ax_raster_gt.scatter(spike_times, np.ones_like(spike_times) * i, s=1, c='k', marker='|')
    ax_raster_gt.set_xlabel('Time (s)')
    ax_raster_gt.set_ylabel('GT Unit')
    ax_raster_gt.set_title('(A) Ground Truth Spike Raster')
    ax_raster_gt.set_xlim(0, t_max_raster)
    ax_raster_gt.set_yticks(range(sorting_gt.get_num_units()))

    ax_raster_det = fig.add_subplot(gs[0, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, max(sorting.get_num_units(), 1)))
    for i, uid in enumerate(sorting.unit_ids):
        spike_train = sorting.get_unit_spike_train(uid)
        spike_times = spike_train[spike_train < sample_max] / sampling_frequency
        ax_raster_det.scatter(spike_times, np.ones_like(spike_times) * i, s=1,
                              c=[colors[i % len(colors)]], marker='|')
    ax_raster_det.set_xlabel('Time (s)')
    ax_raster_det.set_ylabel('Detected Unit')
    ax_raster_det.set_title(f'(B) Detected Spike Raster ({sorter_used})')
    ax_raster_det.set_xlim(0, t_max_raster)

    ax_trace = fig.add_subplot(gs[0, 2])
    trace_dur = 0.1
    n_trace_samples = int(trace_dur * sampling_frequency)
    traces = recording_raw.get_traces(start_frame=0, end_frame=n_trace_samples)
    time_axis = np.arange(n_trace_samples) / sampling_frequency * 1000
    n_show_ch = min(4, traces.shape[1])
    for ch in range(n_show_ch):
        offset = ch * 150
        ax_trace.plot(time_axis, traces[:, ch] + offset, linewidth=0.5, color='k')
    ax_trace.set_xlabel('Time (ms)')
    ax_trace.set_ylabel('Channel (offset)')
    ax_trace.set_title('(C) Raw Recording (100ms)')
    ax_trace.set_xlim(0, trace_dur * 1000)

    ax_acc = fig.add_subplot(gs[1, 0])
    unit_labels = [str(uid) for uid in sorting_gt.unit_ids]
    bar_colors = ['#2ca02c' if a > 0.8 else '#ff7f0e' if a > 0.5 else '#d62728' for a in accuracies]
    ax_acc.bar(unit_labels, accuracies, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_acc.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Well-detected (0.8)')
    ax_acc.axhline(y=np.mean(accuracies), color='blue', linestyle=':', alpha=0.7,
                   label=f'Mean ({np.mean(accuracies):.2f})')
    ax_acc.set_xlabel('GT Unit ID')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('(D) Sorting Accuracy per GT Unit')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=8)

    ax_pr = fig.add_subplot(gs[1, 1])
    x = np.arange(len(unit_labels))
    width = 0.35
    ax_pr.bar(x - width / 2, precisions, width, label='Precision', color='#1f77b4', alpha=0.8)
    ax_pr.bar(x + width / 2, recalls, width, label='Recall', color='#ff7f0e', alpha=0.8)
    ax_pr.set_xlabel('GT Unit ID')
    ax_pr.set_ylabel('Score')
    ax_pr.set_title('(E) Precision & Recall per GT Unit')
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(unit_labels)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(fontsize=8)

    ax_agree = fig.add_subplot(gs[1, 2])
    try:
        agreement = comp.agreement_scores
        if hasattr(agreement, 'values'):
            agree_mat = agreement.values.astype(float)
        else:
            agree_mat = np.array(agreement, dtype=float)
        im = ax_agree.imshow(agree_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax_agree.set_xlabel('Detected Unit')
        ax_agree.set_ylabel('GT Unit')
        ax_agree.set_title('(F) Agreement Matrix')
        plt.colorbar(im, ax=ax_agree, shrink=0.8, label='Agreement')
    except Exception as e:
        ax_agree.text(0.5, 0.5, f'Agreement matrix\nnot available:\n{e}',
                      ha='center', va='center', transform=ax_agree.transAxes, fontsize=9)
        ax_agree.set_title('(F) Agreement Matrix')

    ax_tmpl = fig.add_subplot(gs[2, 0])
    if has_templates and templates is not None:
        n_show = min(5, templates.shape[0])
        best_ch = np.argmax(np.abs(templates).max(axis=1), axis=1)
        t_samples = np.arange(templates.shape[1]) / sampling_frequency * 1000
        for i in range(n_show):
            ax_tmpl.plot(t_samples, templates[i, :, best_ch[i]], label=f'Unit {sorting.unit_ids[i]}')
        ax_tmpl.set_xlabel('Time (ms)')
        ax_tmpl.set_ylabel('Amplitude (µV)')
        ax_tmpl.set_title('(G) Detected Unit Templates')
        ax_tmpl.legend(fontsize=7, ncol=2)
    else:
        ax_tmpl.text(0.5, 0.5, 'Templates not available', ha='center', va='center',
                     transform=ax_tmpl.transAxes)
        ax_tmpl.set_title('(G) Detected Unit Templates')

    ax_summary = fig.add_subplot(gs[2, 1])
    ax_summary.axis('off')
    summary_text = (
        f"Task 132: Spike Sorting\n"
        f"{'─' * 35}\n"
        f"Sorter: {sorter_used}\n"
        f"Recording: {recording_raw.get_num_channels()} ch × "
        f"{recording_raw.get_total_duration():.0f}s @ {sampling_frequency/1000:.0f} kHz\n"
        f"\n"
        f"GT Units:       {sorting_gt.get_num_units()}\n"
        f"Detected Units: {sorting.get_num_units()}\n"
        f"Well-detected:  {int(np.sum(accuracies > 0.8))}/{len(accuracies)}\n"
        f"\n"
        f"Mean Accuracy:  {np.mean(accuracies):.4f}\n"
        f"Mean Precision: {np.mean(precisions):.4f}\n"
        f"Mean Recall:    {np.mean(recalls):.4f}\n"
    )
    if template_cc is not None:
        summary_text += f"Template CC:    {template_cc:.4f}\n"
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax_fr = fig.add_subplot(gs[2, 2])
    gt_rates = []
    det_rates = []
    for i, uid in enumerate(sorting_gt.unit_ids):
        gt_rate = len(sorting_gt.get_unit_spike_train(uid)) / recording_raw.get_total_duration()
        gt_rates.append(gt_rate)
        matched_uid = hungarian_match.get(uid, -1)
        if matched_uid is not None and matched_uid != -1:
            det_rate = len(sorting.get_unit_spike_train(matched_uid)) / recording_raw.get_total_duration()
        else:
            det_rate = 0.0
        det_rates.append(det_rate)

    ax_fr.bar(x - width / 2, gt_rates, width, label='GT', color='#2ca02c', alpha=0.8)
    ax_fr.bar(x + width / 2, det_rates, width, label='Detected', color='#9467bd', alpha=0.8)
    ax_fr.set_xlabel('GT Unit ID')
    ax_fr.set_ylabel('Firing Rate (Hz)')
    ax_fr.set_title('(I) Firing Rate Comparison')
    ax_fr.set_xticks(x)
    ax_fr.set_xticklabels(unit_labels)
    ax_fr.legend(fontsize=8)

    fig.suptitle('Task 132: Spike Sorting from Extracellular Recordings\n'
                 '(Inverse Problem: Blind Source Separation of Neural Signals)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved reconstruction_result.png")

    return metrics


# ============================================================
# Lightweight evaluation for comparing agent vs standard
# Uses spikeinterface comparison without full visualization
# ============================================================
def evaluate_sorting_lightweight(sorting, sorting_gt):
    """
    Quick evaluation: compare sorting to ground truth, return mean accuracy.
    """
    import spikeinterface.full as si
    
    comp = si.compare_sorter_to_ground_truth(
        sorting_gt, sorting,
        exhaustive_gt=True,
        match_score=0.5,
    )
    perf = comp.get_performance(method='by_unit', output='pandas')
    accuracies = perf['accuracy'].values.astype(float)
    mean_acc = float(np.mean(accuracies))
    return mean_acc


def main():
    # ============================================================
    # 1. Parse data paths
    # ============================================================
    data_paths = ['/data/yjh/spikeinterface_sorting_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    outer_data_path = None
    inner_data_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(p)
        else:
            outer_data_path = p

    # Also scan directory for inner data files
    std_data_dir = os.path.dirname(outer_data_path)
    if os.path.isdir(std_data_dir):
        for fname in os.listdir(std_data_dir):
            full_path = os.path.join(std_data_dir, fname)
            if full_path in data_paths:
                continue
            if fname.endswith('.pkl') and 'parent_function_run_inversion' in fname:
                inner_data_paths.append(full_path)
            elif fname.endswith('.pkl') and 'parent_run_inversion' in fname:
                inner_data_paths.append(full_path)

    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")

    is_chained = len(inner_data_paths) > 0

    # ============================================================
    # 2. Load outer data
    # ============================================================
    print("\nLoading outer data...")
    with open(outer_data_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data['args']
    outer_kwargs = outer_data['kwargs']
    std_output = outer_data['output']

    print(f"  Function name: {outer_data['func_name']}")
    print(f"  Number of args: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")

    # Extract recording_cmr from args
    recording_cmr = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('recording_cmr')

    # ============================================================
    # 3. We need ground truth sorting for evaluation
    # The ground truth isn't directly in run_inversion's args/output.
    # We need to recover it from the MEArec recording or from the
    # broader pipeline data.
    # ============================================================
    
    # Try to get sorting_gt from the recording (MEArec recordings contain GT)
    sorting_gt = None
    recording_raw = None
    
    try:
        import spikeinterface.full as si
        
        # Check if there's a study/ground truth file in the working directory
        working_dir = outer_args[1] if len(outer_args) > 1 else outer_kwargs.get('working_dir', '/tmp/test_inversion_workdir')
        print(f"  Working dir from data: {working_dir}")
        
        # Look for the MEArec file or ground truth in the broader pipeline
        # Check standard data directory for other pickle files that might contain GT
        for fname in os.listdir(std_data_dir):
            if fname.endswith('.pkl') and fname != os.path.basename(outer_data_path):
                try:
                    fpath = os.path.join(std_data_dir, fname)
                    with open(fpath, 'rb') as f:
                        other_data = dill.load(f)
                    # Check if this data contains sorting_gt or recording_raw
                    if 'output' in other_data:
                        out = other_data['output']
                        if isinstance(out, tuple):
                            for item in out:
                                if hasattr(item, 'get_unit_spike_train') and sorting_gt is None:
                                    # Could be sorting_gt
                                    if hasattr(item, 'unit_ids'):
                                        print(f"  Found potential sorting_gt in {fname}")
                                        sorting_gt = item
                                if hasattr(item, 'get_traces') and recording_raw is None:
                                    print(f"  Found potential recording_raw in {fname}")
                                    recording_raw = item
                        elif hasattr(out, 'get_unit_spike_train') and sorting_gt is None:
                            sorting_gt = out
                        elif hasattr(out, 'get_traces') and recording_raw is None:
                            recording_raw = out
                    # Also check args
                    if 'args' in other_data:
                        for item in other_data['args']:
                            if hasattr(item, 'get_unit_spike_train') and sorting_gt is None:
                                if hasattr(item, 'unit_ids'):
                                    print(f"  Found potential sorting_gt in args of {fname}")
                                    sorting_gt = item
                            if hasattr(item, 'get_traces') and hasattr(item, 'get_num_channels') and recording_raw is None:
                                # Distinguish raw from CMR - raw typically isn't preprocessed
                                print(f"  Found potential recording in args of {fname}")
                                if recording_raw is None:
                                    recording_raw = item
                    if 'kwargs' in other_data:
                        for k, item in other_data['kwargs'].items():
                            if hasattr(item, 'get_unit_spike_train') and sorting_gt is None:
                                print(f"  Found potential sorting_gt in kwargs[{k}] of {fname}")
                                sorting_gt = item
                            if hasattr(item, 'get_traces') and recording_raw is None:
                                print(f"  Found potential recording_raw in kwargs[{k}] of {fname}")
                                recording_raw = item
                except Exception as e:
                    print(f"  Could not load {fname}: {e}")
                    continue
    except Exception as e:
        print(f"  Error scanning for GT: {e}")
        traceback.print_exc()

    # If recording_raw not found, use recording_cmr as fallback
    if recording_raw is None:
        print("  WARNING: recording_raw not found, using recording_cmr as fallback")
        recording_raw = recording_cmr

    # ============================================================
    # 4. Run the agent's function
    # ============================================================
    print("\n" + "="*60)
    print("Running agent's run_inversion...")
    print("="*60)

    # Import the agent function
    from agent_run_inversion import run_inversion

    # Create a fresh working directory for the agent
    agent_working_dir = '/tmp/test_agent_inversion_workdir'
    os.makedirs(agent_working_dir, exist_ok=True)

    try:
        if not is_chained:
            # Direct execution
            # Reconstruct args, replacing working_dir with our temp dir
            agent_args = list(outer_args)
            agent_kwargs = dict(outer_kwargs)
            
            # Replace working_dir to avoid conflicts
            if len(agent_args) > 1:
                agent_args[1] = agent_working_dir
            elif 'working_dir' in agent_kwargs:
                agent_kwargs['working_dir'] = agent_working_dir
            else:
                agent_args = [recording_cmr, agent_working_dir]

            agent_output = run_inversion(*agent_args, **agent_kwargs)
            agent_sorting, agent_sorter_used = agent_output
            
            # Standard output
            std_sorting, std_sorter_used = std_output
            
            print(f"\nAgent: sorter={agent_sorter_used}, units={agent_sorting.get_num_units()}")
            print(f"Standard: sorter={std_sorter_used}, units={std_sorting.get_num_units()}")
        else:
            # Chained execution
            agent_args = list(outer_args)
            agent_kwargs = dict(outer_kwargs)
            if len(agent_args) > 1:
                agent_args[1] = agent_working_dir
            elif 'working_dir' in agent_kwargs:
                agent_kwargs['working_dir'] = agent_working_dir

            agent_operator = run_inversion(*agent_args, **agent_kwargs)
            
            # Load inner data
            inner_path = inner_data_paths[0]
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            
            agent_output = agent_operator(*inner_args, **inner_kwargs)
            std_output_inner = inner_data['output']
            
            # For chained, the final result is what we evaluate
            if isinstance(agent_output, tuple) and len(agent_output) == 2:
                agent_sorting, agent_sorter_used = agent_output
            else:
                agent_sorting = agent_output
                agent_sorter_used = "unknown"
            
            if isinstance(std_output_inner, tuple) and len(std_output_inner) == 2:
                std_sorting, std_sorter_used = std_output_inner
            else:
                std_sorting = std_output_inner
                std_sorter_used = "unknown"

    except Exception as e:
        print(f"\nERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # 5. Evaluation
    # ============================================================
    print("\n" + "="*60)
    print("Evaluating results...")
    print("="*60)

    # If we have ground truth, use proper evaluation
    if sorting_gt is not None:
        print("\nEvaluating AGENT output against ground truth...")
        try:
            agent_results_dir = '/tmp/test_agent_results'
            os.makedirs(agent_results_dir, exist_ok=True)
            
            agent_metrics = evaluate_results(
                sorting=agent_sorting,
                sorting_gt=sorting_gt,
                recording_cmr=recording_cmr,
                recording_raw=recording_raw,
                sorter_used=agent_sorter_used,
                results_dir=agent_results_dir,
            )
            score_agent = agent_metrics['mean_accuracy']
        except Exception as e:
            print(f"  Full evaluation failed for agent: {e}")
            traceback.print_exc()
            try:
                score_agent = evaluate_sorting_lightweight(agent_sorting, sorting_gt)
            except Exception as e2:
                print(f"  Lightweight evaluation also failed: {e2}")
                traceback.print_exc()
                score_agent = -1.0

        print("\nEvaluating STANDARD output against ground truth...")
        try:
            std_results_dir = '/tmp/test_std_results'
            os.makedirs(std_results_dir, exist_ok=True)
            
            std_metrics = evaluate_results(
                sorting=std_sorting,
                sorting_gt=sorting_gt,
                recording_cmr=recording_cmr,
                recording_raw=recording_raw,
                sorter_used=std_sorter_used,
                results_dir=std_results_dir,
            )
            score_std = std_metrics['mean_accuracy']
        except Exception as e:
            print(f"  Full evaluation failed for standard: {e}")
            traceback.print_exc()
            try:
                score_std = evaluate_sorting_lightweight(std_sorting, sorting_gt)
            except Exception as e2:
                print(f"  Lightweight evaluation also failed: {e2}")
                traceback.print_exc()
                score_std = -1.0
    else:
        # No ground truth available - compare agent vs standard directly
        print("\nWARNING: No ground truth found. Comparing agent output vs standard output directly.")
        print("Using standard sorting as pseudo ground truth...")
        
        try:
            score_agent = evaluate_sorting_lightweight(agent_sorting, std_sorting)
            score_std = 1.0  # Standard compared to itself would be perfect
            print(f"  Agent agreement with standard: {score_agent:.4f}")
        except Exception as e:
            print(f"  Direct comparison failed: {e}")
            traceback.print_exc()
            # Fallback: compare number of units and basic stats
            n_agent = agent_sorting.get_num_units()
            n_std = std_sorting.get_num_units()
            print(f"  Agent units: {n_agent}, Standard units: {n_std}")
            
            if n_agent == 0 and n_std > 0:
                score_agent = 0.0
                score_std = 1.0
            elif n_std == 0:
                score_agent = 1.0
                score_std = 1.0
            else:
                # Rough comparison based on unit count ratio
                ratio = min(n_agent, n_std) / max(n_agent, n_std)
                score_agent = ratio
                score_std = 1.0

    # ============================================================
    # 6. Verification & Reporting
    # ============================================================
    print("\n" + "="*60)
    print(f"Scores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    print("="*60)

    # Primary metric is accuracy (higher is better)
    # Allow 10% margin of degradation
    margin = 0.90  # agent must be at least 90% of standard

    if score_std <= 0:
        # Standard had zero accuracy - just check agent isn't negative
        print("Standard score is 0 or negative. Checking agent is non-negative...")
        if score_agent >= 0:
            print("PASS: Agent performed at least as well as standard (both ~0).")
            sys.exit(0)
        else:
            print("FAIL: Agent score is negative.")
            sys.exit(1)

    relative_performance = score_agent / score_std if score_std > 0 else float('inf')
    print(f"Relative performance (agent/std): {relative_performance:.4f}")
    print(f"Required threshold: {margin:.4f}")

    if score_agent >= score_std * margin:
        print(f"\nPASS: Agent accuracy ({score_agent:.4f}) >= {margin*100:.0f}% of standard ({score_std:.4f})")
        print(f"  Threshold: {score_std * margin:.4f}")
        sys.exit(0)
    else:
        print(f"\nFAIL: Agent accuracy ({score_agent:.4f}) < {margin*100:.0f}% of standard ({score_std:.4f})")
        print(f"  Threshold: {score_std * margin:.4f}")
        print(f"  Deficit: {score_std * margin - score_agent:.4f}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nUNHANDLED EXCEPTION: {e}")
        traceback.print_exc()
        sys.exit(1)