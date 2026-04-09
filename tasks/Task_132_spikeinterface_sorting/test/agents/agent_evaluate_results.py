import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import time

def evaluate_results(
    sorting,
    sorting_gt,
    recording_cmr,
    recording_raw,
    sorter_used: str,
    results_dir: str,
    bin_size_ms: float = 1.0,
) -> dict:
    """
    Evaluate sorting results against ground truth and generate visualizations.
    
    Args:
        sorting: Detected sorting object
        sorting_gt: Ground truth sorting object
        recording_cmr: Preprocessed recording
        recording_raw: Raw recording (for visualization)
        sorter_used: Name of the sorter used
        results_dir: Directory for saving results
        bin_size_ms: Time bin size for raster representation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    import spikeinterface.full as si
    
    os.makedirs(results_dir, exist_ok=True)
    
    # ============================================================
    # Compare with ground truth
    # ============================================================
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
    
    # ============================================================
    # Extract waveforms and compute template correlation
    # ============================================================
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
        import traceback
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
            import traceback
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
    
    # ============================================================
    # Save metrics
    # ============================================================
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
    
    # Save arrays
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
    
    # ============================================================
    # Visualization
    # ============================================================
    print("\n[7/7] Creating visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Panel A: Ground truth spike raster
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
    
    # Panel B: Detected spike raster
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
    
    # Panel C: Raw trace snippet
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
    
    # Panel D: Per-unit accuracy bar chart
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
    
    # Panel E: Precision & Recall grouped bar chart
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
    
    # Panel F: Agreement matrix
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
    
    # Panel G: Templates (if available)
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
    
    # Panel H: Summary metrics text
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
    
    # Panel I: Firing rate comparison
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
