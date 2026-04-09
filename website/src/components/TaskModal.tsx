'use client';

import { useEffect, useCallback } from 'react';
import { X, Copy, Check, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import type { TaskData } from '@/app/page';
import { getCodeSnippet } from '@/data/codeSnippets';

/* ── Domain accent colors ── */
const DOMAIN_ACCENT: Record<string, string> = {
  A: '#06b6d4', B: '#8b5cf6', C: '#f59e0b', D: '#ef4444',
  E: '#22c55e', F: '#3b82f6', G: '#ec4899', H: '#f97316',
};

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/* ── Helper: image path for a task ── */
function getImagePath(task: TaskData): string {
  return `${BASE_PATH}/images/tasks/${task.images.folder}_${task.images.vis_result}`;
}

/* ── Code Block with copy ── */
function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  return (
    <div className="code-block relative group">
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800/50">
        <span className="text-[11px] text-zinc-500 font-medium tracking-wide uppercase">Python — Agent Implementation</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-cyan-400 transition"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-[12.5px] leading-[1.7] text-zinc-300/90">
        <code>{code}</code>
      </pre>
    </div>
  );
}

/* ── Metric Pill ── */
function MetricPill({ label, value, accent }: { label: string; value: string | number; accent: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-900/70 border border-zinc-800/60">
      <div className="w-2 h-2 rounded-full" style={{ background: accent }} />
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="text-sm font-semibold text-zinc-200 ml-auto">{value}</span>
    </div>
  );
}

function DescriptionBlock({ text, accent }: { text: string; accent: string }) {
  const lines = text.split('\n').map((line) => line.trim()).filter(Boolean);
  return (
    <div className="space-y-2.5">
      {lines.map((line, idx) => {
        const item = line.match(/^(\d+)\.\s*(?:\*\*(.+?)\*\*|([^:]+)):\s*(.*)$/);
        if (item) {
          const num = item[1];
          const label = (item[2] || item[3] || '').trim();
          const content = item[4].trim();
          return (
            <div key={`${idx}-${num}`} className="text-sm text-zinc-400 leading-relaxed">
              <span className="text-zinc-300">{num}.</span>{' '}
              <span className="font-semibold" style={{ color: accent }}>{label}</span>
              {content ? <>: {content}</> : null}
            </div>
          );
        }
        const cleaned = line.replace(/\*\*(.*?)\*\*/g, '$1');
        return (
          <p key={idx} className="text-sm text-zinc-400 leading-relaxed">
            {cleaned}
          </p>
        );
      })}
    </div>
  );
}

/* ── Main Modal ── */
interface TaskModalProps { task: TaskData; onClose: () => void; }

export default function TaskModal({ task, onClose }: TaskModalProps) {
  const accent = DOMAIN_ACCENT[task.domain] || '#06b6d4';

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // Prevent body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);

  const code = getCodeSnippet(task.id, task.title, task.domain_name);
  const imagePath = getImagePath(task);

  const hasMetrics = task.metrics && (task.metrics.psnr || task.metrics.ssim || task.metrics.eval_type);

  return (
    <div className="modal-overlay fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md" onClick={onClose}>
      <div className="modal-content w-full max-w-5xl max-h-[90vh] glass-card overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center gap-4 px-6 py-4 border-b border-zinc-800/50">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold" style={{ background: `${accent}18`, color: accent }}>
              {task.id}
            </div>
            <div className="min-w-0">
              <h2 className="text-lg font-bold text-white truncate">{task.title}</h2>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ background: `${accent}18`, color: accent }}>
                  {task.domain_name}
                </span>
                <span>Task {task.id_num}</span>
              </div>
            </div>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition">
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 lg:divide-x lg:divide-zinc-800/50">
            {/* Left: Formulation + Code */}
            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Problem Description</h3>
                <DescriptionBlock text={task.description} accent={accent} />
              </div>

              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Agent Code</h3>
                <CodeBlock code={code} />
              </div>
            </div>

            {/* Right: Image + Metrics */}
            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Visualization</h3>
                <div className="rounded-xl overflow-hidden border border-zinc-800/50 bg-black/30">
                  <img src={imagePath} alt={task.title} className="w-full h-auto" loading="lazy" />
                </div>
              </div>

              {hasMetrics && (
                <div>
                  <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Metrics</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {task.metrics.psnr && <MetricPill label="PSNR" value={task.metrics.psnr} accent={accent} />}
                    {task.metrics.ssim && <MetricPill label="SSIM" value={task.metrics.ssim} accent={accent} />}
                    {task.metrics.eval_type && <MetricPill label="Eval" value={task.metrics.eval_type} accent="#71717a" />}
                  </div>
                </div>
              )}

              {/* Domain Info */}
              <div className="metric-highlight">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full" style={{ background: accent }} />
                  <span className="text-sm font-semibold text-zinc-200">{task.domain_name}</span>
                </div>
                <p className="text-xs text-zinc-500">
                  Domain {task.domain} • This task evaluates the agent&apos;s ability to solve {task.title.toLowerCase()} problems.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
