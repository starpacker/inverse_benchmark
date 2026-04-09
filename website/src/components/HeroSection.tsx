'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  FlaskConical, Github, ExternalLink, ChevronRight,
  Microscope, Camera, Activity, Globe2, Telescope,
  Rainbow, Cog, Zap
} from 'lucide-react';

/* ── Constants ── */
const COMPARE_TASKS = [
  { id: '44', title: 'Digital Holographic Microscopy', domain: 'Optical Microscopy', hasInput: true },
  { id: '15', title: 'Lensless ADMM Reconstruction', domain: 'Computational Photography', hasInput: true },
  { id: '43', title: 'Abel Inversion (PyAbel)', domain: 'X-ray / CT', hasInput: false },
] as const;

const DOMAIN_ICONS = [Microscope, Camera, Zap, Activity, Globe2, Telescope, Rainbow, Cog];
const DOMAIN_LABELS = [
  'Optical Microscopy', 'Computational Photography', 'X-ray / CT',
  'Medical Imaging', 'Geophysics', 'Astrophysics',
  'Spectroscopy / Scattering', 'Mechanics',
];

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/* ── Helper: Stat Card ── */
function StatCard({ value, label, icon }: { value: string; label: string; icon: React.ReactNode }) {
  return (
    <div className="glass-card p-5 flex items-center gap-4 group hover:border-cyan-500/25 transition">
      <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500/10 to-teal-500/10 flex items-center justify-center text-cyan-400 group-hover:from-cyan-500/20 group-hover:to-teal-500/20 transition">
        {icon}
      </div>
      <div>
        <div className="text-2xl font-bold gradient-text leading-none">{value}</div>
        <div className="text-xs text-zinc-500 mt-1 tracking-wide uppercase">{label}</div>
      </div>
    </div>
  );
}

/* ── Helper: Compare Teaser (custom implementation) ── */
function CompareTeaser({ taskId, title, hasInput }: { taskId: string; title: string; hasInput: boolean }) {
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const leftImg = hasInput ? `${BASE_PATH}/images/compare/task${taskId}_input.png` : `${BASE_PATH}/images/compare/task${taskId}_gt.png`;
  const rightImg = `${BASE_PATH}/images/compare/task${taskId}_recon.png`;
  const leftLabel = hasInput ? 'Input' : 'Ground Truth';
  const rightLabel = 'Reconstruction';

  const updatePosition = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    const handleMove = (e: MouseEvent | TouchEvent) => {
      e.preventDefault();
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      updatePosition(clientX);
    };
    const handleUp = () => setIsDragging(false);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    window.addEventListener('touchmove', handleMove, { passive: false });
    window.addEventListener('touchend', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleUp);
    };
  }, [isDragging, updatePosition]);

  return (
    <div className="flex flex-col gap-2">
      <div
        ref={containerRef}
        className="teaser-container relative select-none"
        style={{ height: '260px', cursor: isDragging ? 'col-resize' : 'ew-resize' }}
        onMouseDown={(e) => { setIsDragging(true); updatePosition(e.clientX); }}
        onTouchStart={(e) => { setIsDragging(true); updatePosition(e.touches[0].clientX); }}
      >
        {/* Right (full) */}
        <img src={rightImg} alt={rightLabel} className="absolute inset-0 w-full h-full object-cover" draggable={false} />
        {/* Left (clipped) */}
        <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderPos}%` }}>
          <img src={leftImg} alt={leftLabel} className="absolute inset-0 w-full h-full object-cover" style={{ minWidth: containerRef.current ? `${containerRef.current.offsetWidth}px` : '100%' }} draggable={false} />
        </div>
        {/* Slider line */}
        <div className="absolute top-0 bottom-0" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}>
          <div className="w-0.5 h-full bg-white/80 shadow-lg" />
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white/90 backdrop-blur-sm shadow-lg flex items-center justify-center">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 3L2 8L5 13M11 3L14 8L11 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
          </div>
        </div>
        {/* Labels */}
        <div className="absolute bottom-2 left-3 bg-black/70 text-xs text-zinc-300 px-2 py-0.5 rounded backdrop-blur-sm pointer-events-none">
          {leftLabel}
        </div>
        <div className="absolute bottom-2 right-3 bg-black/70 text-xs text-cyan-300 px-2 py-0.5 rounded backdrop-blur-sm pointer-events-none">
          {rightLabel}
        </div>
      </div>
      <div className="text-center">
        <p className="text-xs font-medium text-zinc-300">Task {taskId}</p>
        <p className="text-[11px] text-zinc-500">{title}</p>
      </div>
    </div>
  );
}

/* ── Main Component ── */
interface HeroProps { totalTasks: number; totalDomains: number; }

export default function HeroSection({ totalTasks, totalDomains }: HeroProps) {
  return (
    <section className="relative overflow-hidden pb-4">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-cyan-500/[0.03] blur-[120px] animate-pulse-slow" />
        <div className="absolute -bottom-40 -right-40 w-[500px] h-[500px] rounded-full bg-teal-500/[0.03] blur-[120px] animate-pulse-slow-delay" />
      </div>

      <div className="max-w-6xl mx-auto px-6 pt-16 pb-8 relative z-10">
        {/* Badges */}
        <div className="flex flex-wrap items-center gap-3 mb-8 animate-fade-in">
          <span className="badge">
            <FlaskConical size={14} className="text-cyan-400" />
            Computational Imaging Benchmark
          </span>
          <a href="https://github.com/starpacker/inverse_benchmark" target="_blank" rel="noopener noreferrer" className="badge hover:text-cyan-300">
            <Github size={14} /> GitHub <ExternalLink size={11} className="opacity-50" />
          </a>
          <a href="https://arxiv.org/abs/2501.00000" target="_blank" rel="noopener noreferrer" className="badge hover:text-purple-300">
            📄 arXiv Paper <ExternalLink size={11} className="opacity-50" />
          </a>
        </div>

        {/* Title */}
        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-5 animate-slide-up">
          <span className="text-white">Agent</span>
          <span className="text-white">-</span>
          <span className="gradient-text">Imaging</span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg sm:text-xl text-zinc-400 max-w-3xl leading-relaxed mb-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          A universal agent for solving&nbsp;
          <span className="text-cyan-300 font-medium">{totalTasks} computational imaging inverse problems</span> across&nbsp;
          <span className="text-teal-300 font-medium">{totalDomains} scientific domains</span> — from optical microscopy to astrophysical imaging.
        </p>

        {/* Abstract */}
        <div className="glass-card p-6 max-w-4xl mb-10 animate-slide-up" style={{ animationDelay: '0.15s' }}>
          <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <ChevronRight size={14} className="text-cyan-400" /> Abstract
          </h3>
          <p className="text-sm leading-relaxed text-zinc-400">
            Computational imaging inverse problems involve recovering latent physical quantities from indirect measurements.
            Traditional approaches require extensive domain expertise, custom algorithms, and manual hyper-parameter tuning for each specific modality.
            <strong className="text-zinc-200"> Agent-Imaging</strong> is a comprehensive benchmark that evaluates whether autonomous AI agents — powered by large language models — can independently understand,
            implement, and solve diverse inverse problems across the full spectrum of computational imaging. Our benchmark spans
            <strong className="text-zinc-200"> {totalTasks} tasks</strong> over <strong className="text-zinc-200">{totalDomains} scientific domains</strong>,
            each requiring the agent to interpret the forward model, choose or invent a reconstruction algorithm, write working code, and achieve quantitative accuracy.
          </p>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-12 animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <StatCard value={String(totalTasks)} label="Inverse Problems" icon={<FlaskConical size={20} />} />
          <StatCard value={String(totalDomains)} label="Scientific Domains" icon={<Globe2 size={20} />} />
          <StatCard value="10+" label="LLM Agents Tested" icon={<Zap size={20} />} />
          <StatCard value="1200+" label="Code Evaluations" icon={<Cog size={20} />} />
        </div>

        {/* Domain Icons */}
        <div className="flex flex-wrap gap-3 mb-12 animate-slide-up" style={{ animationDelay: '0.25s' }}>
          {DOMAIN_ICONS.map((Icon, i) => (
            <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-full border border-zinc-800/60 bg-zinc-900/30 text-xs text-zinc-500 hover:border-cyan-500/20 hover:text-zinc-300 transition">
              <Icon size={14} className="text-cyan-500/70" />
              {DOMAIN_LABELS[i]}
            </div>
          ))}
        </div>

        {/* Compare Sliders */}
        <div className="animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Zap size={14} className="text-cyan-400" /> Example Reconstructions — Drag to Compare
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {COMPARE_TASKS.map((t) => (
              <CompareTeaser key={t.id} taskId={t.id} title={t.title} hasInput={t.hasInput} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
