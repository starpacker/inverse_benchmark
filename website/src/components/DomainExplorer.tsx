'use client';

import { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { DomainData, TaskData } from '@/app/page';

/* ── Color Maps ── */
const DOMAIN_COLORS: Record<string, string> = {
  A: '#06b6d4', B: '#8b5cf6', C: '#f59e0b', D: '#ef4444',
  E: '#22c55e', F: '#3b82f6', G: '#ec4899', H: '#f97316',
};
const DOMAIN_BG: Record<string, string> = {
  A: 'rgba(6,182,212,0.06)', B: 'rgba(139,92,246,0.06)', C: 'rgba(245,158,11,0.06)', D: 'rgba(239,68,68,0.06)',
  E: 'rgba(34,197,94,0.06)', F: 'rgba(59,130,246,0.06)', G: 'rgba(236,72,153,0.06)', H: 'rgba(249,115,22,0.06)',
};

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

function getImagePath(task: TaskData): string {
  return `${BASE_PATH}/images/tasks/${task.images.folder}_${task.images.vis_result}`;
}

interface Props {
  domains: [string, DomainData][];
  getTasksForDomain: (key: string) => TaskData[];
  onSelectTask: (task: TaskData) => void;
}

export default function DomainExplorer({ domains, getTasksForDomain, onSelectTask }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = (key: string) => setExpanded((prev) => (prev === key ? null : key));

  return (
    <section className="max-w-6xl mx-auto px-6 py-12" id="domains">
      <h2 className="text-3xl font-bold text-white mb-2">Explore by Domain</h2>
      <p className="text-zinc-500 text-sm mb-10">Click a domain to reveal its tasks. Click any task card for details.</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {domains.map(([key, domain]) => {
          const isExpanded = expanded === key;
          const accent = DOMAIN_COLORS[key] || '#06b6d4';
          const bg = DOMAIN_BG[key] || 'rgba(6,182,212,0.06)';
          const tasks = isExpanded ? getTasksForDomain(key) : [];

          return (
            <div key={key} className="col-span-1 sm:col-span-1">
              {/* Domain Card */}
              <div
                className="domain-card glass-card neon-border p-5 rounded-2xl"
                style={{ borderColor: isExpanded ? `${accent}33` : undefined }}
                onClick={() => toggle(key)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{domain.icon}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold tracking-wider" style={{ color: accent }}>
                          {key}
                        </span>
                        <h3 className="text-base font-semibold text-white">{domain.name_en}</h3>
                      </div>
                      <p className="text-xs text-zinc-500 mt-0.5">{domain.name_zh}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium px-2 py-1 rounded-full" style={{ background: bg, color: accent }}>
                      {domain.task_count} tasks
                    </span>
                    {isExpanded ? <ChevronDown size={16} className="text-zinc-500" /> : <ChevronRight size={16} className="text-zinc-500" />}
                  </div>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed line-clamp-2">{domain.desc}</p>
              </div>

              {/* Expanded Task Grid */}
              {isExpanded && tasks.length > 0 && (
                <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-3 animate-fade-in">
                  {tasks.map((task) => (
                    <div
                      key={task.id}
                      className="task-card rounded-xl overflow-hidden border border-zinc-800/50 bg-zinc-900/40 hover:bg-zinc-900/70"
                      onClick={(e) => { e.stopPropagation(); onSelectTask(task); }}
                    >
                      <div className="aspect-[4/3] relative bg-black/40 overflow-hidden">
                        <img
                          src={getImagePath(task)}
                          alt={task.title}
                          className="w-full h-full object-cover opacity-80 hover:opacity-100 transition"
                          loading="lazy"
                        />
                        <div className="absolute top-2 left-2">
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: `${accent}25`, color: accent }}>
                            {task.id}
                          </span>
                        </div>
                      </div>
                      <div className="p-3">
                        <p className="text-xs font-medium text-zinc-200 line-clamp-2 leading-snug">{task.title}</p>
                        {task.metrics?.psnr && (
                          <p className="text-[10px] text-zinc-600 mt-1">PSNR: {task.metrics.psnr}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
