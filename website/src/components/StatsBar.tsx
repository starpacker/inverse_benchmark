'use client';

import type { DomainData } from '@/app/page';

const DOMAIN_COLORS: Record<string, string> = {
  A: '#06b6d4', B: '#8b5cf6', C: '#f59e0b', D: '#ef4444',
  E: '#22c55e', F: '#3b82f6', G: '#ec4899', H: '#f97316',
};

interface Props { domains: [string, DomainData][]; }

export default function StatsBar({ domains }: Props) {
  return (
    <section className="max-w-6xl mx-auto px-6 py-8">
      <div className="glass-card p-5 rounded-2xl">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {domains.map(([key, domain]) => {
            const accent = DOMAIN_COLORS[key] || '#06b6d4';
            return (
              <a
                key={key}
                href="#domains"
                className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl hover:bg-zinc-800/40 transition group"
              >
                <span className="text-lg">{domain.icon}</span>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-zinc-300 truncate group-hover:text-white transition">
                    {domain.name_en}
                  </div>
                  <div className="text-[11px] font-bold" style={{ color: accent }}>
                    {domain.task_count} tasks
                  </div>
                </div>
              </a>
            );
          })}
        </div>
      </div>
    </section>
  );
}
