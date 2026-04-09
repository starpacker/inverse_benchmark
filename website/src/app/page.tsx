'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import HeroSection from '@/components/HeroSection';
import DomainExplorer from '@/components/DomainExplorer';
import StatsBar from '@/components/StatsBar';
import TaskModal from '@/components/TaskModal';
import Footer from '@/components/Footer';

/* ── Type definitions ── */
export interface TaskData {
  id: string;
  id_num: number;
  name: string;
  domain: string;
  domain_name: string;
  title: string;
  description: string;
  metrics: {
    psnr?: number | string;
    ssim?: number | string;
    eval_type?: string;
    [key: string]: unknown;
  };
  images: { folder: string; vis_result: string };
}

export interface DomainData {
  name_en: string;
  name_zh: string;
  icon: string;
  desc: string;
  task_count: number;
  task_ids: number[];
}

export interface TasksDB {
  meta: { title: string; total_tasks: number; total_domains: number };
  domains: Record<string, DomainData>;
  tasks: Record<string, TaskData>;
}

/* ── Resolve basePath at runtime for fetch / img URLs ── */
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/* ── Page Component ── */
export default function Home() {
  const [db, setDb] = useState<TasksDB | null>(null);
  const [selectedTask, setSelectedTask] = useState<TaskData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${BASE_PATH}/data/tasks_db.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: TasksDB) => setDb(data))
      .catch((e) => setError(e.message));
  }, []);

  const domainEntries = useMemo(
    () => (db ? Object.entries(db.domains) : []),
    [db]
  );

  const getTasksForDomain = useCallback(
    (domainKey: string): TaskData[] => {
      if (!db) return [];
      const domain = db.domains[domainKey];
      if (!domain) return [];
      return domain.task_ids
        .map((id) => {
          const padded = String(id).padStart(2, '0');
          return db.tasks[`task_${padded}`];
        })
        .filter(Boolean);
    },
    [db]
  );

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="glass-card p-8 text-center max-w-md">
          <div className="text-4xl mb-4">⚠️</div>
          <h2 className="text-lg font-semibold text-red-400 mb-2">Failed to load data</h2>
          <p className="text-zinc-500 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!db) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-2 border-cyan-500/40 border-t-cyan-400 rounded-full animate-spin" />
          <p className="text-zinc-500 text-sm tracking-wide">Loading benchmark data…</p>
        </div>
      </div>
    );
  }

  return (
    <main className="grid-bg min-h-screen">
      <HeroSection totalTasks={db.meta.total_tasks} totalDomains={db.meta.total_domains} />
      <StatsBar domains={domainEntries} />
      <DomainExplorer
        domains={domainEntries}
        getTasksForDomain={getTasksForDomain}
        onSelectTask={setSelectedTask}
      />
      {selectedTask && (
        <TaskModal task={selectedTask} onClose={() => setSelectedTask(null)} />
      )}
      <Footer />
    </main>
  );
}
