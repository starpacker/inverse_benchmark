'use client';

import { Github, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-zinc-800/40 mt-16">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="text-xl">🔬</span>
            <div>
              <span className="text-sm font-semibold text-white">Agent-Imaging</span>
              <p className="text-xs text-zinc-600">Computational Imaging Benchmark</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com/starpacker/inverse_benchmark"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-zinc-500 hover:text-white transition"
            >
              <Github size={14} /> GitHub <ExternalLink size={11} className="opacity-50" />
            </a>
            <span className="text-zinc-800">|</span>
            <a href="https://arxiv.org/abs/2501.00000" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-xs text-zinc-500 hover:text-white transition">
              📄 Paper <ExternalLink size={11} className="opacity-50" />
            </a>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-zinc-800/30 text-center">
          <p className="text-[11px] text-zinc-600 leading-relaxed max-w-lg mx-auto">
            If you use this benchmark in your research, please cite our paper.
            Agent-Imaging is an open benchmark — contributions welcome.
          </p>
          <p className="text-[10px] text-zinc-700 mt-3">
            © {new Date().getFullYear()} Agent-Imaging Team
          </p>
        </div>
      </div>
    </footer>
  );
}
