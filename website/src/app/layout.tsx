import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Agent-Imaging — Computational Imaging Benchmark',
  description: 'A comprehensive multi-task benchmark for evaluating autonomous agents on computational imaging inverse problems. 83 tasks across 8 domains.',
  keywords: ['computational imaging', 'benchmark', 'inverse problems', 'AI agents', 'deep learning'],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
          crossOrigin="anonymous"
        />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔬</text></svg>" />
      </head>
      <body className="min-h-screen">
        {children}
      </body>
    </html>
  );
}
