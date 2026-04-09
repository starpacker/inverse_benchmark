# website/ — Public Showcase Frontend

A **Next.js + TypeScript + Tailwind** website that publicly showcases the inverse problem benchmark: task descriptions, reference reconstructions, and side-by-side comparisons of agent / model outputs.

> Originally a standalone repo named `agent-imaging-website`. Merged here as `website/` on 2026-04-10 so the showcase lives next to the code that generated it.

---

## 🛠️ Tech stack

- **Next.js** (App Router) + **TypeScript**
- **Tailwind CSS**
- Static export — deploys via the `.github/workflows/deploy.yml` GitHub Action

---

## 📁 Structure

```
website/
├── public/
│   ├── data/tasks_db.json       ← task metadata + result links
│   └── images/
│       ├── tasks/               ← per-task visualization images
│       └── compare/             ← GT / input / reconstruction comparisons
├── src/
│   ├── app/                     ← Next.js App Router pages
│   ├── components/              ← reusable React components
│   └── data/                    ← data utilities
└── scripts/                     ← data ingestion / build scripts
```

---

## 🚀 Local dev

```bash
cd website
npm install
npm run dev
# open http://localhost:3000
```

For deployment, the GitHub Action in `.github/workflows/deploy.yml` builds and publishes the static export.

---

## 🔗 Where the data comes from

The website reads task descriptions and result images that are produced by the agents in [`../agents/`](../agents/) when run against the tasks in [`../tasks/`](../tasks/), graded by the harnesses in [`../harnesses/`](../harnesses/). The `tasks_db.json` is regenerated from the upstream task metadata.
