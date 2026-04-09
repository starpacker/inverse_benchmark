# Agent-Imaging Website Working Guide (`agent-imaging-website`)

## 1) Purpose and Scope

This document is the operational guide for teammates and autonomous coding agents working on this repository.

Primary goals:
- Understand how the site is structured and deployed.
- Modify UI/data/scripts safely without breaking GitHub Pages hosting.
- Know the highest-impact revision opportunities and common pitfalls.

Repository type:
- Static-exported Next.js 14 site for browsing benchmark tasks.
- Data-driven UI sourced from `public/data/tasks_db.json`.
- Assets served from `public/images/tasks` and `public/images/compare`.

---

## 2) Tech Stack and Runtime Constraints

Core stack:
- Next.js `14.2.x` + React `18` + TypeScript.
- Tailwind CSS + custom global CSS.
- Static export mode (`output: 'export'`) for GitHub Pages.

Important configuration:
- `next.config.js` sets:
  - `output: 'export'`
  - `basePath: '/agent-imaging-website'`
  - `assetPrefix: '/agent-imaging-website'`
  - `trailingSlash: true`
  - `env.NEXT_PUBLIC_BASE_PATH = '/agent-imaging-website'`

Implication:
- All fetch/image URLs should be built from `NEXT_PUBLIC_BASE_PATH` (already done in key components).
- Absolute paths hardcoded as `/...` in JSX/TSX will break deployment under GitHub Pages subpath.

---

## 3) Repository Map and Responsibilities

### App shell
- `src/app/layout.tsx`
  - Global metadata, KaTeX stylesheet injection, dark theme root.
- `src/app/page.tsx`
  - Client entry page.
  - Loads `tasks_db.json`.
  - Owns state: loaded DB, selected task modal, load error.
  - Wires together all major components.

### UI components
- `src/components/HeroSection.tsx`
  - Hero copy, benchmark summary cards, compare-image teaser slider.
- `src/components/StatsBar.tsx`
  - Domain quick-nav summary strip.
- `src/components/DomainExplorer.tsx`
  - Expandable domain cards and per-domain task grids.
- `src/components/TaskModal.tsx`
  - Task detail modal, visualization, metrics, code snippet display.
- `src/components/Footer.tsx`
  - Footer links and citation text.

### Data and content
- `public/data/tasks_db.json`
  - Canonical runtime data source.
- `src/data/codeSnippets.ts`
  - Curated code snippets for selected tasks; fallback generator for others.
- `public/images/tasks/*`
  - Thumbnail/result images used in domain task cards + modal.
- `public/images/compare/*`
  - Inputs/GT/recon images for hero compare teaser.

### Asset/data generation scripts
- `scripts/generate_tasks_db.py`
  - Offline generator for `tasks_db.json` from external sources.
- `scripts/generate_compare_images.py`
  - Offline generator/copy script for compare teaser images.

### Deployment
- `.github/workflows/deploy.yml`
  - Build on `main`, upload `out/`, deploy via GitHub Pages actions.

---

## 4) Runtime Data Contract

The app expects this JSON shape:
- `meta`: `{ title, total_tasks, total_domains }`
- `domains`: keyed by domain letter (`A`...`H`)
  - required fields: `name_en`, `name_zh`, `icon`, `desc`, `task_count`, `task_ids`
- `tasks`: keyed by `task_XX`
  - required fields used by UI:
    - `id`, `id_num`, `domain`, `domain_name`, `title`, `description`
    - `images.folder`, `images.vis_result`
  - optional fields:
    - `metrics.psnr`, `metrics.ssim`, `metrics.eval_type`

Image URL construction used in UI:
- Task card/modal:
  - `.../images/tasks/${task.images.folder}_${task.images.vis_result}`
- Compare teaser:
  - `.../images/compare/task${id}_input|gt|recon.png`

If `images.vis_result` becomes missing/`null`, current UI will build broken image URLs. Guarding this is a recommended improvement.

---

## 5) Current UI Data Flow

1. `page.tsx` fetches `public/data/tasks_db.json`.
2. Derived domain entries are computed by `Object.entries(db.domains)`.
3. For a selected domain, tasks are resolved by `domain.task_ids` mapped to `db.tasks['task_${padded}']`.
4. `DomainExplorer` displays cards and opens `TaskModal` on task click.
5. `TaskModal` loads:
   - visualization image path from task data.
   - code snippet from `getCodeSnippet(task.id, task.title, task.domain_name)`.

Notes:
- Domain/task color coding is duplicated in multiple components; centralizing constants is a maintainability win.
- All major page content currently executes as client-side components (`'use client'`).

---

## 6) Build, Dev, Deploy Workflow

Package scripts:
- `npm run dev` → local dev.
- `npm run build` → Next build + static export to `out/`.
- `npm run start` → Next server mode (mostly for non-export checks).

Deployment path:
- Push to `main` triggers GitHub Actions deploy.
- Site base URL assumed as `/agent-imaging-website/`.

Local verification checklist before merge:
1. `npm ci`
2. `npm run build`
3. Confirm `out/` generated with expected assets.
4. Smoke-check key pages/modal behavior with base path awareness.

---

## 7) Known Risks and Inconsistencies

### A) External absolute paths in generator scripts
`scripts/generate_tasks_db.py` and `scripts/generate_compare_images.py` use hardcoded absolute paths outside this repo, for example:
- `/home/yjh/docs/plan/...`
- `/data/yjh/website_assets/...`
- output to `/data/yjh/agent-imaging-website/...`

Risk:
- Scripts are not portable across machines/workspaces.
- Output path does not match this repo path (`/data/yjh/csy/workplace/agent-imaging-website`).

Recommendation:
- Refactor scripts to accept CLI args / env vars and default to repo-relative paths.

### B) Data/assets drift potential
- `public/images/tasks` contains `Task_84_pycurious_reconstruction_result.png` image, but `tasks_db.json` currently reports 83 tasks and has no `task_84`.

Risk:
- Content may become stale or orphaned as benchmark set evolves.

Recommendation:
- Add a validation script to detect:
  - missing task records for images.
  - missing images for task records.
  - domain `task_count` mismatches.

### C) Nullability hazard for image filename
- JSON generator can emit `vis_result: None` in some conditions.
- UI currently assumes a valid filename string.

Recommendation:
- Add runtime guards and fallback placeholders.

### D) Link placeholders
- arXiv URL uses a placeholder-like ID (`2501.00000`) in hero/footer.

Recommendation:
- Move external links to a centralized config and validate before release.

---

## 8) High-Impact Improvement Backlog

### Priority 1: Robustness
- Add JSON schema validation for `tasks_db.json` in CI.
- Add runtime safety for missing images/metrics.
- Add a build-time asset consistency checker.

### Priority 2: Maintainability
- Centralize domain metadata/constants (colors/icons/labels) into one shared module.
- Split large components (`HeroSection`, `TaskModal`) into smaller subcomponents.
- Move repeated string literals (GitHub/arXiv links, labels) into config.

### Priority 3: Data pipeline portability
- Rewrite generator scripts to repo-relative IO with arguments:
  - `--source-report`
  - `--source-descriptions`
  - `--source-assets`
  - `--out-json`
- Add README usage examples for data generation.

### Priority 4: UX and performance
- Consider Next `<Image>` if optimization strategy changes from static unoptimized mode.
- Add loading/error placeholders for images.
- Consider lightweight virtualization/lazy detail loading if task count grows.

---

## 9) Change Playbooks for Future Agents

### Playbook A: Add or update task data
1. Edit/regen `public/data/tasks_db.json`.
2. Ensure `meta.total_tasks`, domain `task_count`, and `task_ids` are consistent.
3. Ensure matching image exists at:
   - `public/images/tasks/${folder}_${vis_result}`
4. Verify task opens in modal and image renders.

### Playbook B: Add new compare teaser example
1. Add image triplet/pair in `public/images/compare`.
2. Update `COMPARE_TASKS` in `HeroSection.tsx`.
3. Ensure `hasInput` flag matches available files (`input` vs `gt` on left side).
4. Test drag behavior on desktop + mobile touch.

### Playbook C: Change deployment path/repo name
1. Update `next.config.js`:
   - `basePath`
   - `assetPrefix`
   - `NEXT_PUBLIC_BASE_PATH`
2. Rebuild and verify all fetch/image URLs.
3. Validate GitHub Pages environment URL in workflow.

### Playbook D: Safely refactor UI
1. Keep `TaskData`, `DomainData`, `TasksDB` types aligned with JSON shape.
2. Preserve modal close behavior:
   - outside click
   - `Esc` key
   - body scroll restore
3. Preserve base-path-safe URL construction.

---

## 10) Coding Conventions Observed

- TypeScript strict mode is enabled; keep typed props/interfaces.
- Alias imports via `@/*` are configured and preferred.
- Styling mixes Tailwind utility classes with custom CSS utility classes.
- Existing code style favors:
  - functional components
  - local helper functions
  - lightweight inline constants

When contributing:
- Follow current naming conventions (`TaskData`, `DomainData`, etc.).
- Keep domain/task key formats stable (`A`...`H`, `task_XX`, `id` as zero-padded string).
- Avoid introducing server-only APIs into client components.

---

## 11) Suggested Immediate Next Tasks (Recommended Sequence)

1. Add `README.md` with setup/build/deploy/data-refresh instructions.
2. Refactor generator scripts for portable relative paths.
3. Add `scripts/validate_tasks_assets.py` and run in CI before deploy.
4. Add runtime image fallback in `DomainExplorer` and `TaskModal`.
5. Consolidate domain constants into one shared module.

---

## 12) Quick File Index

- App entry: `src/app/page.tsx`
- App shell: `src/app/layout.tsx`
- Global styles: `src/app/globals.css`
- Components: `src/components/*.tsx`
- Runtime data: `public/data/tasks_db.json`
- Code snippets: `src/data/codeSnippets.ts`
- Generators: `scripts/generate_tasks_db.py`, `scripts/generate_compare_images.py`
- Deploy: `.github/workflows/deploy.yml`
- Next config: `next.config.js`

This guide should be updated whenever deployment settings, data schema, or generation workflow changes.
