#!/usr/bin/env python3
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def section(md: str, name: str) -> str:
    pattern = rf"^##\s+{re.escape(name)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    m = re.search(pattern, md, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_description(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"^\s*```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    overview = section(text, "overview")
    objective = section(text, "objective_function")
    objective_expr = ""
    m = re.search(r"\*\*full_expression\*\*:\s*(.+)", objective)
    if m:
        objective_expr = m.group(1).strip()
    overview = re.sub(r"\n{3,}", "\n\n", overview).strip()
    objective_expr = re.sub(r"\s+", " ", objective_expr).strip()
    if overview and objective_expr:
        return f"{overview}\n\nObjective: {objective_expr}"
    if overview:
        return overview
    return ""


def clean_description(text: str, max_len: int) -> str:
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"\n---\n?", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    chunk = cleaned[:max_len]
    last_break = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
    if last_break > int(max_len * 0.7):
        return chunk[:last_break].strip()
    return chunk.strip()


def task_candidates(task: dict) -> list[str]:
    out: list[str] = []
    for key in ("name", "title"):
        v = task.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    folder = (
        task.get("images", {}).get("folder")
        if isinstance(task.get("images"), dict)
        else ""
    )
    if isinstance(folder, str) and folder.strip():
        out.append(folder.strip())
        m = re.match(r"Task_\d+_(.+)$", folder.strip(), flags=re.IGNORECASE)
        if m:
            suffix = m.group(1).strip()
            out.append(suffix)
            out.append(suffix.replace("imag", "image"))
            out.append(f"{suffix}-main")
            out.append(f"{suffix}-master")
    unique: list[str] = []
    seen = set()
    for s in out:
        n = normalize(s)
        if n and n not in seen:
            seen.add(n)
            unique.append(s)
    return unique


def best_match(task: dict, desc_files: list[Path]) -> Optional[Path]:
    norm_to_path = {normalize(f.stem.replace("_description", "")): f for f in desc_files}
    candidates = task_candidates(task)
    for c in candidates:
        n = normalize(c)
        if n in norm_to_path:
            return norm_to_path[n]
    best_file = None
    best_score = 0.0
    desc_norms = list(norm_to_path.keys())
    for c in candidates:
        cn = normalize(c)
        for dn in desc_norms:
            score = SequenceMatcher(None, cn, dn).ratio()
            if score > best_score:
                best_score = score
                best_file = norm_to_path[dn]
    if best_score >= 0.9:
        return best_file
    return None


def should_update(task: dict) -> bool:
    desc = str(task.get("description", "")).strip()
    if not desc or desc.endswith("inverse problem."):
        return True
    if 495 <= len(desc) <= 505:
        return True
    if re.search(r"(fitti|optimiz|deco|re|solv|inversio|evaluat)\s*$", desc, flags=re.IGNORECASE):
        return True
    if re.search(r"[A-Za-z0-9]$", desc) and not re.search(r"[.!?…]$", desc):
        return len(desc) >= 450
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-db",
        default="/data/yjh/csy/workplace/agent-imaging-website/public/data/tasks_db.json",
    )
    parser.add_argument("--desc-dir", default="/data/yjh/task_descriptions")
    parser.add_argument("--max-len", type=int, default=3000)
    parser.add_argument("--force-all", action="store_true")
    args = parser.parse_args()

    tasks_db_path = Path(args.tasks_db)
    desc_dir = Path(args.desc_dir)
    payload = json.loads(tasks_db_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    desc_files = sorted(desc_dir.glob("*_description.md"))

    updated = 0
    unmatched = []
    updated_keys = []
    for task_key, task in tasks.items():
        if not isinstance(task, dict):
            continue
        if not args.force_all and not should_update(task):
            continue
        match = best_match(task, desc_files)
        if not match:
            unmatched.append(task_key)
            continue
        desc = parse_description(match)
        if not desc:
            unmatched.append(task_key)
            continue
        task["description"] = clean_description(desc, args.max_len)
        if not task.get("name"):
            stem_name = match.stem.replace("_description", "")
            task["name"] = stem_name
        updated += 1
        updated_keys.append(task_key)

    tasks_db_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"updated={updated}")
    print(f"unmatched={len(unmatched)}")
    if updated_keys:
        print("updated_keys=" + ",".join(updated_keys))
    if unmatched:
        print("unmatched_keys=" + ",".join(unmatched))


if __name__ == "__main__":
    main()
