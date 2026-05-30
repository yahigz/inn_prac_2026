#!/usr/bin/env python3
"""Sequential greedy eval with write-back.

Strategy
--------
1. Load final_result.json from the specified run directory.
2. Collect all candidates (global_id > 0) that beat the **training** baseline
   PAR-2, sorted in chronological order (ascending global_id).
3. Run the baseline solver on the eval dataset once per task to get the
   initial "current best eval PAR-2" for each task.
4. For each candidate in order:
     a. Build a filled solver .cpp from the task template + candidate code.
     b. Run evaluate() on the eval dataset.
     c. If eval PAR-2 < current best eval PAR-2 for that task → ACCEPT:
          - Write the candidate code into ./examples/EasySAT/<task>/EasySAT.cpp
            (replacing {{ replace_code }}) so future training starts from it.
          - Save a .bak backup on first write per task.
          - Update current best eval PAR-2 for that task.
     d. Otherwise → SKIP.
5. Print a summary table and save it as JSON.

Usage
-----
    cd autosat/AutoSAT_v1/AutoSAT
    python3 tools/eval_sequential_writeback.py \\
        --config config.eval_eliza_run_20260518.yaml
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autosat.evaluation.evaluate import evaluate          # noqa: E402
from autosat.utils import copy_folder, revise_file        # noqa: E402
from main import _infer_task_from_code, _validate_filled_solver  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    global_id: int
    train_par2: float
    code: str
    task_name: Optional[str]
    extra_params: dict = field(default_factory=dict)


@dataclass
class EvalRecord:
    global_id: int
    task_name: str
    train_par2: float
    eval_par2: Optional[float]
    prev_best_eval_par2: Optional[float]
    decision: str   # "accepted" | "skipped" | "failed" | "no_baseline"
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_") or "run"


def _latest_result_file(results_dir: Path, method_name: str) -> Optional[Path]:
    matches = sorted(results_dir.glob(f"results_{method_name}_*.txt"),
                     key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _read_par2(result_file: Path) -> float:
    """Parse PAR-2 from the last dict-line of a collect_results_eval output file."""
    last_line = ""
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        raise ValueError(f"Empty eval result file: {result_file}")
    payload = ast.literal_eval(last_line)
    if "PAR-2" not in payload:
        raise KeyError(f"PAR-2 key missing in {result_file}: {payload}")
    return float(payload["PAR-2"])


def _build_solver(
    task_name: str,
    candidate_code: str,
    work_dir: Path,
    label: str,
    eval_timeout: int,
    eval_data_dir: str,
    extra_params: dict,
) -> Path:
    """Fill the task template with candidate_code and return path to the .cpp."""
    template_dir = ROOT / "examples" / "EasySAT" / task_name
    if not template_dir.exists():
        raise FileNotFoundError(f"Task template dir not found: {template_dir}")

    solver_dir = work_dir / label / task_name
    if solver_dir.exists():
        shutil.rmtree(solver_dir)

    copy_folder(src_folder=str(template_dir), num=1, mode="eval",
                target_folder=str(solver_dir))

    # Copy header files
    headers_dir = ROOT / "examples" / "EasySAT" / "original_EasySAT"
    for hdr in ["EasySAT.hpp", "heap.hpp"]:
        src = headers_dir / hdr
        if src.exists():
            shutil.copy(str(src), str(solver_dir / hdr))

    solver_path = solver_dir / "EasySAT_modified.cpp"

    prev_cwd = os.getcwd()
    try:
        os.chdir(solver_dir)
        revise_file(
            file_name="EasySAT.cpp",
            save_dir="EasySAT_modified.cpp",
            timeout=eval_timeout,
            data_dir=f'"{eval_data_dir}"',
            replace_code=candidate_code,
            **extra_params,
        )
    finally:
        os.chdir(prev_cwd)

    ok, err = _validate_filled_solver(str(solver_path), label)
    if not ok:
        raise RuntimeError(f"Compile check failed for {label}: {err}")
    return solver_path


def _run_eval(
    solver_path: Path,
    method_name: str,
    results_dir: Path,
    eval_args,
) -> float:
    """Run evaluate() and return PAR-2."""
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_args.results_save_path = str(results_dir)
    evaluate(eval_args, SAT_solver_file_path=str(solver_path),
             method_name=method_name)
    result_file = _latest_result_file(results_dir, method_name)
    if result_file is None:
        raise FileNotFoundError(
            f"No result file found for {method_name} in {results_dir}")
    return _read_par2(result_file)


def _writeback_template(task_name: str, code: str, project: str) -> None:
    """Replace {{ replace_code }} in the task template with code.
    Saves a .bak backup on first write.
    """
    template_path = ROOT / "examples" / project.strip("/") / task_name / "EasySAT.cpp"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    placeholder_re = re.compile(r"\{\{\s*replace_code\s*\}\}")
    if not placeholder_re.search(content):
        raise ValueError(
            f"No {{{{ replace_code }}}} placeholder in {template_path}")

    # Backup on first write
    backup_path = Path(str(template_path) + ".bak")
    if not backup_path.exists():
        shutil.copy(str(template_path), str(backup_path))
        print(f"  [WriteBack] Backup saved: {backup_path}", flush=True)

    # Use a lambda to avoid re.sub interpreting backslashes in the replacement string
    new_content = placeholder_re.sub(lambda _: code, content)
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  [WriteBack] Updated template: {template_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential greedy eval with write-back for a completed AutoSAT run.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (e.g. config.eval_eliza_run_20260518.yaml)")
    parser.add_argument("--work-dir", default="./temp/eval_seq_work/",
                        help="Temporary directory for generated solver sources.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run eval but do NOT write back to templates.")
    args_ns = parser.parse_args()

    config = _load_yaml(Path(args_ns.config))

    # ── Config values ────────────────────────────────────────────────────────
    run_id = str(config.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_id must be set in config")

    run_dir = ROOT / "results" / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    project          = str(config.get("project", "EasySAT/"))
    eval_data_dir    = str(config.get("eval_data_dir", "../../datasets/cryptography-ascon/eval"))
    eval_parallel    = int(config.get("eval_parallel_size", 7))
    eval_timeout     = int(config.get("eval_timeout", 5000))
    rand_seed        = int(config.get("rand_seed", 42))
    keep_inter       = bool(config.get("keep_intermediate_results", False))
    baseline_solver  = str(config.get("SAT_solver_file_path",
                                      "./examples/EasySAT/original_EasySAT/EasySAT.cpp"))
    llm_model        = str(config.get("llm_model", "eval")).replace("/", "_")
    results_save     = Path(config.get("results_save_path", "./temp/eval_results_sequential/"))
    writeback        = bool(config.get("writeback_to_template", True)) and not args_ns.dry_run

    work_dir = Path(args_ns.work_dir)

    # ── Load final_result.json ───────────────────────────────────────────────
    final_result_path = run_dir / "final_result.json"
    if not final_result_path.exists():
        raise FileNotFoundError(f"final_result.json not found in {run_dir}")
    final = _load_json(final_result_path)

    # Training baseline PAR-2 (key "0")
    if "0" not in final:
        raise ValueError("Key '0' (baseline) missing in final_result.json")
    train_baseline_par2 = float(final["0"]["PAR-2"])
    print(f"[Config] Run: {run_id}", flush=True)
    print(f"[Config] Training baseline PAR-2: {train_baseline_par2}", flush=True)
    print(f"[Config] Eval data dir: {eval_data_dir}", flush=True)
    print(f"[Config] Write-back: {'enabled' if writeback else 'DISABLED (dry-run)'}", flush=True)

    # ── Collect candidates that beat training baseline, sorted by global_id ──
    candidates: list[CandidateRecord] = []
    for key, item in final.items():
        if key == "0":
            continue
        par2 = float(item.get("PAR-2", float("inf")))
        if par2 >= train_baseline_par2:
            continue
        code = str(item.get("prompt", "")).strip()
        if not code:
            continue
        task_name = _infer_task_from_code(code)
        candidates.append(CandidateRecord(
            global_id=int(key),
            train_par2=par2,
            code=code,
            task_name=task_name,
        ))

    candidates.sort(key=lambda c: c.global_id)
    print(f"\n[Candidates] {len(candidates)} candidates beat training baseline "
          f"(PAR-2 < {train_baseline_par2}):", flush=True)
    for c in candidates:
        print(f"  global_id={c.global_id:3d}  task={c.task_name or '?':25s}  "
              f"train_PAR-2={c.train_par2:.0f}", flush=True)

    if not candidates:
        print("\nNo candidates to evaluate. Exiting.", flush=True)
        return 0

    # ── Build eval_args namespace ────────────────────────────────────────────
    import argparse as _ap
    eval_args = _ap.Namespace(
        eval_data_dir=eval_data_dir,
        eval_parallel_size=eval_parallel,
        eval_timeout=eval_timeout,
        keep_intermediate_results=keep_inter,
        rand_seed=rand_seed,
        SAT_solver_file_path=baseline_solver,
        results_save_path=str(results_save),
        project=project,
        task="",
        llm_model=llm_model,
        original=True,
        eval_baseline=False,
    )

    # ── Run baseline eval per task (once) ────────────────────────────────────
    tasks_needed = {c.task_name for c in candidates if c.task_name}
    baseline_eval_par2: dict[str, float] = {}
    print("\n[Baseline] Running baseline eval per task...", flush=True)
    for task_name in sorted(tasks_needed):
        method_name = f"baseline_{task_name}_{llm_model}"
        task_results_dir = results_save / "baseline" / task_name
        eval_args.task = task_name

        # Check if already evaluated
        existing = _latest_result_file(task_results_dir, method_name)
        if existing:
            par2 = _read_par2(existing)
            baseline_eval_par2[task_name] = par2
            print(f"  {task_name}: baseline eval PAR-2 = {par2:.2f} (cached)", flush=True)
            continue

        try:
            par2 = _run_eval(
                solver_path=Path(baseline_solver),
                method_name=method_name,
                results_dir=task_results_dir,
                eval_args=eval_args,
            )
            baseline_eval_par2[task_name] = par2
            print(f"  {task_name}: baseline eval PAR-2 = {par2:.2f}", flush=True)
        except Exception as exc:
            print(f"  {task_name}: baseline eval FAILED: {exc}", flush=True)

    # ── Sequential greedy eval ───────────────────────────────────────────────
    # current_best_eval_par2 starts at baseline eval PAR-2 per task
    current_best: dict[str, float] = dict(baseline_eval_par2)
    records: list[EvalRecord] = []

    print(f"\n[Eval] Processing {len(candidates)} candidates sequentially...\n",
          flush=True)

    for c in candidates:
        task_name = c.task_name
        if not task_name:
            print(f"  [SKIP] global_id={c.global_id}: cannot infer task from code",
                  flush=True)
            records.append(EvalRecord(
                global_id=c.global_id, task_name="?",
                train_par2=c.train_par2, eval_par2=None,
                prev_best_eval_par2=None, decision="failed",
                note="cannot infer task"))
            continue

        prev_best = current_best.get(task_name)
        if prev_best is None:
            print(f"  [SKIP] global_id={c.global_id} task={task_name}: "
                  f"no baseline eval PAR-2 available", flush=True)
            records.append(EvalRecord(
                global_id=c.global_id, task_name=task_name,
                train_par2=c.train_par2, eval_par2=None,
                prev_best_eval_par2=None, decision="no_baseline"))
            continue

        label = f"gid{c.global_id}_{task_name}"
        method_name = f"candidate_{label}_{llm_model}"
        task_results_dir = results_save / label

        # Check if already evaluated
        existing = _latest_result_file(task_results_dir, method_name)
        if existing:
            try:
                eval_par2 = _read_par2(existing)
                print(f"  global_id={c.global_id:3d} task={task_name}: "
                      f"eval PAR-2={eval_par2:.2f} (cached)", flush=True)
            except Exception as exc:
                print(f"  global_id={c.global_id:3d} task={task_name}: "
                      f"cached result read failed: {exc}", flush=True)
                existing = None

        if not existing:
            # Build solver
            try:
                solver_path = _build_solver(
                    task_name=task_name,
                    candidate_code=c.code,
                    work_dir=work_dir,
                    label=label,
                    eval_timeout=eval_timeout,
                    eval_data_dir=eval_data_dir,
                    extra_params=c.extra_params,
                )
            except Exception as exc:
                print(f"  global_id={c.global_id:3d} task={task_name}: "
                      f"BUILD FAILED: {exc}", flush=True)
                records.append(EvalRecord(
                    global_id=c.global_id, task_name=task_name,
                    train_par2=c.train_par2, eval_par2=None,
                    prev_best_eval_par2=prev_best, decision="failed",
                    note=f"build: {exc}"))
                continue

            # Run eval
            eval_args.task = task_name
            try:
                eval_par2 = _run_eval(
                    solver_path=solver_path,
                    method_name=method_name,
                    results_dir=task_results_dir,
                    eval_args=eval_args,
                )
            except Exception as exc:
                print(f"  global_id={c.global_id:3d} task={task_name}: "
                      f"EVAL FAILED: {exc}", flush=True)
                records.append(EvalRecord(
                    global_id=c.global_id, task_name=task_name,
                    train_par2=c.train_par2, eval_par2=None,
                    prev_best_eval_par2=prev_best, decision="failed",
                    note=f"eval: {exc}"))
                continue

        # Decision
        if eval_par2 < prev_best:
            decision = "accepted"
            note = f"improved {prev_best:.2f} → {eval_par2:.2f}"
            print(f"  global_id={c.global_id:3d} task={task_name}: "
                  f"eval PAR-2={eval_par2:.2f} < prev_best={prev_best:.2f} → ACCEPTED",
                  flush=True)
            current_best[task_name] = eval_par2

            if writeback:
                try:
                    _writeback_template(task_name, c.code, project)
                except Exception as exc:
                    print(f"  [WriteBack] WARNING: {exc}", flush=True)
                    note += f" | writeback failed: {exc}"
        else:
            decision = "skipped"
            note = f"eval PAR-2={eval_par2:.2f} >= prev_best={prev_best:.2f}"
            print(f"  global_id={c.global_id:3d} task={task_name}: "
                  f"eval PAR-2={eval_par2:.2f} >= prev_best={prev_best:.2f} → skipped",
                  flush=True)

        records.append(EvalRecord(
            global_id=c.global_id, task_name=task_name,
            train_par2=c.train_par2, eval_par2=eval_par2,
            prev_best_eval_par2=prev_best, decision=decision, note=note))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"{'gid':>5}  {'task':<25}  {'train_PAR2':>10}  "
          f"{'eval_PAR2':>10}  {'prev_best':>10}  {'decision'}", flush=True)
    print("-" * 72, flush=True)
    for r in records:
        ep = f"{r.eval_par2:.2f}" if r.eval_par2 is not None else "—"
        pb = f"{r.prev_best_eval_par2:.2f}" if r.prev_best_eval_par2 is not None else "—"
        print(f"{r.global_id:>5}  {r.task_name:<25}  {r.train_par2:>10.2f}  "
              f"{ep:>10}  {pb:>10}  {r.decision}", flush=True)

    print("\nFinal best eval PAR-2 per task:", flush=True)
    for task_name in sorted(current_best):
        baseline = baseline_eval_par2.get(task_name, float("inf"))
        best = current_best[task_name]
        delta = best - baseline
        print(f"  {task_name:<25}  baseline={baseline:.2f}  "
              f"best={best:.2f}  delta={delta:+.2f}", flush=True)

    # Save summary JSON
    summary_path = results_save / "eval_sequential_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "train_baseline_par2": train_baseline_par2,
            "baseline_eval_par2": baseline_eval_par2,
            "final_best_eval_par2": current_best,
            "records": [r.__dict__ for r in records],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
