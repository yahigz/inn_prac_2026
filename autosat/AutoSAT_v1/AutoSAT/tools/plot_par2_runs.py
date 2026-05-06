import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'runs'
OUTDIR = ROOT / 'analysis_par2'
OUTDIR.mkdir(exist_ok=True)

runs = []
for entry in sorted(ROOT.iterdir()):
    if not entry.is_dir():
        continue
    run_dir = entry
    final_path = run_dir / 'final_result.json'
    snapshots_dir = run_dir / 'snapshots'
    if not final_path.exists():
        continue
    try:
        final = json.loads(final_path.read_text())
    except Exception:
        continue
    # require baseline '0' present in final
    if '0' not in final:
        continue
    # collect snapshot files
    if not snapshots_dir.exists():
        continue
    snapshot_files = sorted(snapshots_dir.glob('iter_*_best.json'))
    # count iterations
    if len(snapshot_files) <= 5:
        continue
    # load iteration -> PAR-2
    iters = []
    par2s = []
    for sf in snapshot_files:
        try:
            s = json.loads(sf.read_text())
            it = int(s.get('iter', sf.stem.split('_')[1]))
            p = s.get('PAR-2', None)
            if p is None:
                p = s.get('PAR-2', None)
            if p is None:
                continue
            iters.append(it)
            par2s.append(p)
        except Exception:
            continue
    if not iters:
        continue
    # sort by iter
    pairs = sorted(zip(iters, par2s), key=lambda x: x[0])
    iters, par2s = zip(*pairs)
    # include baseline from final['0'] as iteration 0
    try:
        baseline_par2 = final['0'].get('PAR-2', None)
        if baseline_par2 is not None:
            iters = (0,) + tuple(iters)
            par2s = (baseline_par2,) + tuple(par2s)
    except Exception:
        pass
    runs.append((run_dir.name, iters, par2s))

# plot each
for name, iters, par2s in runs:
    plt.figure(figsize=(8,4))
    plt.plot(iters, par2s, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('PAR-2')
    plt.title(f'Run {name} PAR-2 vs Iteration')
    plt.grid(True)
    out = OUTDIR / f'{name}_par2.png'
    plt.savefig(out)
    plt.close()
    print(f'Wrote {out}')

print('Done. Generated', len(runs), 'plots into', OUTDIR)
