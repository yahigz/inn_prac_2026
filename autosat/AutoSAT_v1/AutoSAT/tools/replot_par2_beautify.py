#!/usr/bin/env python3
import os, sys, json, glob, re, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS_DIR = os.path.join(ROOT, 'results', 'runs')


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_par2(obj):
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        # direct keys
        for k in obj:
            if re.search(r'par[\-_ ]?2', k, re.I):
                try:
                    return float(obj[k])
                except:
                    pass
        # nested
        for v in obj.values():
            res = extract_par2(v)
            if res is not None:
                return res
    return None


def process_run(run_path, forced_baselines):
    basename = os.path.basename(run_path)
    snaps_dir = os.path.join(run_path, 'snapshots')
    if not os.path.isdir(snaps_dir):
        return None
    files = glob.glob(os.path.join(snaps_dir, 'iter_*_best.json'))
    if not files:
        return None
    def key(p):
        m = re.search(r'iter_(\d+)_best', os.path.basename(p))
        return int(m.group(1)) if m else 999999
    files.sort(key=key)
    iters = []
    par2s = []
    for p in files:
        m = re.search(r'iter_(\d+)_best', os.path.basename(p))
        if not m:
            continue
        it = int(m.group(1))
        data = read_json(p)
        val = extract_par2(data)
        if val is None:
            # fallback to top-level key
            if isinstance(data, dict) and 'PAR-2' in data:
                try:
                    val = float(data['PAR-2'])
                except:
                    val = None
        if val is None:
            continue
        iters.append(it+1)  # shift snapshots to start after baseline(0)
        par2s.append(val)
    if not iters:
        return None
    # find baseline
    baseline = None
    final_json = os.path.join(run_path, 'final_result.json')
    if os.path.isfile(final_json):
        try:
            fr = read_json(final_json)
            if isinstance(fr, dict) and '0' in fr:
                baseline = extract_par2(fr['0'])
            else:
                baseline = extract_par2(fr)
        except:
            baseline = None
    # fallback: check iter_0_best
    if baseline is None:
        iter0 = os.path.join(snaps_dir, 'iter_0_best.json')
        if os.path.isfile(iter0):
            try:
                d0 = read_json(iter0)
                baseline = extract_par2(d0)
            except:
                baseline = None
    # forced baseline mapping
    if basename in forced_baselines:
        try:
            baseline = float(forced_baselines[basename])
        except:
            pass
    if baseline is None:
        # do not draw baseline
        pass
    # prepend baseline at iteration 0 if available
    plot_iters = list(iters)
    plot_par2s = list(par2s)
    if baseline is not None:
        plot_iters = [0] + plot_iters
        plot_par2s = [baseline] + plot_par2s
    # plotting
    out_dir = os.path.join(run_path, 'analysis_par2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'par2_vs_iter.png')
    plt.figure(figsize=(6,4))
    # semilogy
    plt.plot(plot_iters, plot_par2s, linestyle='-', marker='o', markersize=4, linewidth=1.5)
    plt.yscale('log')
    if baseline is not None:
        plt.axhline(baseline, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('PAR-2')
    plt.title(basename)
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='Path to single run dir (basename or full path)')
    parser.add_argument('--force-baseline', action='append', help='Force baseline mapping in form basename=VALUE or fullpath=VALUE')
    args = parser.parse_args()
    forced = {}
    if args.force_baseline:
        for item in args.force_baseline:
            if '=' in item:
                k,v = item.split('=',1)
                forced[os.path.basename(k)] = v
    runs = []
    if args.run:
        # allow basename lookup
        p = args.run
        if not os.path.isabs(p):
            candidate = os.path.join(RUNS_DIR, p)
            if os.path.isdir(candidate):
                runs = [candidate]
            else:
                # try exact match by basename
                for d in glob.glob(os.path.join(RUNS_DIR, '*')):
                    if os.path.basename(d) == p:
                        runs = [d]
                        break
        else:
            runs = [p] if os.path.isdir(p) else []
    else:
        runs = sorted(glob.glob(os.path.join(RUNS_DIR, '*')))
    out = []
    for r in runs:
        path = process_run(r, forced)
        if path:
            out.append(path)
    if not out:
        print('No plots generated.')
        sys.exit(1)
    for p in out:
        print('Wrote', p)

if __name__ == '__main__':
    main()
