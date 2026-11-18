import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, '-X', 'utf8']

STEPS = [
    ('infra',        ['scripts/build_external_baselines_infra.py'],              'm', False),
    ('proof',        ['-m', 'src.experiments.exp_temporal_vs_static_proof'],     'm', False),
    ('conversion',   ['-m', 'src.experiments.exp_temporal_conversion_validation'],'m', False),
    ('overlay',      ['-m', 'src.experiments.exp_v2_trajectory_overlay'],        'm', False),
    ('innovation2',  ['-m', 'src.experiments.exp_innovation2_approximate'],      'm', False),
    ('supplementary',['-m', 'src.experiments.exp_innovation23_supplementary'],   'm', False),
    ('innovation3',  ['-m', 'src.experiments.exp_innovation3_necessity',
                      '--config', 'configs/innovation3.json'],                  'm', False),
    ('e4',           ['-m', 'src.experiments.rev2_E4_tail_settings'],            'm', False),
    ('e7',           ['-m', 'src.experiments.rev2_E7_group_size'],               'm', False),
    ('x4',           ['-m', 'src.experiments.rev3_X4_storage_recompute'],        'm', False),
    ('x1_bounds',    ['-m', 'src.experiments.rev3_X1_bound_check'],              'm', False),
    ('x1_refit',     ['-m', 'src.experiments.rev3_X1_bound_refit'],              'm', False),
    ('paper_core',   ['-m', 'src.experiments.exp_paper_core'],                   'm', True),
    ('revision',     ['-m', 'src.experiments.exp_results_revision'],             'm', False),
    ('csf',          ['-m', 'src.experiments.exp_csf_strengthening'],            'w', True),
    ('e2',           ['-m', 'src.experiments.rev2_E2_bic_refit'],                'm', False),
    ('x2',           ['-m', 'src.experiments.rev3_X2_score_variants'],           'j', True),
    ('x3b',          ['-m', 'src.experiments.rev3_X3b_lit_mechanisms'],          'j', True),
    ('ext_a',        ['-m', 'src.experiments.exp_ext_A_temporal'],               'm', False),
    ('ext_b',        ['-m', 'src.experiments.exp_ext_B_approx'],                 'm', False),
    ('ext_c',        ['-m', 'src.experiments.exp_ext_C_selection'],              'm', False),
    ('sir_small',    ['-m', 'src.experiments.rev4_sir_small'],                   'm', False),
    ('sir_empirical',['-m', 'src.experiments.rev4_sir_empirical'],               'm', False),
    ('audit_scale',  ['-m', 'src.experiments.rev4_audit_scale'],                 'm', False),
    ('dataset_stats',['-m', 'src.experiments.rev4_dataset_stats'],               'm', False),
    ('collect',      ['scripts/collect_figure_data.py'],                         'm', False),
    ('x1',           ['-m', 'src.experiments.rev3_X1_tail_grid'],                'm', False),
    ('figures_core', ['scripts/build_figures_core.py'],                          'm', False),
    ('figures_csf',  ['scripts/build_figures_csf.py'],                           'm', False),
    ('figures_redesign', ['scripts/build_figures_redesign.py'],                  'm', False),
    ('ext_figures',  ['scripts/build_ext_figures.py'],                           'm', False),
    ('sir_figure',   ['scripts/build_rev4_sir_bounds_figure.py'],                'm', False),
    ('tables',       ['scripts/build_manuscript_tables.py'],                     'm', False),
]

DEPS = {
    'e4': ['innovation2'],
    'x4': ['e7'],
    'x1_refit': ['supplementary'],
    'collect': ['proof', 'paper_core', 'revision', 'supplementary', 'innovation3'],
    'x1': ['e4', 'collect'],
    'figures_core': ['collect'],
    'figures_csf': ['csf', 'collect'],
    'figures_redesign': ['csf', 'collect'],
    'ext_a': ['infra'], 'ext_b': ['infra','ext_a'], 'ext_c': ['infra'],
    'ext_figures': ['ext_a', 'ext_b', 'ext_c'],
    'sir_empirical': ['infra', 'ext_a'],
    'audit_scale': ['infra'],
    'sir_figure': ['sir_small'],
    'e2': ['infra','ext_c'],
    'x2': ['revision', 'paper_core'],
    'x3b': ['revision', 'paper_core'],
    'tables': ['collect', 'csf', 'csf_bench', 'x1', 'x2', 'x4', 'dataset_stats', 'audit_scale', 'ext_a', 'ext_b', 'ext_c'],
}

ORDER = [s[0] for s in STEPS]
BY_NAME = {s[0]: s for s in STEPS}

def resolve(requested):
    seen = []
    def visit(n):
        if n in seen:
            return
        for d in DEPS.get(n, []):
            visit(d)
        seen.append(n)
    for n in requested:
        visit(n)
    return [n for n in ORDER if n in seen]

def run_step(name, jobs):
    _, argv, mode, parallel = BY_NAME[name]
    cmd = PY + argv.copy()
    if jobs and parallel:
        flag = {'j': '--jobs', 'w': '--workers'}.get(mode)
        if flag:
            cmd += [flag, str(jobs)]
    t0 = time.time()
    print(f'=== {name}: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f'=== {name} finished in {dt:.1f}s (exit {r.returncode})', flush=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser(description='Reproduce all experiments, figures and tables of the manuscript.')
    ap.add_argument('steps', nargs='*', help='step names to run (default: all)')
    ap.add_argument('--list', action='store_true', help='list steps and exit')
    ap.add_argument('--jobs', type=int, default=0, help='parallel workers where supported')
    args = ap.parse_args()

    if args.list:
        for name, argv, _, par in STEPS:
            deps = DEPS.get(name, [])
            d = f'  deps: {", ".join(deps)}' if deps else ''
            j = '  [--jobs]' if par else ''
            print(f'{name:<18} {" ".join(argv)}{j}{d}')
        return

    unknown = [s for s in args.steps if s not in BY_NAME]
    if unknown:
        sys.exit(f'unknown steps: {unknown}; run with --list')

    plan = resolve(args.steps) if args.steps else ORDER
    failed = []
    for name in plan:
        if run_step(name, args.jobs) != 0:
            failed.append(name)
            print(f'=== {name} FAILED', flush=True)
    if failed:
        sys.exit(f'failed steps: {failed}')
    print('all steps complete')

if __name__ == '__main__':
    main()
