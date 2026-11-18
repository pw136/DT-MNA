from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'figure_data'
OUT = ROOT / 'results' / 'tables' / 'manuscript'

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(name, df):
    df.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    print(name, len(df))

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / 'results' / 'csf_strengthening_v2' / 'runtime.csv', OUT / 'tab_params_cost_raw.csv')
    rt = pd.read_csv(ROOT / 'results' / 'csf_strengthening_v2' / 'runtime.csv')
    keep = rt[rt.context.isin(['hetero_synthetic_00_nu2', 'hetero_High-School_nu2'])]
    save('tab_params_cost', keep[['context', 'nodes', 'model', 'parameters', 'input_bytes',
                                  'parameter_bytes', 'preprocessing_seconds', 'fit_seconds',
                                  'inference_seconds', 'state_vectors']])

    dyn = pd.read_csv(DATA / 'dynamic_processes.csv')
    s0 = pd.read_csv(ROOT / 'results' / 'rev3' / 'X2' / 'score_variant_summary.csv')
    s0 = s0[s0.variant == 'S0'].set_index('method')
    g = dyn.groupby('method').agg(
        networks=('context', 'size'), mean_score=('mean_complexity', 'mean'),
        switches=('switches', 'mean'),
        density_mae=('paired_density_mae', 'mean'), curve_mae=('mean_curve_mae', 'mean'),
        final_mae=('final_density_mae', 'mean'), path_agreement=('paired_path_match_rate', 'mean'),
        over_budget_rate=('test_stage_exceed_rate', 'mean'),
        simulation_seconds=('simulation_seconds', 'median')).reset_index()
    ref = g.loc[g.method == 'Fixed-H', 'simulation_seconds'].iloc[0]
    g['simulation_rel_fixed_H'] = g.simulation_seconds / ref
    g['h_fraction'] = g.method.map(s0['frac_H'])
    g['fitted_scalars'] = g.method.map(s0['fitted_scalars'])
    save('tab_dynamic_process', g)

    rep = read_json(ROOT / 'results' / 'external_baselines' / 'infra' / 'dataset_report.json')
    rows = rep['datasets'] if isinstance(rep, dict) and 'datasets' in rep else rep
    save('tab_ext_datasets', pd.DataFrame(rows))

    for letter, name in [('A', 'tab_ext_temporal'), ('B', 'tab_ext_approx'), ('C', 'tab_ext_selection')]:
        d = ROOT / 'results' / 'external_baselines' / letter / 'summary.csv'
        if d.exists():
            save(name, pd.read_csv(d))

    tail = pd.read_csv(ROOT / 'results' / 'rev3' / 'X1' / 'tail_grid_summary.csv')
    save('tab_tail_sensitivity', tail[tail.conflict == 0.16])
    save('tab_tail_sensitivity_full', tail)

    shutil.copy2(ROOT / 'results' / 'rev3' / 'X2' / 'score_variant_summary.csv', OUT / 'tab_score_sensitivity.csv')

    x4 = pd.read_csv(ROOT / 'results' / 'rev3' / 'X4' / 'summary.csv')
    save('tab_size_cost', x4[['m', 'storage_H', 'storage_G', 'storage_MG', 'storage_C2',
                              'storage_ratio_G_H', 'storage_ratio_MG_H', 'storage_ratio_C2_H',
                              'sim_s_H', 'sim_s_G', 'sim_s_MG', 'sim_s_C2',
                              'sim_ratio_G_H', 'sim_ratio_MG_H', 'sim_ratio_C2_H']])

    ds = read_json(ROOT / 'results' / 'rev4' / 'dataset_stats.json')
    rows = ds['datasets'] if isinstance(ds, dict) and 'datasets' in ds else ds
    save('tab_dataset_stats', pd.DataFrame(rows))

    aud_rows = []
    for p in sorted((ROOT / 'results' / 'rev4' / 'audit_scale').glob('setting_*.json')):
        d = read_json(p)
        for r in d['rows']:
            aud_rows.append({'dataset': d['dataset'], 'mechanism': d['mechanism'], **r})
    aud = pd.DataFrame(aud_rows)
    aud['lower'] = aud.selected != 'H'
    aud_tab = aud.groupby(['n_val', 'n_audit']).agg(
        violation=('violation', 'mean'), lower=('lower', 'mean'),
        n=('violation', 'size'))
    aud_tab.to_csv(ROOT / 'results' / 'rev4' / 'audit_scale_table.csv')
    rng = np.random.RandomState(0)
    aud_res = {}
    for (nv, na), grp in aud.groupby(['n_val', 'n_audit']):
        per = grp.groupby(['dataset', 'mechanism']).violation.agg(['sum', 'size'])
        boots = []
        for _ in range(4000):
            idx = rng.randint(len(per), size=len(per))
            s = per.iloc[idx]
            boots.append(s['sum'].sum() / s['size'].sum())
        aud_res['%d_%d' % (nv, na)] = [
            float(per['sum'].sum() / per['size'].sum()),
            float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    (ROOT / 'results' / 'rev4' / 'audit_scale_summary.json').write_text(
        json.dumps(aud_res, indent=1), encoding='utf-8')
    aud_tab.to_csv(OUT / 'tab_audit_scale.csv')

    sir_dir = ROOT / 'results' / 'rev4' / 'sir_empirical'
    if sir_dir.exists():
        frame = []
        for p in sorted(sir_dir.glob('setting_*.json')):
            frame.append(read_json(p))
        save('tab_sir_selection', pd.json_normalize(frame))

    sdir = ROOT / 'results' / 'rev4' / 'sir_small'
    if (sdir / 'bounds.json').exists():
        save('tab_sir_bounds', pd.DataFrame(read_json(sdir / 'bounds.json')))

if __name__ == '__main__':
    main()
