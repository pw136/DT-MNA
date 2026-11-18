from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'results' / 'paper_core_extension'
REV = ROOT / 'results' / 'results_revision_supplement'
DATA = ROOT / 'results' / 'figure_data'

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def export(name, df):
    df.to_csv(DATA / (name + '.csv'), index=False, encoding='utf-8-sig')
    return df

def boot(vals, seed=20260911, B=4000):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    means = vals[rng.randint(len(vals), size=(B, len(vals)))].mean(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]

def records(d):
    return d.replace({np.nan: None}).to_dict('records')

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    stats = {}

    proof = read_json(ROOT / 'results' / 'summary_tables' / 'temporal_vs_static_proof.json')['controlled']
    curves = pd.DataFrame({
        'time': np.arange(len(proof['original_density'])),
        'original_density': proof['original_density'],
        'reverse_density': proof['reverse_density'],
        'shuffle_density_mean': proof['shuffle_density_mean'],
        'mean_kernel_static_density': proof['mean_kernel_static_density'],
        'static_union_density': proof['static_union_density'],
        'shuffle_p05': proof['shuffle_density_p05'],
        'shuffle_p95': proof['shuffle_density_p95'],
    })
    export('order_curves', curves)
    export('order_permutations', pd.DataFrame({
        'order_index': np.arange(len(proof['shuffle_endpoint_tv'])),
        'endpoint_tv': proof['shuffle_endpoint_tv'],
    }))
    stats['temporal_order'] = {
        'shuffle_count': proof['shuffle_count'],
        'density': proof['endpoint_density'],
        'endpoint_tv': proof['endpoint_tv_from_original'],
    }

    c = read_json(NEW / 'certificate_all.json')
    candidates = export('certificate_candidates', pd.DataFrame([
        {**{k: x[k] for k in ('case', 'nodes', 'horizon', 'mechanism', 'replicate', 'recovery')}, **r}
        for x in c for r in x['rows']]))
    sels = export('certificate_selections', pd.DataFrame([r for x in c for r in x['selections']]))
    certsum = sels.groupby('rule').agg(decisions=('unsafe', 'size'), unsafe=('unsafe', 'sum'),
                                       simplified=('simplified', 'sum'), cost=('cost', 'mean'),
                                       mean_cost_gap=('cost_gap', 'mean')).reset_index()
    export('certificate_summary', certsum)
    stats['certificate_summary'] = records(certsum)
    stats['certificate_by_budget'] = records(
        sels.groupby(['budget', 'rule']).agg(simplified=('simplified', 'sum'), decisions=('unsafe', 'size'),
                                           unsafe=('unsafe', 'sum'), cost=('cost', 'mean')).reset_index())

    audit = export('equal_budget_audit', pd.DataFrame([r for x in read_json(NEW / 'audit_all.json') for r in x['rows']]))
    arows = []
    for (q, method), g in audit.groupby(['budget_group', 'method']):
        means = g.groupby('network').unsafe.mean().to_numpy()
        lo, hi = boot(means)
        arows.append({'q': q, 'method': method, 'decisions': len(g), 'unsafe': int(g.unsafe.sum()),
                      'unsafe_rate': float(g.unsafe.mean()), 'ci_low': lo, 'ci_high': hi,
                      'agreement': float(g.agreement.mean()), 'cost': float(g.cost.mean())})
    export('equal_budget_audit_summary', pd.DataFrame(arows))
    stats['audit_summary'] = arows
    pairs = []
    for q in [4, 12, 24]:
        g = audit[audit.budget_group == q].groupby(['network', 'method']).unsafe.mean().unstack()
        for baseline in ['uniform', 'density_stratified']:
            delta = g['targeted'] - g[baseline]
            pairs.append({'q': q, 'difference': 'targeted_minus_' + baseline,
                          'mean': float(delta.mean()), 'ci95': boot(delta)})
    stats['audit_paired_network_differences'] = pairs

    weighted = export('heterogeneous_graph', pd.DataFrame(read_json(NEW / 'weighted_all.json')))
    stats['weighted'] = {'settings': len(weighted), 'paths': int(weighted.runs.sum()),
                         'matched': int(weighted.matched_paths.sum()),
                         'maximum_probability_error': float(weighted.max_node_error.max()),
                         'naive_min': float(weighted.shared_beta_naive_error.min()),
                         'naive_max': float(weighted.shared_beta_naive_error.max())}

    oldd = pd.read_csv(REV / 'dynamic_results.csv')
    oldd = oldd[oldd.domain == 'synthetic'].copy()
    oldd['source_batch'] = 'archived_synthetic'
    newd = pd.DataFrame([r for x in read_json(NEW / 'dynamic_all.json') for r in x])
    for col in newd.columns:
        if col not in ('context', 'domain', 'method'):
            newd[col] = pd.to_numeric(newd[col])
    newd['source_batch'] = 'new_chronological_real'
    dyn = export('dynamic_processes', pd.concat([oldd, newd], ignore_index=True))
    ds = dyn.groupby('method').agg(contexts=('context', 'size'), mean_complexity=('mean_complexity', 'mean'),
                                   density_mae=('paired_density_mae', 'mean'),
                                   path_match_rate=('paired_path_match_rate', 'mean'),
                                   stage_exceed_rate=('test_stage_exceed_rate', 'mean'),
                                   simulation_seconds=('simulation_seconds', 'mean'),
                                   fit_seconds=('fit_seconds_shared', 'mean')).reset_index()
    export('dynamic_summary', ds)
    stats['dynamic_summary'] = records(ds)

    syn = export('synthetic_independent_tests', pd.read_csv(REV / 'fair_results.csv').query("domain=='synthetic'").copy())
    stats['synthetic_rmse'] = records(syn.groupby(['mechanism', 'model']).rmse.agg(['mean', 'std']).reset_index())

    real = export('real_timeblock_tests', pd.DataFrame([r for x in read_json(NEW / 'real_all.json') for r in x['rows']]))
    real['ratio_to_SG_time'] = real.rmse / real.set_index(['context', 'model']).rmse.loc[[(x, 'SG-time') for x in real.context]].to_numpy()
    export('real_timeblock_tests', real)
    stats['real_rmse'] = records(real.groupby(['dataset', 'events', 'model']).rmse.agg(['mean', 'std', 'min', 'max']).reset_index())
    stats['real_best_counts'] = real.loc[real.groupby('context').rmse.idxmin()].model.value_counts().to_dict()
    stats['real_case_count'] = int(real.context.nunique())
    stats['real_data_blocks'] = 12
    stats['real_selected_fit_failures'] = int((~real.optimizer_success).sum())
    stats['real_node_range'] = [int(real.nodes.min()), int(real.nodes.max())]

    opt = []
    for p in NEW.rglob('fits.json'):
        for row in read_json(p):
            opt.append({'file': str(p.relative_to(ROOT)), 'model': row.get('model'),
                        'success': row.get('success'), 'message': row.get('message'), 'eta': row.get('eta')})
    export('optimization_status', pd.DataFrame(opt))
    stats['optimization_total'] = len(opt)
    stats['optimization_failures'] = sum(x['success'] is False for x in opt)

    shutil.copy2(ROOT / 'results' / 'tables' / 'innovation2_a11_certificate_tasks.csv',
                 DATA / 'sis_certificates.csv')
    cap = pd.read_csv(ROOT / 'results' / 'tables' / 'innovation2_a12_capacity_collision.csv')
    cap['input_label'] = cap.representation.map({'G': 'psi_G', 'MG': 'psi_MG', 'SC2': 'psi_grp'})
    export('capacity_inputs', cap)
    shutil.copy2(ROOT / 'results' / 'summary_tables' / 'innovation3_a1_synergy_budget_phase.csv',
                 DATA / 'budget_choices.csv')
    shutil.copy2(REV / 'dynamic' / 'synthetic_0' / 'timeline.csv',
                 DATA / 'dynamic_example_timeline.csv')

    (DATA / 'statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'real_cases': stats['real_case_count'],
                      'fit_records': stats['optimization_total'],
                      'fit_failures': stats['optimization_failures'],
                      'weighted': stats['weighted'],
                      'audit_differences': pairs}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
