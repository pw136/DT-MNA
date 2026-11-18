# DT-MNA: Dynamical Transformability and Modeling Necessity Assessment

Reproduction package for the experiments, figures, and tables reported in the
manuscript "Dynamical Transformability and Modeling Necessity Assessment in
Temporal Higher-Order Networks". Running the pipeline regenerates every numeric
output under `results/` and writes the manuscript figures to `results/figures/`
(`images1.pdf` … `images9.pdf`, `ext_A_temporal.pdf`, `ext_B_approx.pdf`,
`ext_C_selection.pdf`, `sir_bounds.pdf`) and table evidence CSVs to
`results/tables/manuscript/`.

## Requirements

- Python 3.14 (developed and measured on Python 3.14.6, Windows; any
  platform with the pinned packages works)
- `pip install -r requirements.txt`
- Arial (or a metric-compatible substitute) for the figure scripts
- Third-party code for the external baseline steps (`infra`, `e2`, `ext_a`,
  `ext_b`, `ext_c`, `sir_empirical`, `audit_scale`; see below)

## Third-party dependency (Lucas et al. reducibility)

The external-baseline experiments call the hypergraph reducibility code of
Lucas et al. Clone it at the pinned commit before running the external steps:

    git clone https://github.com/maximelucas/hypergraph_reducibility third_party/hypergraph_reducibility
    git -C third_party/hypergraph_reducibility checkout bd3fbedf110b4304d20d561d4f8816e1b27623c9

The steps that need it fail with a clear error message if the directory is
absent. All other steps run without it.

## Usage

    python -X utf8 reproduce.py            # run all steps in dependency order
    python -X utf8 reproduce.py --list     # list steps and dependencies
    python -X utf8 reproduce.py x1 x4      # run selected steps (dependencies
                                           # of selected steps are added
                                           # automatically)
    python -X utf8 reproduce.py --jobs 20  # parallel workers where supported
                                             # (paper_core, csf, x2, x3b)

All outputs are written under `results/` inside this repository; nothing is
written outside the repository root.

## Step runtimes

Measured on a 28-logical-core Windows workstation, Python 3.14.6, with the
stated parallel options. Steps marked (*) take hours; the values shown are the project-record
runtimes on the same machine class. All other values are measured.

| Step | Content | Runtime |
|---|---|---|
| infra | dataset preprocessing, calibration, external methods | 10-12 min |
| proof | temporal vs. static proof, 200 shuffles | 3 s |
| conversion | exact-transformation validation | 40 s |
| overlay | trajectory overlay | 16 s |
| innovation2 | approximate transformability suite A0:A10 | 33 s |
| supplementary | certificate + capacity experiments | 4.7 min |
| innovation3 | higher-order necessity suite | 1.8 min |
| e4 | tail-setting conflict scan | 4 s |
| e7 / x4 | group-size storage and runtime | 2.4 min / 13 s |
| x1_bounds / x1_refit | bound-check and refit | 89 s / 26 s |
| x1 | tail-sensitivity grid + `images2` | 10 s |
| paper_core | certificates, audits, real and dynamic suites | 3 min (--jobs 8) |
| revision | fair tests, dynamic processes | 3.5 min |
| csf | exact/heterogeneous/SIS strengthening | 3 min (--workers 8) |
| csf_bench | sequential cost benchmark for the cost table | 2 min |
| e2 | BIC refit on six datasets | 37 min |
| x2 / x3b | score variants / literature mechanisms | 19 s / 61 s (--jobs 8) |
| ext_a / ext_b / ext_c | external baselines A/B/C | (*) 1.8 h / 6.6 h / 2.5 h |
| sir_small | small-network SIR bounds | 2.2 min |
| sir_empirical | SIR selection on empirical windows | (*) 3.9 h |
| audit_scale | audit coverage on empirical windows | (*) 35 min |
| dataset_stats | empirical dataset statistics | 7 s |
| collect / figures_* / ext_figures / sir_figure / tables | data joins, figures, table CSVs | 4-16 s each |

`ext_a` through `ext_c` are the most expensive steps; they can run in parallel
from separate terminals (`python -X utf8 reproduce.py ext_a` etc.). With all
steps in one invocation, the steps already listed in the dependency graph run
automatically.

Results folders carry a code-version tag. Rerunning a step whose previous
outputs were produced by different code requires deleting that step's results
folder first (for example `results/results_revision_supplement`).

## Figure/table to step mapping

| Output | Produced by |
|---|---|
| `images1` | `figures_core` (data via `collect`, inputs `proof`, `paper_core`) |
| `images2` | `x1` (tail grid; capacity inputs via `collect` and `supplementary`) |
| `images3` | `figures_csf` (certificates via `collect`; axis labels added with pypdf) |
| `images4` | `figures_csf` (input `csf`) |
| `images5`, `images6` | `figures_core` (inputs `revision` fair tests, `paper_core` real tests) |
| `images7`, `images8` | `figures_redesign` (inputs `csf`, `collect` audit/budget joins) |
| `images9` | `figures_core` (dynamic processes via `collect`, inputs `revision`, `paper_core`) |
| `ext_A_temporal`, `ext_B_approx`, `ext_C_selection` | `ext_figures` (inputs `ext_a`, `ext_b`, `ext_c`) |
| `sir_bounds` | `sir_figure` (input `sir_small`) |
| Params/cost table | `tables` (input `csf_bench` runtime.csv) |
| Complete-process table | `tables` (input `collect` dynamic joins) |
| External tables (datasets, temporal, approx, selection) | `tables` (inputs `infra`, `ext_a/b/c` summaries) |
| Tail / score / size-cost appendix tables | `tables` (inputs `x1`, `x2`, `x4`) |
| Dataset stats, SIR selection, audit coverage | `tables` (inputs `dataset_stats`, `sir_empirical`, `audit_scale`) |
| In-text temporal-ordering numbers | `results/figure_data/statistics.json` (input `proof`) |
| In-text certificate/audit numbers | `results/figure_data/statistics.json` (input `paper_core`) |
| In-text BIC refit numbers | `results/rev2/E2/` setting JSONs |
| In-text storage recompute | `results/rev3/X4/summary.csv` |

## Datasets

The repository includes the datasets required by the experiments:

- `data/email-Enron`, `data/NDC-classes-full`, `data/contact-high-school`,
  `data/contact-primary-school`, temporal higher-order interaction data from
  Benson et al., "Simplicial closure and higher-order link prediction",
  PNAS 115(48), 2018 (Austin R. Benson's datasets page). See the README /
  DATA-DESCRIPTION files inside each folder.
- `data/external_baselines/raw/*.json`, six temporal datasets
  (`contact-primary-school`, `contact-high-school`, `hospital-lyon`, `invs15`,
  `sfhh-conference`, `email-eu`) in XGI-data / SocioPatterns JSON format
  (github.com/xgi-org/xgi-data).

## Layout

    reproduce.py           pipeline entry point
    requirements.txt       pinned dependencies
    configs/               experiment configs used by the pipeline
    data/                  input datasets (see above)
    src/                   library + experiment modules
    scripts/               data collection, figure builders, table builders
    third_party/           (created by you: Lucas et al. code, see above)

`src/` and `scripts/` ship without comments or docstrings; the code is
otherwise identical to the code that produced the paper results.
