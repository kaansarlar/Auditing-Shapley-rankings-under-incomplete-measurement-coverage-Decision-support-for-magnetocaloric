# Magnetocaloric Shapley audit revision

Companion source and inputs for **Shapley identifiability and measurement coverage in magnetocaloric alloy portfolios** (31 October 2026).

## Reproduce

Use Python 3.11 or later with the package versions in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

No network access is required after installing dependencies. The two source CSV inputs are included. The run selects the 140-record La–Fe–Si cohort and 71-record nested H–C cohort; computes exact completion bounds, ranking witnesses, all eight H–C information states, source-omission and source-grouped representation checks; runs 2000+2000 paired synthetic worlds with 1000 bootstrap replicates; and creates the manuscript figures.

Outputs are written to `magnetic_results/`, `revision_code/results/` and `magnetic_figures/`. Those directories are generated locally and excluded from Git tracking. Runtime is typically a few minutes and depends on the solver and machine. Completion witnesses are checked by an independent permutation formula and exact access-envelope reconstruction. Solver status and objective bounds are recorded. `SOURCE_PROVENANCE.md` describes data origins and licenses.

The new measured-score class fixes attained scores only. The stronger access-value class also fixes observed coalition access values. Bounds, point allocations and ranking certificates are conditional on the declared finite table, utility, baseline and cap. They are not confidence intervals or physical dopant effects.

The HEA baseline and historical long benchmark outputs are archived at https://doi.org/10.5281/zenodo.23602541. This source folder supplements that version; it does not change its contents. The submission ZIP supplies generated revision outputs and a verification record in addition to this source-only folder.
