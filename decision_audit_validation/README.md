# Decision audit validation

Run from the repository root with the dependencies in `revision_20261031/requirements.txt`:

```bash
python decision_audit_validation/run_checks.py --output results/decision_audit_validation
```

This script reproduces two declared checks. The three-player example is synthetic. It varies a hypothetical upper bound on one missing score and verifies allocation intervals and joint ranking margins with a two-variable linear program. It supplies no physical constraint on an alloy.

The utility sensitivity reconstructs the unchanged 140-record eligible NIMS cohort and its nested 71-record H–C cohort from the 1113-record normalized source CSV already supplied in `revision_20261031/magnetic_data/`. No generated analysis output is needed as input. The seven temperature-window/decay settings are listed in the script. They are retrospective sensitivity choices, not fitted parameters or additional experiments. It reports observed-catalog leaders and the signed H-minus-C margin when the joint score is withheld. Positive rescaling of the entropy normalization leaves the observed-catalog ranking unchanged; this is verified separately.

Outputs are generated in the requested local directory. Numerical outputs are not stored in this folder. The source input checksum and numerical assertions are written with the generated results. The calculations do not evaluate an adaptive measurement policy or demonstrate scalability beyond the small player sets used in the paper.
