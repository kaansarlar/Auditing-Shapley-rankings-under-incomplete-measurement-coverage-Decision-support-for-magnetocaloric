# Revision validation on 8 November 2026

Run from the repository root with the existing requirements:

```bash
python revision_20261108/run_policy_comparison.py
python revision_20261108/run_policy_diagnostics.py
python revision_20261108/run_nims_threshold.py
python revision_20261108/run_scaling.py
python revision_20261108/run_dense_scaling.py
python revision_20261031/revision_code/run_magnetic.py
python revision_20261108/plot_completion.py
```

The policy experiment uses 2,000 paired worlds per generator, two initial
coverage layouts, three noise regimes, two pooling rules and three policies.
It writes every world result, input utility/noise table, adaptive and fixed
support sequence, summary and paired bootstrap contrast under `results/`.
Both layouts have four supports of sizes 2,3,3,4 and counts 1,3,7,6.
The second layout has neither a veto player nor equal incidence columns.
This layout contrast changes several structural features together; it does
not identify a causal effect of veto removal alone.

The adaptive utility GP-UCB comparator uses a fixed prior and working noise
model. It may revisit a support. Its ranking evaluation still uses exactly
the observed-support estimator used by the other policies. It is not an
optimal ranking-acquisition method or a distribution-free certificate.
No method consults a hidden utility before that support is observed.

The saved-world diagnostic reports early-budget recovery transitions with and
without sensor noise. Recovery can fall when additional supports change the
induced incomplete-catalog game, even without observation error. At budget 27,
random expansion covers all 31 nonempty supports and exceeds GP-UCB in all four
independent-noise comparisons; GP-UCB visits about 12.5–13.5 supports. Its
budget-16 advantage is therefore not uniform across budgets.

The NIMS experiment conditions on the unchanged primary utility and cohort.
Its uniform upper bound on absent support scores is hypothetical additional
information, not a physical performance bound established by the data.
No single extra support-score revelation can certify Co under the remaining
unit caps: constructive all-value obstruction witnesses are exported.
Checking a fixed competitor at all envelope breakpoints covers every
intermediate revealed score by piecewise linearity.

The runtime experiment solves one pairwise-margin problem per sparse instance,
with four random nonempty measured supports and three replicates at each of
5 through 10 players. This does not measure full-audit cost or difficult dense
instances. A second check adds approximately 25% and 50% observed coverage
at 5, 6 and 8 players, two replicates per cell, both random observed scores
and a constructed leader whose singleton reaches the unit cap. It produces
negative and positive pairwise margins. These remain one-margin timing
checks rather than full-audit or worst-case benchmarks.
The subset maximum transform uses O(p 2^p) operations. The MILP
still requires exponentially many access variables; dense constraint assembly
also limits memory. Above seven players, factorial permutation enumeration is
skipped; reconstruction and efficiency remain checked for every witness.

All analyses are retrospective revision checks. Generated outputs are ignored
by git and can be recreated with the commands above. Original simulations
and their seed remain unchanged.
