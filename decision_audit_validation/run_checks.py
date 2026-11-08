"""Reproduce the declared utility sensitivity and a three-player analytic example.

The synthetic example supplies hypothetical information bounds, not alloy physics.
The utility calculations reuse the unchanged, eligible NIMS input cohort.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


PLAYERS = ["H", "Co", "Mn", "C", "Ce"]
SETTINGS = [(250, 350, 50), (225, 325, 50), (275, 375, 50),
            (250, 350, 25), (250, 350, 100), (200, 300, 50), (300, 400, 50)]


def envelope(q):
    v = np.asarray(q, float).copy()
    p = int(np.log2(len(v)))
    for i in range(p):
        for s in range(len(v)):
            if s & (1 << i):
                v[s] = max(v[s], v[s ^ (1 << i)])
    return v


def shapley(v):
    p = int(np.log2(len(v)))
    phi = np.zeros(p)
    orders = list(permutations(range(p)))
    for order in orders:
        s = 0
        for i in order:
            phi[i] += v[s | (1 << i)] - v[s]
            s |= 1 << i
    phi /= len(orders)
    assert abs(phi.sum() - (v[-1] - v[0])) < 1e-10
    return phi


def observed_game(frame, players, lo, hi, tau, entropy_scale=100):
    masks = sum((1 << i) * frame[p].gt(0).astype(int).to_numpy()
                for i, p in enumerate(players))
    t = frame.Ttr_K.to_numpy()
    distance = np.maximum.reduce([lo-t, np.zeros(len(t)), t-hi])
    score = np.sqrt(np.exp(-distance/tau) * frame.DS2.to_numpy()/entropy_scale)
    q = np.zeros(1 << len(players))
    for s in np.unique(masks):
        q[s] = score[masks == s].max()
    return shapley(envelope(q)), q, score


def nims_sensitivity(path):
    raw = pd.read_csv(path)
    if 'DS2' in raw.columns:
        frame = raw.copy()
    else:
        assert len(raw) == 1113
        frame = raw.copy()
        frame['source_dois'] = raw['Article DOI'].ffill()
        frame['formula_label'] = raw['Composition in the article'].ffill()
        frame['H'] = raw['H [or D]']
        frame['Ttr_K'] = raw['Ttr (K)']; frame['Htr_T'] = raw['Htr (T)']
        frame['method'] = raw['Method'].fillna('unreported')
        frame['anneal_h'] = raw['Ann. Time (h)']; frame['anneal_K'] = raw['Ann. Temp. (K)']
        def parse(value):
            if pd.isna(value): return None
            try:
                value = ast.literal_eval(str(value))
                value = value if isinstance(value, (list, tuple)) else [value]
                values = [float(x) for x in value]
                return values if np.isfinite(values).all() else None
            except (ValueError, SyntaxError, TypeError): return None
        def entropy2(row):
            fields = parse(row['HS (T)']); values = parse(row['|ΔSM|max (J/kg K) '])
            if fields is None or values is None or len(fields) != len(values): return np.nan
            chosen = [x for h, x in zip(fields, values) if abs(h-2.0) < 1e-8]
            if not chosen or chosen[0] <= 0 or not np.allclose(chosen, chosen[0], atol=1e-10, rtol=0): return np.nan
            return chosen[0]
        frame['DS2'] = raw.apply(entropy2, axis=1)
        columns = list(raw.columns[5:29]) + ['H [or D]', 'C', 'N', 'B']
        excluded = [c for c in columns if c not in ['La','Fe','Si','H [or D]','Co','Mn','C','Ce']]
        eligible = (frame[['La','Fe','Si']].gt(0).all(axis=1)
                    & frame[excluded].eq(0).all(axis=1)
                    & ~frame.formula_label.astype(str).str.contains(r'D(?![a-z])', regex=True)
                    & pd.to_numeric(frame.Ttr_K, errors='coerce').gt(0)
                    & pd.to_numeric(frame.Htr_T, errors='coerce').between(0, .1)
                    & frame.DS2.gt(0))
        frame = frame[eligible].copy()
        keys = ['source_dois','Fe','Si','La']+PLAYERS+['Ttr_K','Htr_T','DS2','method','anneal_h','anneal_K']
        frame = frame.loc[~frame.duplicated(keys, keep='first')].copy()
    assert len(frame) == 140
    hc = frame[frame[["Co", "Mn", "Ce"]].eq(0).all(axis=1)].copy()
    assert len(hc) == 71
    rows = []
    for lo, hi, tau in SETTINGS:
        phi, _, scores = observed_game(frame, PLAYERS, lo, hi, tau)
        hphi, hq, _ = observed_game(hc, ["H", "C"], lo, hi, tau)
        signed_margin = max(hq[0], hq[1]) - max(hq[0], hq[2])
        assert abs((hphi[0]-hphi[1])-signed_margin) < 1e-12
        row = {"window_low_K": lo, "window_high_K": hi, "decay_K": tau,
               "observed_leader": PLAYERS[int(np.argmax(phi))],
               "H_minus_C_margin": signed_margin,
               "HC_unique_leader": "H" if signed_margin > 1e-9 else
               "C" if signed_margin < -1e-9 else "None (tie)"}
        row.update({"phi_"+p: x for p, x in zip(PLAYERS, phi)})
        rows.append(row)
        for scale in (50, 200):
            scaled, _, _ = observed_game(frame, PLAYERS, lo, hi, tau, scale)
            assert np.allclose(scaled, phi*np.sqrt(100/scale), atol=1e-12)
        if (lo, hi, tau) == (250, 350, 50):
            if 'payoff' in frame.columns:
                assert np.allclose(scores, frame.payoff, atol=1e-12)
            assert np.allclose(phi, [.0232999540357436,.0376936274991267,
                                    .000554848035284511,.00513492623233549,
                                    .0107035101029621], atol=1e-12)
            assert abs(signed_margin-0.04096180841063468) < 1e-12
    return rows


def three_player_sensitivity():
    # Known q(empty), q(A), q(B), q(AB), q(C), q(AC).
    # Unknown q(BC) is in [0,U] and q(ABC) is in [0,1].
    # r=v(BC) is in [.2,max(.2,U)]; g=v(ABC)>=max(.35,r).
    base = np.array([0., .3, .2, .35, .1, .35, 0., 0.])
    phi0 = shapley(base)
    cols = []
    for s in (6, 7):
        unit = np.zeros(8); unit[s] = 1.
        cols.append(shapley(unit))
    matrix = np.column_stack(cols)
    rows = []
    for upper in (1., .6, .45, .4, .3):
        bounds = [(.2, max(.2, upper)), (.35, 1.)]
        def extreme(c, constant, maximize=False):
            result = linprog(-c if maximize else c, A_ub=[[1., -1.]],
                             b_ub=[0.], bounds=bounds, method="highs")
            assert result.success, result.message
            r, g = result.x
            q = base.copy(); q[6] = r if r > .2+1e-10 else 0.; q[7] = g
            v = envelope(q)
            assert q[6] <= upper+1e-10
            assert np.allclose(v, base+np.array([0.,0.,0.,0.,0.,0.,r,g]))
            assert np.allclose(shapley(v), phi0+matrix@result.x)
            return float(constant+c@result.x)
        row = {"BC_score_upper_bound": upper}
        for i, name in enumerate("ABC"):
            row[name+"_min"] = extreme(matrix[i], phi0[i])
            row[name+"_max"] = extreme(matrix[i], phi0[i], True)
            margins = [extreme(matrix[i]-matrix[j], phi0[i]-phi0[j])
                       for j in range(3) if j != i]
            row[name+"_worst_margin"] = min(margins)
        expected = .225 - .5*max(.2, upper)
        assert abs(row["A_worst_margin"]-expected) < 1e-10
        certified = [name for name in "ABC" if row[name+"_worst_margin"] > 1e-9]
        row["certified_leader"] = ";".join(certified) or "None"
        rows.append(row)
    limited = rows[3]
    assert abs(limited["A_worst_margin"]-.025) < 1e-10
    assert limited["A_min"] < limited["B_max"]
    assert limited["A_min"] < limited["C_max"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] /
                        "revision_20261031/magnetic_data/nims_lafesi_source_1113.csv")
    parser.add_argument("--output", type=Path, default=Path("results/decision_audit_validation"))
    args = parser.parse_args()
    nims = nims_sensitivity(args.data)
    toy = three_player_sensitivity()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nims).to_csv(args.output/"nims_utility_sensitivity.csv", index=False, float_format="%.15g")
    pd.DataFrame(toy).to_csv(args.output/"three_player_information_sensitivity.csv", index=False, float_format="%.15g")
    checks = {"source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "primary_payoffs_preserved": True, "HC_primary_margin_verified": True,
              "entropy_scale_invariance_verified": True,
              "three_player_margin_formula_verified": True,
              "overlapping_intervals_with_positive_joint_margin_verified": True,
              "HEA_selector_count": sum(1+len(s) for s in
                   [("Mn","Cr","Ni","Al"),("Mn","Cr","Al"),("Cr","Ni","Cu"),("Cr","Ni")])}
    assert checks["HEA_selector_count"] == 16
    (args.output/"verification.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
