from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from completion import weights, envelope, permutation_shapley, solve
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"magnetic_results"
OUT.mkdir(exist_ok=True)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))

def audit_scores(frame, players, score="payoff", mask="mask"):
    n = len(players)
    q = frame.groupby(mask)[score].max().to_dict()
    v = envelope([q.get(s, 0) for s in range(1 << n)])
    W = weights(n)
    phi = W @ v
    assert np.allclose(phi, permutation_shapley(v))
    assert abs(phi.sum() - (v[-1] - v[0])) < 1e-9
    supports = sorted(q)
    B = np.array([[(s >> i) & 1 for i in range(n)] for s in supports])
    return {"records": len(frame), "supports": len(q), "possible_supports": 1 << n,
            "rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(B)), B]))),
            "rank_columns": n + 1, "empty_support_observed": 0 in q,
            "v_empty": float(v[0]), "v_full": float(v[-1]),
            "veto_incidence": [p for i,p in enumerate(players) if B[:,i].all()],
            "equivalent_incidence_pairs": [[players[i],players[j]] for i,j in combinations(range(n), 2)
                                           if np.array_equal(B[:,i],B[:,j])],
            "direct_contrasts": {p: sum(not(s >> i & 1) and (s | 1 << i) in q for s in q)
                                 for i,p in enumerate(players)},
            "shapley": {p: float(phi[i]) for i,p in enumerate(players)}}, q, v

def all_bounds(label, players, observed, caps, modes=("score", "access")):
    records, witnesses = [], []
    for cap_label, cap in caps.items():
        for mode in modes:
            for i, p in enumerate(players):
                entry = {"case": label, "cap_class": cap_label, "cap": cap,
                         "completion_class": mode, "player": p}
                for obj in ("min", "max", "lead", "trail"):
                    r = solve(len(players), observed, cap, i, obj, mode)
                    entry[obj] = r["value"]
                    for s in range(1 << len(players)):
                        witnesses.append({**{k: entry[k] for k in ("case", "cap_class", "cap", "completion_class", "player")},
                                          "objective": obj, "mask": s,
                                          "observed_score": s in observed,
                                          "q": r["q"][s], "v": r["v"][s],
                                          **{f"phi_{p2}": r["phi"][j] for j,p2 in enumerate(players)},
                                          "objective_value": r["value"], "solver_objective": r["objective"],
                                          "dual_bound": r["dual_bound"], "mip_gap": r["gap"],
                                          "envelope_residual": r["envelope_residual"],
                                          "selector_binaries": r["selector_binaries"]})
                records.append(entry)
                print(label, cap_label, mode, p, {x: round(entry[x], 6) for x in ("min", "max", "lead", "trail")}, flush=True)
    return pd.DataFrame(records), pd.DataFrame(witnesses)
