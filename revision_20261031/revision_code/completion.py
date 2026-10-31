"""Exact bounded completions of finite maximum-access games.

Known support scores and known coalition access values are different evidence.
All solutions are computational statements about a declared finite game, not
physical response predictions. Empty-support scores need not be zero.
"""
from math import factorial
from itertools import permutations, product
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse import csr_matrix


def weights(n):
    W = np.zeros((n, 1 << n))
    for i in range(n):
        for s in range(1 << n):
            if s & (1 << i):
                continue
            k = s.bit_count()
            w = factorial(k) * factorial(n-k-1) / factorial(n)
            W[i, s] -= w
            W[i, s | (1 << i)] += w
    return W


def envelope(q):
    q = np.asarray(q, float)
    return np.array([max(q[s] for s in range(len(q)) if s & c == s)
                     for c in range(len(q))])


def permutation_shapley(v):
    n = int(np.log2(len(v)))
    phi = np.zeros(n)
    for order in permutations(range(n)):
        s = 0
        for i in order:
            nxt = s | (1 << i)
            phi[i] += v[nxt] - v[s]
            s = nxt
    return phi / factorial(n)


def solve(n, observed, cap, player, objective="min", mode="score", contrast=None):
    """Optimize Shapley or a strict-rank margin over a declared completion class.

    score: only q(s) for measured supports is fixed; v=max-access envelope.
    access: v(s) is fixed at the currently observed envelope on measured supports.
    Every returned witness fixes every supplied measured score exactly.
    """
    if 0 not in observed:
        raise ValueError("Specify the measured or explicitly assumed empty-support score.")
    if any(not 0 <= q <= cap for q in observed.values()):
        raise ValueError("Scores must be in [0, cap].")
    N = 1 << n
    W = weights(n)
    coefficient = W[player] if contrast is None else W[player] - W[contrast]
    rank = objective in ("lead", "trail")
    # For an observed C, its envelope is max(q(C), v(C\\{i}): i in C).
    # Unknown supports can be reconstructed as q(C)=v(C).
    choices = [(s, [-1] + [s ^ (1 << i) for i in range(n) if s >> i & 1])
               for s in observed if s]
    nb = sum(len(p) for _, p in choices) if mode == "score" else 0
    total = N + nb + int(rank)
    lower = np.zeros(total)
    upper = np.full(total, cap, dtype=float)
    if nb:
        upper[N:N+nb] = 1
    lower[0] = upper[0] = observed[0]
    if rank:
        lower[-1], upper[-1] = -cap, cap
    rows, lo, hi = [], [], []

    def add(entries, low=-np.inf, high=np.inf):
        row = np.zeros(total)
        for k, a in entries.items():
            row[k] = a
        rows.append(row); lo.append(low); hi.append(high)

    for s in range(N):
        for i in range(n):
            if not s >> i & 1:
                add({s: 1, s | (1 << i): -1}, high=0)
    observed_envelope = envelope([observed.get(s, 0) for s in range(N)])
    for s, q in observed.items():
        lower[s] = max(lower[s], q)
        if mode == "access":
            lower[s] = upper[s] = observed_envelope[s]
    if mode == "score":
        j = N
        for s, parents in choices:
            selectors = []
            for parent in parents:
                selectors.append(j)
                # z=1 forces v(C)<=the selected candidate maximum.
                if parent == -1:
                    add({s: 1, j: cap}, high=observed[s] + cap)
                else:
                    add({s: 1, parent: -1, j: cap}, high=cap)
                j += 1
            add({z: 1 for z in selectors}, low=1, high=1)
    elif mode != "access":
        raise ValueError(mode)
    c = np.zeros(total)
    if rank:
        sign = 1 if objective == "lead" else -1
        for other in range(n):
            if other != player:
                row = np.zeros(total)
                row[:N] = -sign * (W[player] - W[other])
                row[-1] = 1
                rows.append(row); lo.append(-np.inf); hi.append(0)
        c[-1] = -1
    else:
        c[:N] = coefficient if objective == "min" else -coefficient
    integrality = np.zeros(total)
    integrality[N:N+nb] = 1
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(csr_matrix(rows), lo, hi),
                  options={"mip_rel_gap": 0.0, "time_limit": 90.0})
    if not result.success:
        raise RuntimeError(f"Completion optimization failed: {result.message}")
    v = result.x[:N]
    q = v.copy()
    for s, value in observed.items():
        q[s] = value
    phi = W @ v
    residual = float(np.max(np.abs(envelope(q) - v)))
    assert residual < 2e-7, residual
    assert np.max(np.abs(permutation_shapley(v) - phi)) < 1e-9
    assert abs(phi.sum() - (v[-1] - v[0])) < 1e-8
    assert all(abs(q[s] - value) < 1e-12 for s, value in observed.items())
    assert np.min(q) > -1e-7 and np.max(q) < cap + 1e-7
    value = result.x[-1] if rank else float(coefficient @ v)
    bound = getattr(result, "mip_dual_bound", None)
    dual = float(result.fun if bound is None else bound)
    assert abs(result.fun-dual) < 2e-7, (result.fun, dual)
    return {"value": float(value), "q": q, "v": v, "phi": phi,
            "objective": float(result.fun), "dual_bound": dual,
            "gap": float(getattr(result, "mip_gap", 0) or 0),
            "envelope_residual": residual, "selector_binaries": nb}


def verification_tests():
    # Independent analytic two-player test: q0=0, qA=.2, qB unknown,
    # qAB=.8. Then phiA=.5*(.2+max(.8,qB)-qB), hence [.1,.5].
    low = solve(2, {0: 0., 1: .2, 3: .8}, 1, 0, "min")
    high = solve(2, {0: 0., 1: .2, 3: .8}, 1, 0, "max")
    assert abs(low["value"]-.1) < 1e-8
    assert abs(high["value"]-.5) < 1e-8
    # Complete scores with nonzero empty baseline: exact allocation, and
    # shifting every support by a constant leaves Shapley unchanged.
    q = np.array([.25, .6, .45, .7])
    target = permutation_shapley(envelope(q))
    for i in range(2):
        for objective in ("min", "max"):
            r = solve(2, dict(enumerate(q)), 1, i, objective)
            assert abs(r["value"]-target[i]) < 1e-8
    assert np.allclose(target, permutation_shapley(envelope(q-.25)))
    # Sparse three-player exact solution must contain every independently
    # enumerated score completion at all observed-value/bound breakpoints.
    obs = {0: 0., 3: .4, 7: .7}
    missing = [s for s in range(8) if s not in obs]
    values = []
    for vals in product([0., .4, .7, 1.], repeat=len(missing)):
        x = np.zeros(8)
        for s, z in obs.items(): x[s] = z
        for s, z in zip(missing, vals): x[s] = z
        values.append(permutation_shapley(envelope(x)))
    values = np.asarray(values)
    for i in range(3):
        for objective, expected in [("min", values[:, i].min()), ("max", values[:, i].max())]:
            r = solve(3, obs, 1, i, objective)
            assert abs(r["value"]-expected) < 1e-8, (i, objective)
    return {"analytic_two_player": True, "nonzero_empty_baseline": True,
            "constant_shift_invariance": True, "exhaustive_three_player_grid": True}
