"""Numerical validation on random instances; writes the ``validation`` section of results.json.

Every check is an invariant that must hold up to the stated tolerance: strong LP duality
between the Dutch-book LP and the distance LP, Wolfe's projection against exact support
enumeration, first-order optimality, the generalised Pythagorean inequality for the Brier
and log scores, and coherent extension bounds against the modus-ponens closed form.
"""

from __future__ import annotations

import itertools

import numpy as np
from _common import log, update_results

from dutchbook import (
    coherence_distance,
    dominance_certificate,
    euclidean_projection,
    extension_bounds,
    find_dutch_book,
    kl_projection,
)

SEED = 20261004


def random_truth_matrix(rng: np.random.Generator, n_atoms: int, n_formulas: int) -> np.ndarray:
    cols = rng.random((2**n_atoms, n_formulas)) < 0.5
    return np.unique(cols.astype(float), axis=0)


def brute_force_projection(W: np.ndarray, q: np.ndarray) -> np.ndarray:
    k = W.shape[0]
    best, best_p = np.inf, W[0]
    for r in range(1, k + 1):
        for support in itertools.combinations(range(k), r):
            V = W[list(support)]
            kkt = np.zeros((r + 1, r + 1))
            kkt[:r, :r] = 2 * V @ V.T
            kkt[:r, r] = 1.0
            kkt[r, :r] = 1.0
            lam = np.linalg.lstsq(kkt, np.concatenate([2 * V @ q, [1.0]]), rcond=None)[0][:r]
            if (lam >= -1e-12).all() and abs(lam.sum() - 1) < 1e-9:
                p = V.T @ np.clip(lam, 0, None)
                d = float(np.sum((p - q) ** 2))
                if d < best - 1e-15:
                    best, best_p = d, p
    return best_p


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = 2000
    duality = {"l1": 0.0, "linf": 0.0, "l2": 0.0}
    proj_err = kkt = 0.0
    n_brute = 0
    pyth_brier = pyth_log = np.inf
    gap_max = gap_max_kl = 0.0
    for i in range(n):
        W = random_truth_matrix(rng, int(rng.integers(1, 5)), int(rng.integers(2, 8)))
        q = rng.random(W.shape[1])
        for stake, dist in (("l1", "linf"), ("linf", "l1"), ("l2", "l2")):
            book = find_dutch_book(W, q, stake)
            duality[stake] = max(
                duality[stake], abs(book.profit - coherence_distance(W, q, dist).value)
            )
        proj = euclidean_projection(W, q)
        g = proj.point - q
        kkt = max(kkt, float(-(W @ g - g @ proj.point).min()))
        if W.shape[0] <= 7:
            n_brute += 1
            proj_err = max(proj_err, float(np.abs(proj.point - brute_force_projection(W, q)).max()))
        cert = dominance_certificate(W, q, proj.point, "brier", proj.gap)
        pyth_brier = min(pyth_brier, float(cert.margin.min()))
        gap_max = max(gap_max, proj.gap)
        klp = kl_projection(W, q)
        certk = dominance_certificate(W, klp.q, klp.point, "log", klp.gap)
        pyth_log = min(pyth_log, float(certk.margin.min()))
        gap_max_kl = max(gap_max_kl, klp.gap)
        if i % 500 == 0:
            log(f"validation {i}/{n}")
    ext_err = 0.0
    for _ in range(500):
        a, c = rng.random(2)
        b = extension_bounds(["A", "A & B"], [a, a * c], "B")
        ext_err = max(ext_err, abs(b.lower - a * c), abs(b.upper - (a * c + 1 - a)))
    update_results(
        "validation",
        {
            "seed": SEED,
            "random_instances": n,
            "max_abs_duality_error": duality,
            "projection_instances_vs_brute_force": n_brute,
            "max_abs_error_projection_vs_brute_force": proj_err,
            "max_kkt_violation": kkt,
            "min_pythagorean_margin_brier": pyth_brier,
            "max_duality_gap_brier": gap_max,
            "min_pythagorean_margin_log": pyth_log,
            "max_duality_gap_log": gap_max_kl,
            "modus_ponens_instances": 500,
            "max_abs_error_modus_ponens_bounds": ext_err,
        },
    )
    log("validation done")


if __name__ == "__main__":
    main()
