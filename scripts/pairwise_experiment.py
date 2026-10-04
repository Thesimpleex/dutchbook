"""Local versus global coherence on the ForecastBench question graphs.

For one model configuration, round and resolution date, the combination tuples form a graph:
questions are nodes and every tuple is an edge. Each edge table is replaced by the closest
joint table with the model's own marginals (``marginal_consistent``), so every pair is
coherent by construction and all pairs agree on shared marginals. Whatever incoherence
remains is global and can only exist on connected components with a cycle; on forests the
theorem says none remains, which this experiment also checks on real data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from _common import CACHE, RAW, log, update_results

from dutchbook import PairwiseSystem, audit_pairwise, from_tuples, marginal_consistent
from dutchbook import forecastbench as fb

CONFIGS = (
    "Claude-3-5-Sonnet-20240620 (zero shot)",
    "GPT-4o-2024-05-13 (zero shot)",
    "Llama-3.3-70B-Instruct-Turbo (zero shot)",
)
ENUMERATE_UP_TO = 10
MAX_ATOMS = 30


def audit_component(comp: PairwiseSystem) -> dict[str, float | int | bool]:
    method = "enumerate" if len(comp.marginals) <= ENUMERATE_UP_TO else "column-generation"
    (rep,) = audit_pairwise(comp, method=method)  # type: ignore[arg-type]
    return {
        "atoms": len(comp.marginals),
        "edges": len(comp.edges),
        "forest": rep.is_forest,
        "max_local": rep.max_local_profit,
        "global": rep.global_profit,
    }


def components_of(table: fb.TupleTable, model_name: str) -> list[tuple[str, str, PairwiseSystem]]:
    mi = next(i for i, m in enumerate(table.models) if m[1] == model_name)
    rows = np.where(table.model_idx == mi)[0]
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i in rows:
        groups[(int(table.round_idx[i]), str(table.resolution_date[i]))].append(int(i))
    out = []
    for (r, date), idx in sorted(groups.items()):
        pairs = [tuple(table.pair[i].split("|")) for i in idx]
        system = marginal_consistent(from_tuples(pairs, table.q[idx]))  # type: ignore[arg-type]
        for comp in system.components():
            out.append((table.rounds[r], date, comp))
    return out


def summarise(rows: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    def part(sel: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
        if not sel:
            return {"components": 0}
        g = np.array([float(r["global"]) for r in sel])
        return {
            "components": len(sel),
            "globally_incoherent": int((g > 1e-9).sum()),
            "share_globally_incoherent": float((g > 1e-9).mean()),
            "mean_global_profit": float(g.mean()),
            "max_global_profit": float(g.max()),
            "mean_atoms": float(np.mean([r["atoms"] for r in sel])),
            "max_atoms": int(max(int(r["atoms"]) for r in sel)),
        }

    return {
        "forests": part([r for r in rows if r["forest"]]),
        "with_cycles": part([r for r in rows if not r["forest"]]),
        "max_local_profit_over_all_edges": float(max(float(r["max_local"]) for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    table = fb.load_tuples(RAW, CACHE)
    result = {}
    for name in CONFIGS:
        every = components_of(table, name)
        comps = [c for c in every if len(c[2].marginals) <= MAX_ATOMS]
        log(f"{name}: {len(comps)} of {len(every)} components with at most {MAX_ATOMS} atoms")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(audit_component, [c for _, _, c in comps], chunksize=4))
        result[name] = summarise(rows)
        result[name]["components_total"] = len(every)
        result[name]["components_skipped_over_max_atoms"] = len(every) - len(comps)
        log(f"  {result[name]}")
    update_results(
        "pairwise_graphs",
        {
            "configurations": result,
            "enumerate_up_to_atoms": ENUMERATE_UP_TO,
            "max_atoms_per_component": MAX_ATOMS,
        },
    )


if __name__ == "__main__":
    main()
