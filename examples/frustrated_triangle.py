"""Local coherence does not imply global coherence on a graph with a cycle.

Three events a, b, c with every pair "always different": each pair of forecasts is
coherent on its own, but no assignment of three binary values makes all three pairs differ.
"""

from dutchbook import PairwiseSystem, audit_pairwise

different = (0.0, 0.5, 0.5, 0.0)  # P(TT), P(TF), P(FT), P(FF) for a pair
system = PairwiseSystem(
    {"a": 0.5, "b": 0.5, "c": 0.5},
    {("a", "b"): different, ("b", "c"): different, ("a", "c"): different},
)
(report,) = audit_pairwise(system)
print(f"graph is a forest: {report.is_forest}")
print(f"largest profit against a single pair: {report.max_local_profit:.3f}")
print(f"guaranteed profit against the whole system: {report.global_profit:.3f}")
orders = report.global_audit.book.orders(report.global_audit.forecasts, report.global_audit.labels)
for o in orders:
    print(f"  {o.side:<4} {o.label:<8} at {o.price:.2f}, stake {o.stake:.3f}")
