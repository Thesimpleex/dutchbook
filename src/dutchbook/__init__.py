"""Dutch-book audit and accuracy-dominance repair for probability forecasts."""

from dutchbook.audit import Audit, Repair, audit, repair
from dutchbook.coherence import (
    Distance,
    DutchBook,
    Order,
    coherence_distance,
    find_dutch_book,
    is_coherent,
)
from dutchbook.conditional import (
    ConditionalBook,
    ConditionalCoherence,
    ConditionalEvent,
    check_conditional_coherence,
    find_conditional_dutch_book,
    parse_conditional,
)
from dutchbook.dominance import (
    DominanceCertificate,
    brier_scores,
    dominance_certificate,
    log_scores,
)
from dutchbook.extension import ExtensionBounds, IncoherentForecastsError, extension_bounds
from dutchbook.logic import (
    Formula,
    FormulaSyntaxError,
    WorldSpace,
    enumerate_worlds,
    parse,
    world_space,
)
from dutchbook.oracle import ColumnGeneration, MilpOracle, find_dutch_book_cg
from dutchbook.pairwise import (
    ComponentReport,
    PairwiseSystem,
    audit_pairwise,
    fit_cells_to_marginals,
    from_tuples,
    marginal_consistent,
)
from dutchbook.projection import (
    EnumeratedVertices,
    Projection,
    euclidean_projection,
    kl_divergence,
    kl_projection,
)

__all__ = [
    "Audit",
    "ColumnGeneration",
    "ComponentReport",
    "ConditionalBook",
    "ConditionalCoherence",
    "ConditionalEvent",
    "Distance",
    "DominanceCertificate",
    "DutchBook",
    "EnumeratedVertices",
    "ExtensionBounds",
    "Formula",
    "FormulaSyntaxError",
    "IncoherentForecastsError",
    "MilpOracle",
    "Order",
    "PairwiseSystem",
    "Projection",
    "Repair",
    "WorldSpace",
    "audit",
    "audit_pairwise",
    "brier_scores",
    "check_conditional_coherence",
    "coherence_distance",
    "dominance_certificate",
    "enumerate_worlds",
    "euclidean_projection",
    "extension_bounds",
    "find_conditional_dutch_book",
    "find_dutch_book",
    "find_dutch_book_cg",
    "fit_cells_to_marginals",
    "from_tuples",
    "is_coherent",
    "kl_divergence",
    "kl_projection",
    "log_scores",
    "marginal_consistent",
    "parse",
    "parse_conditional",
    "repair",
    "world_space",
]

__version__ = "0.1.0"
