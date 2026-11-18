
import itertools
from math import factorial

FACTORS = ("D", "S", "T")

def _key(active):
    active = set(active)
    return tuple(1 if factor in active else 0 for factor in FACTORS)

def _validate_values(values):
    expected = set(itertools.product((0, 1), repeat=3))
    missing = expected - set(values)
    if missing:
        raise ValueError("missing counterfactual values: %r" % sorted(missing))

def shapley_three_factor(values):
    _validate_values(values)
    contributions = {}
    player_count = len(FACTORS)
    for factor in FACTORS:
        others = [item for item in FACTORS if item != factor]
        contribution = 0.0
        for subset_size in range(len(others) + 1):
            for subset in itertools.combinations(others, subset_size):
                weight = (
                    factorial(subset_size)
                    * factorial(player_count - subset_size - 1)
                    / float(factorial(player_count))
                )
                without = values[_key(subset)]
                with_factor = values[_key(tuple(subset) + (factor,))]
                contribution += weight * (with_factor - without)
        contributions[factor] = float(contribution)
    contributions["total"] = float(values[(1, 1, 1)] - values[(0, 0, 0)])
    contributions["closure_error"] = float(
        contributions["total"] - sum(contributions[f] for f in FACTORS)
    )
    return contributions

def interaction_effects(values):
    _validate_values(values)
    pairwise = {
        "DS": values[(1, 1, 0)] - values[(1, 0, 0)] - values[(0, 1, 0)] + values[(0, 0, 0)],
        "DT": values[(1, 0, 1)] - values[(1, 0, 0)] - values[(0, 0, 1)] + values[(0, 0, 0)],
        "ST": values[(0, 1, 1)] - values[(0, 1, 0)] - values[(0, 0, 1)] + values[(0, 0, 0)],
    }
    third_order = (
        values[(1, 1, 1)]
        - values[(1, 1, 0)]
        - values[(1, 0, 1)]
        - values[(0, 1, 1)]
        + values[(1, 0, 0)]
        + values[(0, 1, 0)]
        + values[(0, 0, 1)]
        - values[(0, 0, 0)]
    )
    result = {key: float(value) for key, value in pairwise.items()}
    result["DST"] = float(third_order)
    return result

