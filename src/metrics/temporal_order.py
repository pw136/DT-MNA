
from ..theory.commutator import reordering_error_and_bound

def order_error_metrics(initial_distribution, kernels, permutation):
    return reordering_error_and_bound(initial_distribution, kernels, permutation)

