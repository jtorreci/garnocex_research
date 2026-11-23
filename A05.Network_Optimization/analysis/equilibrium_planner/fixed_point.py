"""Fixed point of the paper's cost-based assignment (Algorithm 1), same rule as ../iterative_cost_assignment.py:
Jacobi sweeps from the nearest-plant seed until no municipality changes plant."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from a05data import *

def fixed_point(A_seed, P, D, max_iter=200):
    """Return (assignment, sweeps, converged)."""
    m = D.shape[1]; A = A_seed.copy()
    for it in range(1, max_iter + 1):
        T = np.bincount(A, weights=P, minlength=m)
        An = np.argmin(RHO * D + unit_cost(np.maximum(T, 1))[None, :], axis=1)
        if np.array_equal(An, A): return A, it, True
        A = An
    return A, max_iter, False

if __name__ == "__main__":
    names, P, plants, D = load()
    print("municipalities", len(P), "plants", len(plants), "total t", round(P.sum()))
    A0 = np.argmin(D, axis=1)
    print("seed (nearest): %.4f EUR/t, %d plants" % system_cost(A0, P, D)[1:3])
    A, sweeps, ok = fixed_point(A0, P, D)
    tot, cpt, n, _ = system_cost(A, P, D)
    print(f"fixed point: {cpt:.4f} EUR/t, plants {n}, total {tot:,.0f}, sweeps {sweeps}, converged {ok}")
    np.save(OUTPUT_DIR + "/iter_assign.npy", A)
