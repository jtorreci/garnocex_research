# Equilibrium versus planner: three simulations for A05

Exploratory analysis (2026-11-23), not yet part of the manuscript. It asks how the fixed-point network of the
paper compares with the system-optimal (planner) network, and whether the fixed point depends on its seed.

| Script | What it does | Output (`outputs/equilibrium_planner/`) |
|---|---|---|
| `fixed_point.py` | Fixed point of the paper's cost-based assignment: Jacobi sweeps from the nearest-plant seed until no municipality changes plant (same rule as `iterative_cost_assignment.py`). | `iter_assign.npy` |
| `fixed_point_seeds.py` | 50 random initial assignments (seed 2026): the same fixed-point rule, and sequential best response in which a municipality moves only if its own cost falls (a Nash equilibrium). | `seeds.npy` |
| `planner_milp.py [K] [time_limit_s]` | Concave-cost location-assignment MILP (single sourcing, all 32 groups as candidates). The fixed cost is underestimated by secants on K log-spaced intervals above T0, so the dual bound is a certified lower bound; the solution is re-evaluated with the exact cost. Default K = 40. | `milp_assign_K40.npy` |
| `compare_methods.py` | Cost, plants, C_max, Gini, t-km, and the number of municipalities that would gain by switching plant. | `method_comparison.csv` |

Run in this order from any directory: `fixed_point.py`, `fixed_point_seeds.py`, `planner_milp.py`, `compare_methods.py`.
Requirements: numpy, pandas, highspy (HiGHS 1.16 was used; the MILP took 496 s and closed the gap to 0).
Inputs are read from `data/tablas/`, as in `iterative_cost_assignment.py`.
