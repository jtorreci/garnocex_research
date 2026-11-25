# -*- coding: utf-8 -*-
"""
optimize_metaheuristics.py
==========================
Compares the simple greedy pruning heuristic of A05 against metaheuristics:
1. Simulated Annealing (SA) for each network size N from 3 to 32.
2. Genetic Algorithm (GA) for unconstrained optimization of active plants.

Generates a comparison CSV and plots the cost curves to discuss robustness.
"""

import argparse
import math
import os
import random
import re
import sys
import numpy as np
import pandas as pd

# ============================================================
# COST MODEL (same as A01/A05)
# ============================================================
C0 = 40_000
T0 = 5_000
V_TREAT = 0.35
RHO_TRANS = 0.35
DOTACION = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description="A05 metaheuristics optimization")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser.parse_args()

# In scripts, args might be parsed. Let's make it safe for direct imports too.
if __name__ == "__main__":
    ARGS = parse_args()
else:
    class DummyArgs:
        no_plot = True
    ARGS = DummyArgs()

def treatment_cost_per_tonne(T):
    if T <= 0:
        return 0.0
    if T < T0:
        cfix = C0
    else:
        cfix = C0 * (math.log2(T / T0) + 1)
    return cfix / T + V_TREAT

# ============================================================
# LOAD DATA
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Input tables (shared with A01) live in this paper's own data/tablas folder.
TABLAS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "tablas")

# Full distance matrix
asig_real = pd.read_csv(os.path.join(TABLAS_DIR, "asignacion_municipios_real.csv"))
df_dist = pd.read_csv(os.path.join(TABLAS_DIR, "distancias_reales_plantas_municipios.csv"))
df_dist = df_dist.rename(columns={"origin_id": "plant_id", "destination_id": "municipio", "total_cost": "distance"})

# Population data
df_mun_data = pd.read_csv(os.path.join(TABLAS_DIR, "datos_voronoi_municipios.csv"), encoding="utf-8")

def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()

pop_lookup = {}
for _, row in df_mun_data.iterrows():
    key = norm(row.get("NOMBRE", ""))
    if key and not pd.isna(row.get("HBT")):
        pop_lookup[key] = int(row["HBT"])

municipalities = []
for _, row in asig_real.iterrows():
    mun = str(row["municipio"]).strip()
    key = norm(mun)
    pop = pop_lookup.get(key, 0)
    if pop > 0:
        municipalities.append({"municipio": mun, "key": key, "poblacion": pop, "produccion": pop * DOTACION})

df_mun = pd.DataFrame(municipalities)

dist_lookup = {}
for _, row in df_dist.iterrows():
    pid = row["plant_id"]
    mun_key = norm(str(row["municipio"]))
    dist_km = float(row["distance"]) / 1000.0
    if mun_key and dist_km > 0:
        dist_lookup[(pid, mun_key)] = dist_km

all_plants_raw = sorted(set(pid for pid, _ in dist_lookup.keys()))

CONSOLIDATION_15KM = {
    2: 1, 12: 1,
    17: 3,
    15: 6,
    40: 19,
    32: 20,
    23: 22, 26: 22, 28: 22,
    34: 27, 35: 27, 37: 27,
    38: 36, 43: 36,
}

dist_lookup_consolidated = {}
for (pid, mkey), dist in dist_lookup.items():
    cpid = CONSOLIDATION_15KM.get(pid, pid)
    existing = dist_lookup_consolidated.get((cpid, mkey))
    if existing is None or dist < existing:
        dist_lookup_consolidated[(cpid, mkey)] = dist

all_plants = sorted(set(cpid for cpid, _ in dist_lookup_consolidated.keys()))
dist_lookup = dist_lookup_consolidated

# Pre-convert DataFrame to a list of dicts to avoid iterrows() overhead
mun_records = df_mun.to_dict("records")

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_network_cost(active_plants):
    """Returns the total system cost in EUR/t for a given set of active plants."""
    if len(active_plants) == 0:
        return float("inf")
    
    # Pre-build lookup for speed in optimization loops
    active_list = list(active_plants)
    
    # We assign each municipality to the nearest active plant
    # For speed, pre-aggregate production and distance
    tot_cost = 0.0
    plant_productions = {pid: 0.0 for pid in active_list}
    plant_tkms = {pid: 0.0 for pid in active_list}
    
    for mun in mun_records:
        key = mun["key"]
        prod = mun["produccion"]
        
        best_dist = float("inf")
        best_plant = None
        for pid in active_list:
            d = dist_lookup.get((pid, key))
            if d is not None and d < best_dist:
                best_dist = d
                best_plant = pid
        
        if best_plant is not None:
            plant_productions[best_plant] += prod
            plant_tkms[best_plant] += prod * best_dist
            
    # Compute total cost
    for pid in active_list:
        prod = plant_productions[pid]
        if prod > 0:
            C_trans = plant_tkms[pid] * RHO_TRANS
            C_treat = prod * treatment_cost_per_tonne(prod)
            tot_cost += C_trans + C_treat
            
    total_prod = df_mun["produccion"].sum()
    return tot_cost / total_prod if total_prod > 0 else float("inf")

# ============================================================
# SIMULATED ANNEALING (SA) FOR FIXED SIZE N
# ============================================================
def run_simulated_annealing(N, max_iter=2000):
    """Finds the best set of N active plants using Simulated Annealing."""
    if N == len(all_plants):
        return set(all_plants), evaluate_network_cost(set(all_plants))
    
    # Initial state: random N plants
    current_state = set(random.sample(all_plants, N))
    current_cost = evaluate_network_cost(current_state)
    
    best_state = set(current_state)
    best_cost = current_cost
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.99
    
    for i in range(max_iter):
        # Generate neighbor by swapping one active with one inactive
        active_list = list(current_state)
        inactive_list = list(set(all_plants) - current_state)
        
        u = random.choice(active_list)
        v = random.choice(inactive_list)
        
        neighbor = (current_state - {u}) | {v}
        neighbor_cost = evaluate_network_cost(neighbor)
        
        delta = neighbor_cost - current_cost
        
        # Acceptance probability
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_state = set(current_state)
                best_cost = current_cost
                
        T *= cooling_rate
        if T < 1e-4:
            T = 1e-4
            
    return best_state, best_cost

# ============================================================
# GENETIC ALGORITHM (GA) FOR UNCONSTRAINED/CONSTRAINED OPTIMIZATION
# ============================================================
def run_genetic_algorithm(pop_size=50, generations=100, constrained_N=None):
    """
    Genetic Algorithm to find the optimal active plants.
    If constrained_N is set, chromosomes are repaired to have exactly N active plants.
    """
    num_genes = len(all_plants)
    
    def create_chromosome():
        if constrained_N is not None:
            indices = random.sample(range(num_genes), constrained_N)
            chromo = [0] * num_genes
            for idx in indices:
                chromo[idx] = 1
            return chromo
        else:
            # Unconstrained: random bits
            return [random.choice([0, 1]) for _ in range(num_genes)]
            
    # Initialize population
    population = [create_chromosome() for _ in range(pop_size)]
    
    def chromo_to_set(chromo):
        return set(all_plants[i] for i, val in enumerate(chromo) if val == 1)
        
    def evaluate_chromosome(chromo):
        active = chromo_to_set(chromo)
        return evaluate_network_cost(active)
        
    best_chromo = None
    best_cost = float("inf")
    
    for gen in range(generations):
        # Evaluate fitness (lower cost is better)
        scores = []
        for chromo in population:
            cost = evaluate_chromosome(chromo)
            scores.append((cost, chromo))
            
        scores.sort(key=lambda x: x[0])
        
        if scores[0][0] < best_cost:
            best_cost = scores[0][0]
            best_chromo = list(scores[0][1])
            
        # Selection: Tournament selection
        selected = []
        for _ in range(pop_size):
            # Tournament of size 3
            t1, t2, t3 = random.sample(scores, 3)
            winner = min([t1, t2, t3], key=lambda x: x[0])[1]
            selected.append(list(winner))
            
        # Crossover
        next_gen = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % pop_size]
            
            if random.random() < 0.8:
                # Uniform crossover
                child1 = []
                child2 = []
                for g in range(num_genes):
                    if random.random() < 0.5:
                        child1.append(parent1[g])
                        child2.append(parent2[g])
                    else:
                        child1.append(parent2[g])
                        child2.append(parent1[g])
            else:
                child1 = list(parent1)
                child2 = list(parent2)
                
            next_gen.extend([child1, child2])
            
        # Mutation
        for chromo in next_gen:
            if random.random() < 0.2:
                # Flip 1 or 2 bits
                for _ in range(random.choice([1, 2])):
                    idx = random.randint(0, num_genes - 1)
                    chromo[idx] = 1 - chromo[idx]
            
            # Repair if constrained
            if constrained_N is not None:
                ones = [i for i, val in enumerate(chromo) if val == 1]
                if len(ones) < constrained_N:
                    needed = constrained_N - len(ones)
                    zero_indices = [i for i, val in enumerate(chromo) if val == 0]
                    for idx in random.sample(zero_indices, needed):
                        chromo[idx] = 1
                elif len(ones) > constrained_N:
                    excess = len(ones) - constrained_N
                    for idx in random.sample(ones, excess):
                        chromo[idx] = 0
                        
        population = next_gen
        
    return chromo_to_set(best_chromo), best_cost

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("METAHEURISTICS OPTIMIZATION AND BENCHMARKING (A05)")
    print("=" * 70)
    
    # 1. Load greedy pruning results
    outputs_dir = os.path.join(SCRIPT_DIR, "..", "outputs")
    greedy_csv = os.path.join(outputs_dir, "pareto_podado.csv")
    if os.path.exists(greedy_csv):
        df_greedy = pd.read_csv(greedy_csv)
        print(f"Loaded greedy pruning curve from {greedy_csv}")
    else:
        print("Greedy CSV not found! Run analisis_podado.py first.")
        sys.exit(1)
        
    # We will compute SA for each N from 3 to 32
    results = []
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("\nRunning Simulated Annealing for N = 3 to 32...")
    for N in range(3, 33):
        # We find greedy cost
        greedy_row = df_greedy[df_greedy["n_plants"] == N]
        greedy_cost = greedy_row["cost_per_tonne"].values[0] if len(greedy_row) > 0 else float("inf")
        
        # Run SA
        sa_active, sa_cost = run_simulated_annealing(N, max_iter=3000)
        
        # Run GA (constrained)
        ga_active, ga_cost = run_genetic_algorithm(pop_size=60, generations=100, constrained_N=N)
        
        # Best metaheuristic cost
        meta_cost = min(sa_cost, ga_cost)
        best_method = "SA" if sa_cost <= ga_cost else "GA"
        best_active = sa_active if sa_cost <= ga_cost else ga_active
        
        gap = ((greedy_cost - meta_cost) / meta_cost) * 100.0 if meta_cost > 0 else 0.0
        
        results.append({
            "n_plants": N,
            "greedy_cost_eur_t": greedy_cost,
            "sa_cost_eur_t": sa_cost,
            "ga_cost_eur_t": ga_cost,
            "best_meta_cost_eur_t": meta_cost,
            "best_method": best_method,
            "gap_percentage": gap,
            "active_plants": ",".join(map(str, sorted(best_active)))
        })
        
        print(f"  N={N:2d}: Greedy={greedy_cost:.4f} | SA={sa_cost:.4f} | GA={ga_cost:.4f} | Best Meta={meta_cost:.4f} (Gap={gap:+.3f}%)")
        
    df_comp = pd.DataFrame(results)
    comp_csv = os.path.join(outputs_dir, "comparison_metaheuristics.csv")
    df_comp.to_csv(comp_csv, index=False, float_format="%.4f")
    print(f"\nSaved comparison table to {comp_csv}")
    
    # 2. Run unconstrained GA (N can vary freely)
    print("\nRunning Unconstrained Genetic Algorithm (N is free to vary)...")
    unconstrained_active, unconstrained_cost = run_genetic_algorithm(pop_size=100, generations=200, constrained_N=None)
    print(f"  Unconstrained GA Optimum: {len(unconstrained_active)} plants active, Cost = {unconstrained_cost:.4f} EUR/t")
    print(f"  Active plants: {sorted(unconstrained_active)}")
    
    unc_csv = os.path.join(outputs_dir, "unconstrained_ga.csv")
    pd.DataFrame([{
        "n_plants": len(unconstrained_active),
        "cost_eur_t": unconstrained_cost,
        "active_plants": ",".join(map(str, sorted(unconstrained_active))),
    }]).to_csv(unc_csv, index=False, float_format="%.4f")
    print(f"Saved unconstrained GA result to {unc_csv}")

    # Plot comparison (shared with the plot-only script plot_metaheuristics.py)
    if not ARGS.no_plot:
        try:
            from plot_metaheuristics import plot_comparison
            plot_comparison(df_comp, len(unconstrained_active), unconstrained_cost)
        except ImportError:
            print("matplotlib not available, skipping plot")

    print("\nMetaheuristics execution complete.")
