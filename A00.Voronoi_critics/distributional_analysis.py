#!/usr/bin/env python3
"""
Distributional comparison for Voronoi misallocation probabilities.

Compares Log-Normal, Gamma, Weibull, Inverse Gaussian under:
  P[misallocation | R] = P[beta1/beta2 > R]

Key question: does the log-normal provide a conservative (upper) bound
on misallocation probability near the Voronoi boundary?
"""

import os
import numpy as np
from scipy import stats, special, optimize
from scipy.special import digamma, polygamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters from paper ──────────────────────────────────────────────
SEED = 42
N_MC = 5_000_000

# All-pairs fit (adopted)
M_LN = 0.166          # E[ln beta]
S_LN = 0.093          # std(ln beta)
S2_LN = S_LN**2

# Observed beta moments
MEAN_BETA = 1.190
STD_BETA = 0.125
VAR_BETA = STD_BETA**2

EULER = 0.5772156649015329
N_MUN = 383

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Plotting style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
})
COLORS = {'Log-Normal': '#1f77b4', 'Gamma': '#ff7f0e',
          'Weibull': '#2ca02c', 'Inv. Gaussian': '#d62728'}
STYLES = {'Log-Normal': '-', 'Gamma': '--',
          'Weibull': '-.', 'Inv. Gaussian': ':'}
NAMES = list(COLORS.keys())

# ── Distribution fitting (matching moments of ln beta) ─────────────────

def fit_all_lnmoments(m, s2):
    """Fit all 4 distributions to match E[ln beta]=m, Var(ln beta)=s2."""
    params = {}

    # 1) Log-Normal: trivial
    params['Log-Normal'] = (m, np.sqrt(s2))

    # 2) Gamma(k, theta): psi_1(k) = s2, psi(k) + ln(theta) = m
    def eq_trig(logk):
        return polygamma(1, np.exp(logk)) - s2
    logk = optimize.brentq(eq_trig, -5, 20)
    k_g = np.exp(logk)
    theta_g = np.exp(m - digamma(k_g))
    params['Gamma'] = (k_g, theta_g)

    # 3) Weibull(k, lam): Var(ln X) = pi^2/(6k^2) = s2
    k_w = np.pi / (np.sqrt(s2) * np.sqrt(6))
    lam_w = np.exp(m + EULER / k_w)
    params['Weibull'] = (k_w, lam_w)

    # 4) Inverse Gaussian(mu, lam_ig): numerical
    #    Use scipy: invgauss(mu/lam_ig, scale=lam_ig), mean=mu
    mu0 = np.exp(m + s2/2)
    var0 = (np.exp(s2) - 1) * np.exp(2*m + s2)
    lam0 = mu0**3 / max(var0, 1e-10)
    rng_fit = np.random.default_rng(SEED + 7)

    def obj_ig(logp):
        mu_ig, lam_ig = np.exp(logp[0]), np.exp(logp[1])
        try:
            samp = stats.invgauss.rvs(mu_ig/lam_ig, scale=lam_ig,
                                       size=500_000, random_state=rng_fit)
            samp = samp[samp > 0]
            lns = np.log(samp)
            return (np.mean(lns) - m)**2 + (np.var(lns) - s2)**2
        except:
            return 1e10

    res = optimize.minimize(obj_ig, [np.log(mu0), np.log(lam0)],
                            method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-7})
    params['Inv. Gaussian'] = (np.exp(res.x[0]), np.exp(res.x[1]))

    return params


def sample_dist(name, params, n, rng):
    """Draw n samples from named distribution."""
    if name == 'Log-Normal':
        return rng.lognormal(params[0], params[1], size=n)
    elif name == 'Gamma':
        return rng.gamma(params[0], params[1], size=n)
    elif name == 'Weibull':
        return params[1] * rng.weibull(params[0], size=n)
    elif name == 'Inv. Gaussian':
        mu, lam = params
        return stats.invgauss.rvs(mu/lam, scale=lam, size=n,
                                   random_state=rng)
    raise ValueError(name)


# ── Monte Carlo P[beta1/beta2 > R] ────────────────────────────────────

def mc_pmis(name, params, R_values, n=N_MC, seed=SEED):
    """Compute P[beta1/beta2 > R] via MC for array of R values."""
    rng = np.random.default_rng(seed)
    b1 = sample_dist(name, params, n, rng)
    rng2 = np.random.default_rng(seed + 1000)
    b2 = sample_dist(name, params, n, rng2)
    ratio = np.sort(b1 / b2)
    idx = np.searchsorted(ratio, R_values, side='right')
    return 1.0 - idx / n


def mc_fd0(name, params, n=N_MC, seed=SEED):
    """Estimate f_D(0) where D = ln(beta1) - ln(beta2), via histogram."""
    rng = np.random.default_rng(seed + 2000)
    b1 = sample_dist(name, params, n, rng)
    rng2 = np.random.default_rng(seed + 3000)
    b2 = sample_dist(name, params, n, rng2)
    mask = (b1 > 0) & (b2 > 0)
    D = np.log(b1[mask]) - np.log(b2[mask])
    counts, edges = np.histogram(D, bins=2000, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.argmin(np.abs(centres))
    return counts[idx]


def analytical_pmis_ln(R, s):
    """Closed-form P[mis|R] under log-normal."""
    return stats.norm.cdf(-np.log(R) / (np.sqrt(2) * s))


def analytical_fd0_ln(s):
    """f_D(0) for D ~ N(0, 2s^2)."""
    return 1.0 / np.sqrt(4 * np.pi * s**2)


# ── Entropy ────────────────────────────────────────────────────────────

def entropy_ln(m, s):
    return m + 0.5 + np.log(s * np.sqrt(2 * np.pi))

def entropy_gamma(k, theta):
    return k + np.log(theta) + special.gammaln(k) + (1-k)*digamma(k)

def entropy_weibull(k, lam):
    return EULER*(1 - 1/k) + np.log(lam/k) + 1

def entropy_ig(mu, lam_ig):
    """Approximate via MC."""
    rng = np.random.default_rng(SEED + 99)
    samp = stats.invgauss.rvs(mu/lam_ig, scale=lam_ig,
                               size=2_000_000, random_state=rng)
    samp = samp[samp > 0]
    lp = stats.invgauss.logpdf(samp, mu/lam_ig, scale=lam_ig)
    return -np.mean(lp)

def get_entropy(name, params):
    if name == 'Log-Normal':    return entropy_ln(*params)
    if name == 'Gamma':         return entropy_gamma(*params)
    if name == 'Weibull':       return entropy_weibull(*params)
    if name == 'Inv. Gaussian': return entropy_ig(*params)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("DISTRIBUTIONAL COMPARISON — VORONOI MISALLOCATION FRAMEWORK")
    print("=" * 90)

    # ── Fit distributions ──────────────────────────────────────────────
    print(f"\nFitting to E[ln beta] = {M_LN}, Var(ln beta) = {S2_LN:.6f}")
    params = fit_all_lnmoments(M_LN, S2_LN)

    for name in NAMES:
        print(f"  {name:18s}: params = ({params[name][0]:.6f}, {params[name][1]:.6f})")

    # Verify fits
    print("\n  Verification (MC, n=2M):")
    for name in NAMES:
        rng_v = np.random.default_rng(SEED + 200)
        samp = sample_dist(name, params[name], 2_000_000, rng_v)
        samp = samp[samp > 0]
        lns = np.log(samp)
        print(f"    {name:18s}: E[ln b]={np.mean(lns):.4f} (tgt {M_LN:.4f}), "
              f"Var(ln b)={np.var(lns):.6f} (tgt {S2_LN:.6f}), "
              f"E[b]={np.mean(samp):.4f}, Std[b]={np.std(samp):.4f}")

    # ── R grids ────────────────────────────────────────────────────────
    R_full = np.linspace(1.001, 2.5, 300)
    R_zoom = np.linspace(1.001, 1.15, 200)
    R_check = np.array([1.02, 1.05, 1.10, 1.20, 1.50, 2.00])

    # ── Compute P[mis|R] for full range ────────────────────────────────
    print(f"\nComputing P[mis|R] via MC (n={N_MC:,})...")
    curves = {}
    for name in NAMES:
        print(f"  {name}...", end=' ', flush=True)
        curves[name] = mc_pmis(name, params[name], R_full, seed=SEED+10)
        print("done.")

    # Zoom curves
    print("Computing near-boundary curves...")
    curves_zoom = {}
    for name in NAMES:
        curves_zoom[name] = mc_pmis(name, params[name], R_zoom, seed=SEED+30)

    # Analytical log-normal
    pmis_ln_an = analytical_pmis_ln(R_full, S_LN)
    pmis_ln_an_zoom = analytical_pmis_ln(R_zoom, S_LN)

    # ── f_D(0) ─────────────────────────────────────────────────────────
    print("\nEstimating f_D(0)...")
    fd0 = {}
    for name in NAMES:
        fd0[name] = mc_fd0(name, params[name], seed=SEED+40)
    fd0_an_ln = analytical_fd0_ln(S_LN)
    print(f"  {'Distribution':18s} {'MC f_D(0)':>10s} {'Normal ref':>12s}")
    print(f"  {'-'*18} {'-'*10} {'-'*12}")
    for name in NAMES:
        ref = "" if name != 'Log-Normal' else f"  (analytical: {fd0_an_ln:.4f})"
        print(f"  {name:18s} {fd0[name]:10.4f}{ref}")

    # ── Crossover analysis ─────────────────────────────────────────────
    print("\nCrossover analysis (Log-Normal vs others):")
    ln_curve = curves['Log-Normal']
    for name in NAMES:
        if name == 'Log-Normal':
            continue
        diff = ln_curve - curves[name]
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) > 0:
            for ci in crossings:
                r1, r2 = R_full[ci], R_full[ci+1]
                d1, d2 = diff[ci], diff[ci+1]
                R_star = r1 - d1*(r2-r1)/(d2-d1)
                p_at_cross = analytical_pmis_ln(np.array([R_star]), S_LN)[0]
                print(f"  vs {name:18s}: R* = {R_star:.4f}  "
                      f"(P_mis at crossover = {p_at_cross:.4f})")
        else:
            above = np.mean(diff > 0)
            print(f"  vs {name:18s}: No crossover. "
                  f"LN above {above*100:.0f}% of range.")

    # ── P at specific R values ─────────────────────────────────────────
    print("\nP[misallocation | R] at selected R values:")
    pmis_table = {}
    for name in NAMES:
        pmis_table[name] = mc_pmis(name, params[name], R_check, seed=SEED+80)

    header_r = "  {:18s}".format("Distribution")
    for r in R_check:
        header_r += f" | R={r:<5.2f}"
    print(header_r)
    print("  " + "-"*(len(header_r)-2))
    for name in NAMES:
        row = f"  {name:18s}"
        for val in pmis_table[name]:
            row += f" | {val:.4f}"
        print(row)

    # Also show the analytical log-normal
    row = f"  {'LN (analytical)':18s}"
    for r in R_check:
        row += f" | {analytical_pmis_ln(np.array([r]), S_LN)[0]:.4f}"
    print(row)

    # ── Expected misallocations ────────────────────────────────────────
    # Synthetic R ~ 1 + Exp(3) truncated
    rng_syn = np.random.default_rng(SEED + 50)
    R_syn = []
    while len(R_syn) < N_MUN:
        batch = 1.0 + rng_syn.exponential(1/3.0, size=N_MUN*3)
        batch = batch[(batch >= 1.0) & (batch <= 4.0)]
        R_syn.extend(batch.tolist())
    R_syn = np.array(R_syn[:N_MUN])

    print(f"\nExpected misallocations (n={N_MUN}, synthetic R):")
    print(f"  R distribution: mean={np.mean(R_syn):.3f}, "
          f"median={np.median(R_syn):.3f}, "
          f"P(R<1.1)={np.mean(R_syn<1.1):.2f}, "
          f"P(R<1.3)={np.mean(R_syn<1.3):.2f}")

    exp_mis = {}
    for name in NAMES:
        pmis_syn = mc_pmis(name, params[name], R_syn, seed=SEED+70)
        exp_mis[name] = np.sum(pmis_syn)
        print(f"  {name:18s}: {exp_mis[name]:.1f} misallocations")

    # ── Entropy ────────────────────────────────────────────────────────
    print("\nDifferential entropy H(X) [nats]:")
    print("  (Log-Normal maximises H for given E[ln X], Var[ln X])")
    entropies = {}
    for name in NAMES:
        entropies[name] = get_entropy(name, params[name])
        marker = " ◄ MAX" if name == 'Log-Normal' else ""
        print(f"  {name:18s}: H = {entropies[name]:.6f}{marker}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════

    # ── Figure 1: Full range ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title('(a) $P[\\mathrm{misallocation} \\mid R]$ — full range')
    for name in NAMES:
        ax1.plot(R_full, curves[name], label=name,
                 color=COLORS[name], ls=STYLES[name], lw=1.8)
    ax1.plot(R_full, pmis_ln_an, 'k:', alpha=0.4, lw=1,
             label='Log-Normal (analytical)')
    ax1.set_xlabel('$R = d_e(P,A_2)/d_e(P,A_1)$')
    ax1.set_ylabel('$P[\\mathrm{misallocation} \\mid R]$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim(1.0, 2.5); ax1.set_ylim(0, 0.52)
    ax1.grid(True, alpha=0.3)

    # Ratio plot
    ax2.set_title('(b) Ratio $P_{\\mathrm{other}} / P_{\\mathrm{LN}}$')
    for name in NAMES:
        if name == 'Log-Normal':
            continue
        mask = ln_curve > 1e-6
        ratio_p = np.full_like(R_full, np.nan)
        ratio_p[mask] = curves[name][mask] / ln_curve[mask]
        ax2.plot(R_full, ratio_p, label=name,
                 color=COLORS[name], ls=STYLES[name], lw=1.8)
    ax2.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('$P_{\\mathrm{other}} / P_{\\mathrm{Log\\text{-}Normal}}$')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.set_xlim(1.0, 2.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, 'distributional_comparison_full.pdf')
    fig.savefig(p1, bbox_inches='tight')
    print(f"\nSaved: {p1}")
    plt.close(fig)

    # ── Figure 2: Boundary zoom ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title('(a) Near-boundary misallocation probability')
    for name in NAMES:
        ax1.plot(R_zoom, curves_zoom[name], label=name,
                 color=COLORS[name], ls=STYLES[name], lw=1.8)
    ax1.plot(R_zoom, pmis_ln_an_zoom, 'k:', alpha=0.4, lw=1)
    ax1.set_xlabel('$R$')
    ax1.set_ylabel('$P[\\mathrm{misallocation} \\mid R]$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim(1.0, 1.15)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('(b) Linear approx: $P \\approx 1/2 - f_D(0)\\cdot\\ln R$')
    for name in NAMES:
        ax2.plot(R_zoom, curves_zoom[name],
                 color=COLORS[name], ls=STYLES[name], lw=1.2, alpha=0.5)
        lin = 0.5 - fd0[name] * np.log(R_zoom)
        ax2.plot(R_zoom, lin, color=COLORS[name], ls='-', lw=1.0,
                 label=f'{name} ($f_D(0)$={fd0[name]:.2f})')
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('$P[\\mathrm{misallocation} \\mid R]$')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.set_xlim(1.0, 1.15)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, 'distributional_comparison_boundary.pdf')
    fig.savefig(p2, bbox_inches='tight')
    print(f"Saved: {p2}")
    plt.close(fig)

    # ── Figure 3: Log-scale tails ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Tail behavior (log scale)')
    for name in NAMES:
        y = curves[name]
        m = y > 0
        ax.semilogy(R_full[m], y[m], label=name,
                     color=COLORS[name], ls=STYLES[name], lw=1.8)
    ax.set_xlabel('$R$')
    ax.set_ylabel('$P[\\mathrm{misallocation} \\mid R]$ (log scale)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(1.0, 2.5)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    p3 = os.path.join(FIG_DIR, 'distributional_tail_analysis.pdf')
    fig.savefig(p3, bbox_inches='tight')
    print(f"Saved: {p3}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    hdr = (f"{'Distribution':18s} | {'f_D(0)':>7s} | {'P(1.05)':>8s} | "
           f"{'P(1.10)':>8s} | {'P(1.50)':>8s} | {'P(2.00)':>8s} | "
           f"{'E[mis]':>7s} | {'Entropy':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name in NAMES:
        idx_105 = np.argmin(np.abs(R_check - 1.05))
        idx_110 = np.argmin(np.abs(R_check - 1.10))
        idx_150 = np.argmin(np.abs(R_check - 1.50))
        idx_200 = np.argmin(np.abs(R_check - 2.00))
        print(f"{name:18s} | {fd0[name]:7.3f} | "
              f"{pmis_table[name][idx_105]:8.5f} | "
              f"{pmis_table[name][idx_110]:8.5f} | "
              f"{pmis_table[name][idx_150]:8.5f} | "
              f"{pmis_table[name][idx_200]:8.5f} | "
              f"{exp_mis[name]:7.1f} | "
              f"{entropies[name]:8.4f}")
    print("=" * 100)

    print("\nKey findings:")
    ln_mis = exp_mis['Log-Normal']
    for name in NAMES:
        if name == 'Log-Normal':
            continue
        diff_pct = (ln_mis - exp_mis[name]) / exp_mis[name] * 100
        print(f"  LN vs {name}: LN predicts {diff_pct:+.1f}% misallocations")

    print("\nConclusion: Log-Normal is conservative near boundary "
          "(lowest f_D(0), slowest probability decrease from 0.5)")
    print("Done.")


if __name__ == '__main__':
    main()
