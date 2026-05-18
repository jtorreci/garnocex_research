#!/usr/bin/env python3
"""
Sensitivity of distributional comparison to the dispersion parameter s.
Shows how the four distributions diverge (or not) as s increases.
"""

import os
import numpy as np
from scipy import stats, special, optimize
from scipy.special import digamma, polygamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
N_MC = 3_000_000
EULER = 0.5772156649015329
M_LN = 0.166   # fixed E[ln beta]

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

COLORS = {'Log-Normal': '#1f77b4', 'Gamma': '#ff7f0e',
          'Weibull': '#2ca02c', 'Inv. Gaussian': '#d62728'}
STYLES = {'Log-Normal': '-', 'Gamma': '--',
          'Weibull': '-.', 'Inv. Gaussian': ':'}
NAMES = list(COLORS.keys())

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
})


def fit_all(m, s2):
    params = {}
    s = np.sqrt(s2)
    params['Log-Normal'] = (m, s)

    # Gamma
    def eq(logk):
        return polygamma(1, np.exp(logk)) - s2
    try:
        logk = optimize.brentq(eq, -5, 20)
        k_g = np.exp(logk)
        theta_g = np.exp(m - digamma(k_g))
        params['Gamma'] = (k_g, theta_g)
    except:
        params['Gamma'] = None

    # Weibull
    k_w = np.pi / (s * np.sqrt(6))
    lam_w = np.exp(m + EULER / k_w)
    params['Weibull'] = (k_w, lam_w)

    # Inv Gaussian
    mu0 = np.exp(m + s2/2)
    var0 = (np.exp(s2) - 1) * np.exp(2*m + s2)
    lam0 = mu0**3 / max(var0, 1e-10)
    rng_fit = np.random.default_rng(SEED + 7)
    def obj_ig(logp):
        mu_ig, lam_ig = np.exp(logp[0]), np.exp(logp[1])
        try:
            samp = stats.invgauss.rvs(mu_ig/lam_ig, scale=lam_ig,
                                       size=300_000, random_state=rng_fit)
            samp = samp[samp > 0]
            lns = np.log(samp)
            return (np.mean(lns) - m)**2 + (np.var(lns) - s2)**2
        except:
            return 1e10
    res = optimize.minimize(obj_ig, [np.log(mu0), np.log(lam0)],
                            method='Nelder-Mead', options={'maxiter': 3000})
    params['Inv. Gaussian'] = (np.exp(res.x[0]), np.exp(res.x[1]))

    return params


def sample_dist(name, params, n, rng):
    if name == 'Log-Normal':
        return rng.lognormal(params[0], params[1], size=n)
    elif name == 'Gamma':
        return rng.gamma(params[0], params[1], size=n)
    elif name == 'Weibull':
        return params[1] * rng.weibull(params[0], size=n)
    elif name == 'Inv. Gaussian':
        mu, lam = params
        return stats.invgauss.rvs(mu/lam, scale=lam, size=n, random_state=rng)


def mc_pmis(name, params, R_values, n=N_MC, seed=SEED):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    b1 = sample_dist(name, params, n, rng1)
    b2 = sample_dist(name, params, n, rng2)
    ratio = np.sort(b1 / b2)
    idx = np.searchsorted(ratio, R_values, side='right')
    return 1.0 - idx / n


def mc_fd0(name, params, n=N_MC, seed=SEED):
    rng1 = np.random.default_rng(seed + 2000)
    rng2 = np.random.default_rng(seed + 3000)
    b1 = sample_dist(name, params, n, rng1)
    b2 = sample_dist(name, params, n, rng2)
    mask = (b1 > 0) & (b2 > 0)
    D = np.log(b1[mask]) - np.log(b2[mask])
    counts, edges = np.histogram(D, bins=2000, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts[np.argmin(np.abs(centres))]


def main():
    # Sweep over s values
    s_values = [0.05, 0.093, 0.15, 0.20, 0.30]
    R_full = np.linspace(1.001, 3.0, 400)
    R_check = np.array([1.05, 1.10, 1.20, 1.50])

    # Generate synthetic R for expected misallocations
    rng_syn = np.random.default_rng(SEED + 50)
    R_syn = []
    while len(R_syn) < 383:
        batch = 1.0 + rng_syn.exponential(1/3.0, size=1200)
        batch = batch[(batch >= 1.0) & (batch <= 4.0)]
        R_syn.extend(batch.tolist())
    R_syn = np.array(R_syn[:383])

    all_results = {}

    for s in s_values:
        s2 = s**2
        print(f"\n{'='*70}")
        print(f"s = {s:.3f} (Gamma shape k ~ {1/s2:.1f})")
        print(f"{'='*70}")

        params = fit_all(M_LN, s2)

        results = {}
        for name in NAMES:
            if params[name] is None:
                results[name] = {'fd0': np.nan, 'pmis': [np.nan]*4, 'emis': np.nan}
                continue
            print(f"  {name}...", end=' ', flush=True)
            fd0 = mc_fd0(name, params[name], seed=SEED+40)
            pmis = mc_pmis(name, params[name], R_check, seed=SEED+80)
            pmis_syn = mc_pmis(name, params[name], R_syn, seed=SEED+70)
            emis = np.sum(pmis_syn)
            results[name] = {'fd0': fd0, 'pmis': pmis, 'emis': emis}
            print(f"f_D(0)={fd0:.3f}, E[mis]={emis:.1f}")

        all_results[s] = results

        # Print table
        print(f"\n  {'Distribution':18s} | {'f_D(0)':>7s} | {'P(1.05)':>8s} | "
              f"{'P(1.10)':>8s} | {'P(1.20)':>8s} | {'P(1.50)':>8s} | {'E[mis]':>7s}")
        print(f"  {'-'*80}")
        for name in NAMES:
            r = results[name]
            print(f"  {name:18s} | {r['fd0']:7.3f} | "
                  f"{r['pmis'][0]:8.4f} | {r['pmis'][1]:8.4f} | "
                  f"{r['pmis'][2]:8.4f} | {r['pmis'][3]:8.4f} | {r['emis']:7.1f}")

    # ── Figure: E[mis] vs s, per distribution ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Expected misallocations vs s
    ax = axes[0]
    ax.set_title('(a) Expected misallocations vs $s$')
    for name in NAMES:
        y = [all_results[s][name]['emis'] for s in s_values]
        ax.plot(s_values, y, 'o-', label=name,
                color=COLORS[name], ls=STYLES[name], lw=1.8, ms=5)
    ax.set_xlabel('Dispersion parameter $s$')
    ax.set_ylabel('Expected misallocations ($n=383$)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel (b): f_D(0) vs s
    ax = axes[1]
    ax.set_title('(b) Density $f_D(0)$ vs $s$')
    for name in NAMES:
        y = [all_results[s][name]['fd0'] for s in s_values]
        ax.plot(s_values, y, 'o-', label=name,
                color=COLORS[name], ls=STYLES[name], lw=1.8, ms=5)
    ax.set_xlabel('Dispersion parameter $s$')
    ax.set_ylabel('$f_D(0)$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel (c): Relative difference in E[mis] vs LN
    ax = axes[2]
    ax.set_title('(c) Relative excess of Log-Normal vs others')
    for name in NAMES:
        if name == 'Log-Normal':
            continue
        ln_y = np.array([all_results[s]['Log-Normal']['emis'] for s in s_values])
        ot_y = np.array([all_results[s][name]['emis'] for s in s_values])
        rel = (ln_y - ot_y) / ot_y * 100
        ax.plot(s_values, rel, 'o-', label=name,
                color=COLORS[name], ls=STYLES[name], lw=1.8, ms=5)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Dispersion parameter $s$')
    ax.set_ylabel('$(E_{LN} - E_{other})/E_{other}$ [%]')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'distributional_sensitivity_s_sweep.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY: LN conservatism across s values")
    print("=" * 70)
    for s in s_values:
        ln_emis = all_results[s]['Log-Normal']['emis']
        others_max = max(all_results[s][n]['emis'] for n in NAMES if n != 'Log-Normal')
        others_min = min(all_results[s][n]['emis'] for n in NAMES if n != 'Log-Normal')
        pct_vs_min = (ln_emis - others_min) / others_min * 100
        pct_vs_max = (ln_emis - others_max) / others_max * 100
        print(f"  s={s:.3f}: LN={ln_emis:.1f}, "
              f"range others=[{others_min:.1f}, {others_max:.1f}], "
              f"LN excess vs min: {pct_vs_min:+.1f}%, vs max: {pct_vs_max:+.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
