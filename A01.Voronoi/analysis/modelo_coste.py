import numpy as np
import matplotlib.pyplot as plt

# Define the piecewise logarithmic fixed cost function
def fixed_cost_total(T, C0, T0):
    """
    Calculate the total fixed cost for a given throughput T.
    C0: Base fixed cost (€/year)
    T0: Scale threshold (t/year)
    T: Annual throughput (t/year)
    """
    return np.where(
        T < T0,
        C0,
        C0 * (np.log2(T / T0) + 1)
    )

# Define the fixed cost per tonne function
def fixed_cost_per_tonne(T, C0, T0):
    """
    Calculate the fixed cost per tonne.
    """
    return fixed_cost_total(T, C0, T0) / T

# --- Parameters ---
C0_base = 40000  # Base fixed cost in €/year
T0_base = 5000   # Scale threshold in t/year

# Range of annual throughput (from 100 to 100,000 t/year)
T_range = np.linspace(100, 100000, 1000)

# --- Calculate Base Curves ---
C_total_base = fixed_cost_total(T_range, C0_base, T0_base)
C_unit_base = fixed_cost_per_tonne(T_range, C0_base, T0_base)

# --- Calculate Sensitivity Curves ---
# Sensitivity to C0 (Base Fixed Cost)
C0_low = 30000
C0_high = 50000
C_total_C0_low = fixed_cost_total(T_range, C0_low, T0_base)
C_total_C0_high = fixed_cost_total(T_range, C0_high, T0_base)
C_unit_C0_low = fixed_cost_per_tonne(T_range, C0_low, T0_base)
C_unit_C0_high = fixed_cost_per_tonne(T_range, C0_high, T0_base)

# Sensitivity to T0 (Scale Threshold)
T0_low = 3000
T0_high = 7000
C_total_T0_low = fixed_cost_total(T_range, C0_base, T0_low)
C_total_T0_high = fixed_cost_total(T_range, C0_base, T0_high)
C_unit_T0_low = fixed_cost_per_tonne(T_range, C0_base, T0_low)
C_unit_T0_high = fixed_cost_per_tonne(T_range, C0_base, T0_high)

# --- Create the Dual Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Left Plot: Total Fixed Cost ---
ax1.plot(T_range, C_total_base, color='blue', linewidth=3, label=f'Base: C₀={C0_base/1e3:.0f}k €/yr, T₀={T0_base/1e3:.0f}k t/yr')
ax1.plot(T_range, C_total_C0_low, color='blue', linewidth=1.5, linestyle='--', label=f'C₀={C0_low/1e3:.0f}k €/yr')
ax1.plot(T_range, C_total_C0_high, color='blue', linewidth=1.5, linestyle='--', label=f'C₀={C0_high/1e3:.0f}k €/yr')
ax1.plot(T_range, C_total_T0_low, color='green', linewidth=1.5, linestyle='-.', label=f'T₀={T0_low/1e3:.0f}k t/yr')
ax1.plot(T_range, C_total_T0_high, color='green', linewidth=1.5, linestyle='-.', label=f'T₀={T0_high/1e3:.0f}k t/yr')

ax1.axvline(T0_base, color='red', linestyle=':', alpha=0.7)
ax1.axhline(C0_base, color='gray', linestyle=':', alpha=0.7)
ax1.text(T0_base, 500, f'T₀ = {T0_base/1e3:.0f}k t/yr', rotation=90, verticalalignment='bottom', color='red', fontsize=10)
ax1.text(1000, C0_base+1000, f'C₀ = {C0_base/1e3:.0f}k €/yr', color='gray', fontsize=10)

ax1.set_xlabel('Annual Throughput, $T_i$ (tonnes/year)')
ax1.set_ylabel('Total Fixed Cost, $C_{\\text{fixed total}}$ (€/year)')
ax1.set_title('Total Fixed Cost vs. Annual Throughput')
ax1.legend(loc='upper left', fontsize='small')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1000, 100000)
ax1.set_ylim(1000, 25000000)

# --- Right Plot: Fixed Cost per Tonne ---
ax2.plot(T_range, C_unit_base, color='green', linewidth=3, label=f'Base: C₀={C0_base/1e3:.0f}k €/yr, T₀={T0_base/1e3:.0f}k t/yr')
ax2.plot(T_range, C_unit_C0_low, color='green', linewidth=1.5, linestyle='--', label=f'C₀={C0_low/1e3:.0f}k €/yr')
ax2.plot(T_range, C_unit_C0_high, color='green', linewidth=1.5, linestyle='--', label=f'C₀={C0_high/1e3:.0f}k €/yr')
ax2.plot(T_range, C_unit_T0_low, color='blue', linewidth=1.5, linestyle='-.', label=f'T₀={T0_low/1e3:.0f}k t/yr')
ax2.plot(T_range, C_unit_T0_high, color='blue', linewidth=1.5, linestyle='-.', label=f'T₀={T0_high/1e3:.0f}k t/yr')

ax2.axvline(T0_base, color='red', linestyle=':', alpha=0.7)
ax2.axhline(C0_base/T0_base, color='gray', linestyle=':', alpha=0.7)
ax2.text(T0_base, 0.1, f'T₀ = {T0_base/1e3:.0f}k t/yr', rotation=90, verticalalignment='bottom', color='red', fontsize=10)
ax2.text(1000, C0_base/T0_base+0.1, f'C₀/T₀ = {C0_base/T0_base:.1f} €/t', color='gray', fontsize=10)

ax2.set_xlabel('Annual Throughput, $T_i$ (tonnes/year)')
ax2.set_ylabel('Treatment Cost per Tonne, $C_{\\text{fixed},i}$ (€/tonne)')
ax2.set_title('Treatment Cost per Tonne vs. Annual Throughput')
ax2.legend(loc='upper right', fontsize='small')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1000, 100000)
ax2.set_ylim(0.1, 20)

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')

# --- Finalize and Display ---
plt.tight_layout()
plt.show()