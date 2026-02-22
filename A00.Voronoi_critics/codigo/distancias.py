import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator

# Create output folder if it doesn't exist
output_folder = 'A00'
os.makedirs(output_folder, exist_ok=True)

# Helper function to format and save pandas describe() output to a LaTeX table
def save_latex_table(stats_series, filename, caption, label):
    stats_df = stats_series.to_frame().reset_index()
    stats_df.columns = ['Metric', 'Value']
    
    # Rename metrics for clarity in the table
    metric_map = {
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Std. Dev.',
        'min': 'Minimum',
        '25%': '25\% (Q1)',
        '50%': 'Median (50\%)',
        '75%': '75\% (Q3)',
        'max': 'Maximum'
    }
    stats_df['Metric'] = stats_df['Metric'].replace(metric_map)
    
    latex_str = stats_df.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format='lr',
        header=['Statistic', 'Value'],
        position='htbp'
    )
    
    # Save to file
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {output_path}")

# --- Data Loading ---
print("Cargando datos...")
distancias_euclideas = pd.read_csv('distancias_euclideas.csv', sep=';', decimal=',')
distancias_euclideas.rename(columns={'distance_m': 'euclidean_distance'}, inplace=True)
distancias_reales = pd.read_csv('Matriz_Municipios.csv', sep=';', decimal=',', encoding='latin1')

comparativa = distancias_euclideas[['origin_id', 'destination_id', 'euclidean_distance']].copy()
distancias_reales = distancias_reales.rename(columns={distancias_reales.columns[0]: 'origin_id', 'destinatio': 'destination_id'})
comparativa = comparativa.merge(
    distancias_reales[['origin_id', 'destination_id', 'total_cost']], 
    on=['origin_id', 'destination_id'], 
    how='left'
).rename(columns={'total_cost': 'real_distance'})

comparativa['diferencia_porcentual'] = comparativa['real_distance'] / comparativa['euclidean_distance']
print("Datos cargados y procesados.")

# --- Goodness-of-Fit Analysis ---
print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO DE BONDAD DE AJUSTE DE DISTRIBUCIONES")
print("="*80)

ratio_data = comparativa['diferencia_porcentual'].dropna()
ratio_data = ratio_data[np.isfinite(ratio_data) & (ratio_data > 0)]

distributions_to_test = {
    'Log-Normal': {'dist': stats.lognorm, 'name_str': 'lognorm'},
    'Gamma': {'dist': stats.gamma, 'name_str': 'gamma'},
    'Weibull': {'dist': stats.weibull_min, 'name_str': 'weibull_min'}
}

results = []
fitted_params = {}

for name, info in distributions_to_test.items():
    dist_obj = info['dist']
    dist_name_str = info['name_str']
    
    params = dist_obj.fit(ratio_data, floc=0) # floc=0 to fix location at 0
    ks_stat, p_value = stats.kstest(ratio_data, dist_name_str, args=params)
    results.append({'Distribution': name, 'K-S Statistic': ks_stat, 'P-Value': p_value})
    fitted_params[name] = params
    print(f"Ajuste para {name} completado.")

results_df = pd.DataFrame(results)
print("\nResultados de los tests de bondad de ajuste:")
print(results_df.to_string(index=False))

# --- Generate Comparison Plot ---
fig_comp, ax_comp = plt.subplots(figsize=(12, 8))

# Plot histogram of actual data
ax_comp.hist(ratio_data, bins=50, density=True, color='#cccccc', alpha=0.7, label='Empirical Data')

# Plot fitted distributions
x = np.linspace(ratio_data.min(), ratio_data.max(), 1000)
colors = {'Log-Normal': 'red', 'Gamma': 'blue', 'Weibull': 'green'}

for name, params in fitted_params.items():
    dist_obj = distributions_to_test[name]['dist']
    pdf = dist_obj.pdf(x, *params)
    ax_comp.plot(x, pdf, color=colors[name], linestyle='--', linewidth=2, label=f'{name} Fit')

ax_comp.set_title('Comparison of Fitted Distributions to Empirical Data', fontsize=16)
ax_comp.set_xlabel(r'Ratio $d_{r}/d_{e}$', fontsize=12)
ax_comp.set_ylabel('Density', fontsize=12)
ax_comp.legend()
ax_comp.grid(True, alpha=0.3)

comp_plot_path = os.path.join(output_folder, 'distribution_comparison.png')
plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig_comp)
print(f"\nGráfico de comparación de distribuciones guardado en: {comp_plot_path}")