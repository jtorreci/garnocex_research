# unified_analysis.py
# Unified waste management cost analysis
# Approaches: by plant, by municipality, by tonne
# Output: Matrix of plots + violin plots + scatter with log-log regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, t
import os

# Configuration
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Global variables
output_folder = "./figuras_unificadas"
costes_por_planta = None
costes_por_municipio = None
costes_por_tonelada = None

def load_and_prepare_data():
    """Load and prepare all datasets"""
    global costes_por_planta, costes_por_municipio, costes_por_tonelada
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Data paths
    ruta_plantas = "datos_voronoi.csv"
    ruta_municipios = "datos_voronoi_municipios.csv"
    
    # Load data
    df_plantas = pd.read_csv(ruta_plantas)
    df_municipios = pd.read_csv(ruta_municipios)
    
    # Rename columns for clarity
    df_plantas = df_plantas.rename(columns={
        'Coste_Var': 'C_trans', 'Coste_Fijo': 'C_trat', 'Coste_Tot': 'C_tot', 'Produccion': 'Production', 'Habs': 'Population'
    })
    df_municipios = df_municipios.rename(columns={
        'C_Trans': 'C_trans', 'C_Trat': 'C_trat', 'C_Tot': 'C_tot', 'Prod': 'Production', 'HBT': 'Population'
    })
    
    # Clean data
    df_plantas = df_plantas[(df_plantas['C_trans'] >= 0) & (df_plantas['C_trat'] >= 0) & (df_plantas['C_tot'] > 0)]
    df_municipios = df_municipios[(df_municipios['C_trans'] >= 0) & (df_municipios['C_trat'] >= 0) & (df_municipios['C_tot'] > 0)]
    
    print(f"✅ Data loaded: {len(df_plantas)} plants, {len(df_municipios)} municipalities")
    
    # Prepare datasets
    costes_por_planta = df_plantas[['C_trans', 'C_trat', 'C_tot', 'Population']].copy()
    costes_por_municipio = df_municipios[['C_trans', 'C_trat', 'C_tot', 'Population']].copy()
    
    # Analysis by tonne (weighted by production)
    expanded_data = []
    for _, row in df_municipios.iterrows():
        n_samples = int(row['Production'])
        if n_samples > 0:
            expanded_data.extend([
                {'C_trans': row['C_trans'], 'C_trat': row['C_trat'], 'C_tot': row['C_tot'], 'Population': row['Population']}
            ] * n_samples)
    costes_por_tonelada = pd.DataFrame(expanded_data)

def generate_descriptive_statistics():
    """Generate and save descriptive statistics"""
    def get_statistics(series):
        return {
            'mean': series.mean(),
            'mode': series.mode()[0] if not series.mode().empty else np.nan,
            'Q1': series.quantile(0.25),
            'Q3': series.quantile(0.75),
            'median': series.quantile(0.50),
            'variance': series.var(),
            'std_dev': series.std(),
            'min': series.min(),
            'max': series.max()
        }

    statistics_data = []
    for name, df in zip(['By Plant', 'By Municipality', 'By Tonne'], [costes_por_planta, costes_por_municipio, costes_por_tonelada]):
        for cost_type in ['C_trans', 'C_trat', 'C_tot']:
            s = df[cost_type]
            stats = get_statistics(s)
            statistics_data.append({
                'Approach': name,
                'Cost_Component': cost_type,
                'Mean': stats['mean'],
                'Mode': stats['mode'],
                'Q1': stats['Q1'],
                'Q3': stats['Q3'],
                'Median': stats['median'],
                'Variance': stats['variance'],
                'Std_Dev': stats['std_dev'],
                'Min': s.min(),
                'Max': s.max(),
            })

    df_statistics = pd.DataFrame(statistics_data)
    df_statistics.to_csv(f"{output_folder}/descriptive_statistics.csv", index=False)
    print("✅ Descriptive statistics generated and saved")

def generate_matrix_histograms():
    """Generate matrix of histograms + KDE"""
    fig_hist, axes_hist = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    approaches = ['By Plant', 'By Municipality', 'By Tonne']
    cost_components = ['C_trans', 'C_trat', 'C_tot']
    titles = ['Transportation Cost', 'Treatment Cost', 'Total Cost']
    dfs = [costes_por_planta, costes_por_municipio, costes_por_tonelada]
    gray_shades = ['0.8', '0.5', '0.2']

    for i, df in enumerate(dfs):
        for j, (var, title) in enumerate(zip(cost_components, titles)):
            color = gray_shades[i]
            sns.histplot(df[var], ax=axes_hist[i, j], kde=True, color=color, bins=25, alpha=0.8, stat='density',
                         kde_kws={'bw_adjust': 3.0, 'cut': 0})
            mean_val = df[var].mean()
            axes_hist[i, j].axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
            axes_hist[i, j].set_title(f'{approaches[i]} - {title}')
            axes_hist[i, j].set_xlabel('Cost (€/t)')
            axes_hist[i, j].set_ylabel('Density' if j == 0 else '')
            axes_hist[i, j].legend()

    # Last row: KDE comparison
    for idx, (df, label, color) in enumerate(zip(dfs, approaches, gray_shades)):
        for j, var in enumerate(cost_components):
            sns.kdeplot(df[var], ax=axes_hist[3, j], color=color, label=label, linewidth=2.2, bw_adjust=3.0, cut=0)
        axes_hist[3, j].set_title(f'KDE Comparison - {titles[j]}')
        axes_hist[3, j].set_xlabel('Cost (€/t)')
        axes_hist[3, j].set_ylabel('Density')
        axes_hist[3, j].legend()

    fig_hist.suptitle('Comparative Cost Analysis by Approach', fontsize=16, weight='bold')
    plt.savefig(f"{output_folder}/matrix_cost_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Matrix histograms generated and saved")

def generate_violin_plots():
    """Generate violin plots with quartiles, mean, and smoothing"""
    fig_violin, axes_violin = plt.subplots(1, 3, figsize=(15, 6), sharey=True, constrained_layout=True)
    cost_components = ['C_trans', 'C_trat', 'C_tot']
    titles = ['Transportation Cost', 'Treatment Cost', 'Total Cost']
    gray_shades = ['0.8', '0.5', '0.2']
    bw = 3.0  # Control de suavizado

    # Prepare long-format data
    data_list = []
    for i, df in enumerate([costes_por_planta, costes_por_municipio, costes_por_tonelada]):
        for var in cost_components:
            data_list.extend([
                {'Approach': ['By Plant', 'By Municipality', 'By Tonne'][i], 'Component': var, 'Cost (€/t)': val}
                for val in df[var].dropna()
            ])
    df_violin = pd.DataFrame(data_list)
    df_violin['Component'] = df_violin['Component'].replace({
        'C_trans': 'Transportation',
        'C_trat': 'Treatment',
        'C_tot': 'Total'
    })

    for ax, comp in zip(axes_violin, titles):
        comp_name = comp.split()[0].capitalize()
        data_comp = df_violin[df_violin['Component'] == comp_name]

        # Dibujar violinplot sin elementos internos
        sns.violinplot(
            data=data_comp,
            x='Approach',
            y='Cost (€/t)',
            palette=gray_shades,
            bw_adjust=bw,
            inner=None,
            density_norm='width',
            cut=0,  # para que no sobresalga del rango real
            ax=ax,
            alpha=0.8,
            linewidth=1.2
        )

        # Media global
        overall_mean = data_comp['Cost (€/t)'].mean()
        ax.axhline(overall_mean, color='red', linestyle='-', linewidth=3, alpha=0.9, label=f'Overall Mean: {overall_mean:.2f}')

        # Línea vertical y marcas por grupo
        approaches = ['By Plant', 'By Municipality', 'By Tonne']
        for i, approach in enumerate(approaches):
            data_group = data_comp[data_comp['Approach'] == approach]['Cost (€/t)'].dropna()
            if data_group.empty:
                continue

            # Calcular percentiles y media
            q1 = np.percentile(data_group, 25)
            q2 = np.percentile(data_group, 50)
            q3 = np.percentile(data_group, 75)
            mean = data_group.mean()

            # Recortar el rango a percentiles 5 y 95
            y_min = np.percentile(data_group, 5)
            y_max = np.percentile(data_group, 95)

            # Línea vertical central
            ax.vlines(i, ymin=y_min, ymax=y_max, color='blue', linestyle='-', linewidth=1.2)

            # Círculos en Q1, Q2, Q3 (blancos con borde negro)
            for y_val in [q1, q2, q3]:
                ax.scatter(i, y_val, s=60, color='white', edgecolor='black', zorder=3)

            # Círculo rojo para la media del grupo
            ax.scatter(i, mean, s=60, color='red', edgecolor='black', zorder=3)

        # Estética
        ax.set_title(f'{comp}')
        ax.set_xlabel('')
        if ax == axes_violin[0]:
            ax.set_ylabel('Cost (€/t)')
        else:
            ax.set_ylabel('')
        
        # Crear elementos para la leyenda personalizada
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='-', linewidth=3, alpha=0.9, label=f'Overall Mean: {overall_mean:.2f}'),
            Line2D([0], [0], color='red', marker='o', markersize=8, linestyle='None', markerfacecolor='red', markeredgecolor='black', label='Group Mean'),
            Line2D([0], [0], color='white', marker='o', markersize=8, linestyle='None', markerfacecolor='white', markeredgecolor='black', label='Quartiles (Q1, Q2, Q3)'),
            Line2D([0], [0], color='blue', linestyle='-', linewidth=1.2, label='Range (5%-95%)')
        ]
        
        ax.legend(handles=legend_elements)
        ax.grid(True, axis='y', alpha=0.3)

    # Título y guardado
    fig_violin.suptitle('Cost Components by Approach', fontsize=14, weight='bold')
    plt.savefig(f"{output_folder}/violinplots_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Violin plots generated and saved")

def generate_stacked_bars():
    """Generate stacked bars: costs and percentages by plant and municipality"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # --- Config ---
    # Plant bars: gray. Municipality bars: elegant contrasting colors.
    gray_trat_plant = '0.7'
    gray_tot_plant = '0.3'
    gray_trat_muni = '0.75'
    gray_tot_muni = '0.25'
    bar_widths = [0.7, 1.2]  # Plant, Municipality
    edgecolor = 'black'
    linewidth = 0.7

    # --- Data and labels ---
    datasets = [costes_por_planta, costes_por_municipio]
    labels = ['Plants', 'Municipalities']

    for col, (df, label) in enumerate(zip(datasets, labels)):
        df_sorted = df.sort_values('C_tot', ascending=False).reset_index(drop=True)
        x = np.arange(len(df_sorted))
        trat = df_sorted['C_trat']
        tot = df_sorted['C_tot']
        resto = tot - trat
        width = bar_widths[col]

        if col == 0:
            # Plants: original gray and border
            axes[0, col].bar(x, trat, color=gray_trat_plant, width=width, label='Treatment (€/t)', edgecolor=edgecolor, linewidth=linewidth)
            axes[0, col].bar(x, resto, bottom=trat, color=gray_tot_plant, width=width, label='Transport + other (€/t)', edgecolor=edgecolor, linewidth=linewidth)
        else:
            # Municipalities: gray, no border
            axes[0, col].bar(x, trat, color=gray_trat_muni, width=width, label='Treatment (€/t)')
            axes[0, col].bar(x, resto, bottom=trat, color=gray_tot_muni, width=width, label='Transport + other (€/t)')

        axes[0, col].axhline(trat.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean Treatment: {trat.mean():.1f}")
        axes[0, col].axhline(tot.mean(), color='red', linestyle='-', linewidth=2, label=f"Mean Total: {tot.mean():.1f}")
        axes[0, col].set_title(f"Costs per {label} (€/t)")
        axes[0, col].set_xlabel(label)
        axes[0, col].set_ylabel('Cost (€/t)')
        axes[0, col].set_xticks([])
        axes[0, col].legend()
        axes[0, col].grid(axis='y', alpha=0.3)

        # --- Stacked bars: percentages ---
        pct_trat = 100 * trat / tot
        pct_resto = 100 * resto / tot
        if col == 0:
            axes[1, col].bar(x, pct_trat, color=gray_trat_plant, width=width, label='% Treatment', edgecolor=edgecolor, linewidth=linewidth)
            axes[1, col].bar(x, pct_resto, bottom=pct_trat, color=gray_tot_plant, width=width, label='% Transport + other', edgecolor=edgecolor, linewidth=linewidth)
        else:
            axes[1, col].bar(x, pct_trat, color=gray_trat_muni, width=width, label='% Treatment')
            axes[1, col].bar(x, pct_resto, bottom=pct_trat, color=gray_tot_muni, width=width, label='% Transport + other')
        axes[1, col].axhline(pct_trat.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean % Treatment: {pct_trat.mean():.1f}%")
        axes[1, col].axhline(100, color='red', linestyle='-', linewidth=2, label="100% Total")
        axes[1, col].set_title(f"Component percentage per {label}")
        axes[1, col].set_xlabel(label)
        axes[1, col].set_ylabel('Percentage (%)')
        axes[1, col].set_xticks([])
        axes[1, col].set_ylim(0, 110)
        axes[1, col].legend()
        axes[1, col].grid(axis='y', alpha=0.3)

    fig.suptitle('Treatment and Total Costs: Stacked Bars by Plant and Municipality', fontsize=16, weight='bold')
    plt.savefig(f"{output_folder}/barras_apiladas_costes_componentes.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Stacked bars generated and saved")

def show_menu():
    """Show interactive menu for plot selection"""
    print("\n" + "="*60)
    print("UNIFIED COST ANALYSIS - PLOT GENERATOR")
    print("="*60)
    print("1. Load and prepare data")
    print("2. Generate descriptive statistics")
    print("3. Generate matrix of histograms + KDE")
    print("4. Generate violin plots")
    print("5. Generate stacked bar charts")
    print("6. Generate ALL plots")
    print("0. Exit")
    print("-"*60)

if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("Select an option (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            print("Loading and preparing data...")
            load_and_prepare_data()
        elif choice == "2":
            if costes_por_planta is None:
                print("⚠️  Please load data first (option 1)")
                continue
            generate_descriptive_statistics()
        elif choice == "3":
            if costes_por_planta is None:
                print("⚠️  Please load data first (option 1)")
                continue
            generate_matrix_histograms()
        elif choice == "4":
            if costes_por_planta is None:
                print("⚠️  Please load data first (option 1)")
                continue
            generate_violin_plots()
        elif choice == "5":
            if costes_por_planta is None:
                print("⚠️  Please load data first (option 1)")
                continue
            generate_stacked_bars()
        elif choice == "6":
            if costes_por_planta is None:
                print("Loading data first...")
                load_and_prepare_data()
            print("Generating all plots...")
            generate_descriptive_statistics()
            generate_matrix_histograms()
            generate_violin_plots()
            generate_stacked_bars()
            print(f"✅ All figures saved in: {output_folder}/")
        else:
            print("❌ Invalid option. Please select 0-6.")