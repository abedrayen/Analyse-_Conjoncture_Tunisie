"""
Script pour générer le rapport complet avec graphiques intégrés
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour générer des fichiers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'DejaVu Sans'

# Créer le dossier pour les graphiques
os.makedirs('graphiques', exist_ok=True)

print("=" * 80)
print("GÉNÉRATION DU RAPPORT AVEC GRAPHIQUES")
print("=" * 80)

# Lire les données
print("\n1. Lecture des données...")
df = pd.read_excel('liberalisation_analyses.xlsx')

print(f"   Dimensions: {df.shape}")
print(f"   Colonnes: {df.columns.tolist()[:5]}...")

# Transformer les données de format long (séries par ligne) vers format wide (années par ligne)
if 'Series Name' in df.columns:
    print("\n   Transformation des données...")
    
    # Identifier les colonnes d'années
    year_cols = [col for col in df.columns if '[YR' in col or col.isdigit()]
    
    # Créer un dictionnaire pour mapper les séries
    series_mapping = {
        'PIB': ['GDP growth (annual %)', 'GDP per capita growth (annual %)'],
        'Taux_interet': ['Real interest rate (%)', 'Interest rate spread'],
        'Credit_prive': ['Domestic credit to private sector (% of GDP)'],
        'IDE': ['Foreign direct investment, net inflows (% of GDP)'],
        'Inflation': ['Inflation, consumer prices (annual %)']
    }
    
    # Fonction pour trouver la série correspondante
    def find_series(df, possible_names):
        for idx, row in df.iterrows():
            series_name = str(row.get('Series Name', ''))
            for name in possible_names:
                if name.lower() in series_name.lower():
                    return idx, series_name
        return None, None
    
    # Extraire les données par série
    data_dict = {}
    years = []
    
    # Extraire les années depuis les noms de colonnes
    for col in year_cols:
        try:
            year = int(col.split('[')[0].strip())
            years.append(year)
        except:
            continue
    
    years = sorted(years)
    
    # Extraire PIB
    idx, name = find_series(df, series_mapping['PIB'])
    if idx is not None:
        pib_data = []
        for year in years:
            col_name = f"{year} [YR{year}]"
            if col_name in df.columns:
                val = pd.to_numeric(df.loc[idx, col_name], errors='coerce')
                pib_data.append(val)
            else:
                pib_data.append(np.nan)
        data_dict['PIB'] = pib_data
        print(f"   [OK] PIB trouve: {name}")
    
    # Extraire Taux d'intérêt
    idx, name = find_series(df, series_mapping['Taux_interet'])
    if idx is not None:
        taux_data = []
        for year in years:
            col_name = f"{year} [YR{year}]"
            if col_name in df.columns:
                val = pd.to_numeric(df.loc[idx, col_name], errors='coerce')
                taux_data.append(val)
            else:
                taux_data.append(np.nan)
        data_dict['Taux_interet'] = taux_data
        print(f"   [OK] Taux d'interet trouve: {name}")
    
    # Extraire Crédit privé
    idx, name = find_series(df, series_mapping['Credit_prive'])
    if idx is not None:
        credit_data = []
        for year in years:
            col_name = f"{year} [YR{year}]"
            if col_name in df.columns:
                val = pd.to_numeric(df.loc[idx, col_name], errors='coerce')
                credit_data.append(val)
            else:
                credit_data.append(np.nan)
        data_dict['Credit_prive'] = credit_data
        print(f"   [OK] Credit prive trouve: {name}")
    
    # Extraire IDE
    idx, name = find_series(df, series_mapping['IDE'])
    if idx is not None:
        ide_data = []
        for year in years:
            col_name = f"{year} [YR{year}]"
            if col_name in df.columns:
                val = pd.to_numeric(df.loc[idx, col_name], errors='coerce')
                ide_data.append(val)
            else:
                ide_data.append(np.nan)
        data_dict['IDE'] = ide_data
        print(f"   [OK] IDE trouve: {name}")
    
    # Extraire Inflation
    idx, name = find_series(df, series_mapping['Inflation'])
    if idx is not None:
        inflation_data = []
        for year in years:
            col_name = f"{year} [YR{year}]"
            if col_name in df.columns:
                val = pd.to_numeric(df.loc[idx, col_name], errors='coerce')
                inflation_data.append(val)
            else:
                inflation_data.append(np.nan)
        data_dict['Inflation'] = inflation_data
        print(f"   [OK] Inflation trouvee: {name}")
    
    # Créer le dataframe de travail
    df_work = pd.DataFrame({
        'Annee': years,
        **data_dict
    })
    
    # Supprimer les lignes avec toutes les valeurs NaN
    df_work = df_work.dropna(how='all', subset=[col for col in df_work.columns if col != 'Annee'])
    
    print(f"\n   Données transformées: {df_work.shape[0]} observations")
    print(f"   Période: {df_work['Annee'].min()} - {df_work['Annee'].max()}")
    
else:
    print("   Format de données non reconnu")
    df_work = pd.DataFrame()

# GRAPHIQUE 1: Évolution des indicateurs principaux
print("\n2. Génération des graphiques...")

if 'Annee' in df_work.columns and len([c for c in ['PIB', 'Taux_interet', 'Credit_prive', 'IDE'] if c in df_work.columns]) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Évolution des Indicateurs Macroéconomiques en Tunisie', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_idx = 0
    if 'PIB' in df_work.columns:
        row, col = divmod(plot_idx, 2)
        axes[row, col].plot(df_work['Annee'], df_work['PIB'], marker='o', linewidth=2.5, 
                        markersize=5, color='#2E86AB', label='PIB')
        axes[row, col].set_title('Évolution du PIB (Croissance annuelle %)', fontweight='bold', pad=15)
        axes[row, col].set_xlabel('Année', fontweight='bold')
        axes[row, col].set_ylabel('Croissance PIB (%)', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3, linestyle='--')
        axes[row, col].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[row, col].legend()
        plot_idx += 1
    
    if 'Taux_interet' in df_work.columns:
        row, col = divmod(plot_idx, 2)
        axes[row, col].plot(df_work['Annee'], df_work['Taux_interet'], marker='s', 
                       color='#F24236', linewidth=2.5, markersize=5, label='Taux d\'intérêt')
        axes[row, col].set_title('Évolution du Taux d\'Intérêt Réel', fontweight='bold', pad=15)
        axes[row, col].set_xlabel('Année', fontweight='bold')
        axes[row, col].set_ylabel('Taux d\'Intérêt (%)', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3, linestyle='--')
        axes[row, col].legend()
        plot_idx += 1
    
    if 'Credit_prive' in df_work.columns:
        row, col = divmod(plot_idx, 2)
        axes[row, col].plot(df_work['Annee'], df_work['Credit_prive'], marker='^', 
                       color='#06A77D', linewidth=2.5, markersize=5, label='Crédit privé')
        axes[row, col].set_title('Crédit au Secteur Privé (% du PIB)', fontweight='bold', pad=15)
        axes[row, col].set_xlabel('Année', fontweight='bold')
        axes[row, col].set_ylabel('Crédit Privé (% PIB)', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3, linestyle='--')
        axes[row, col].legend()
        plot_idx += 1
    
    if 'IDE' in df_work.columns:
        row, col = divmod(plot_idx, 2)
        axes[row, col].plot(df_work['Annee'], df_work['IDE'], marker='d', 
                       color='#F18F01', linewidth=2.5, markersize=5, label='IDE')
        axes[row, col].set_title('Investissements Directs Étrangers (% du PIB)', 
                            fontweight='bold', pad=15)
        axes[row, col].set_xlabel('Année', fontweight='bold')
        axes[row, col].set_ylabel('IDE (% PIB)', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3, linestyle='--')
        axes[row, col].legend()
        plot_idx += 1
    
    # Masquer les axes non utilisés
    for i in range(plot_idx, 4):
        row, col = divmod(i, 2)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('graphiques/evolution_indicators.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("   [OK] Graphique 1: evolution_indicators.png")
    plt.close()

# GRAPHIQUE 2: Matrice de corrélation
corr_cols = ['PIB', 'Taux_interet', 'Credit_prive', 'IDE']
available_cols = [col for col in corr_cols if col in df_work.columns]
if len(available_cols) >= 2:
    # Utiliser la méthode pairwise pour gérer les NaN
    corr_matrix = df_work[available_cols].corr()
    
    # Vérifier qu'il y a des corrélations valides
    if not corr_matrix.isna().all().all():
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                   square=True, linewidths=2, cbar_kws={"shrink": 0.8}, mask=mask,
                   vmin=-1, vmax=1, annot_kws={'size': 12, 'weight': 'bold'})
        plt.title('Matrice de Corrélation entre les Variables Clés', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('graphiques/matrice_correlation.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("   [OK] Graphique 2: matrice_correlation.png")
        plt.close()

# GRAPHIQUE 3: Évolution comparative (toutes les variables sur un même graphique)
if 'Annee' in df_work.columns and len([c for c in ['PIB', 'Taux_interet', 'Credit_prive', 'IDE'] if c in df_work.columns]) >= 2:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normaliser les variables pour comparaison
    vars_to_plot = []
    labels = []
    if 'PIB' in df_work.columns:
        vars_to_plot.append(('PIB', '#2E86AB'))
        labels.append('PIB (Croissance %)')
    if 'Credit_prive' in df_work.columns:
        vars_to_plot.append(('Credit_prive', '#06A77D'))
        labels.append('Crédit Privé (% PIB)')
    if 'IDE' in df_work.columns:
        vars_to_plot.append(('IDE', '#F18F01'))
        labels.append('IDE (% PIB)')
    if 'Taux_interet' in df_work.columns:
        vars_to_plot.append(('Taux_interet', '#F24236'))
        labels.append('Taux d\'Intérêt (%)')
    
    for var, color in vars_to_plot:
        # Normalisation min-max pour comparaison
        values = df_work[var].dropna().values
        if len(values) > 0:
            normalized = (values - values.min()) / (values.max() - values.min()) if values.max() != values.min() else values
            years_subset = df_work.loc[df_work[var].notna(), 'Annee'].values
            ax.plot(years_subset, normalized, marker='o', linewidth=2.5, 
                   markersize=4, color=color, label=var, alpha=0.8)
    
    ax.set_title('Évolution Comparative des Indicateurs (Normalisés)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Année', fontweight='bold')
    ax.set_ylabel('Valeur Normalisée (0-1)', fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('graphiques/evolution_comparative.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("   [OK] Graphique 3: evolution_comparative.png")
    plt.close()

# GRAPHIQUE 4: Analyse de la croissance du PIB
if 'Annee' in df_work.columns and 'PIB' in df_work.columns:
    pib_clean = df_work[['Annee', 'PIB']].dropna()
    if len(pib_clean) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Graphique du PIB avec tendance
        ax1.plot(pib_clean['Annee'], pib_clean['PIB'], marker='o', linewidth=2.5, 
                markersize=5, color='#2E86AB', label='PIB observé')
        
        # Tendance linéaire
        x_numeric = np.arange(len(pib_clean))
        z = np.polyfit(x_numeric, pib_clean['PIB'].values, 1)
        p = np.poly1d(z)
        ax1.plot(pib_clean['Annee'], p(x_numeric), '--', linewidth=2, 
                color='#F24236', label='Tendance linéaire', alpha=0.7)
        
        ax1.set_title('Évolution du PIB avec Tendance', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Année', fontweight='bold')
        ax1.set_ylabel('Croissance PIB (%)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Graphique en barres de la croissance
        ax2.bar(pib_clean['Annee'], pib_clean['PIB'].values, color='#06A77D', alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Taux de Croissance Annuel du PIB (%)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Année', fontweight='bold')
        ax2.set_ylabel('Croissance (%)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        plt.savefig('graphiques/analyse_pib.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("   [OK] Graphique 4: analyse_pib.png")
        plt.close()

# MODÉLISATION ÉCONOMÉTRIQUE
print("\n3. Modélisation économétrique...")

# Utiliser les variables disponibles
model_vars = ['PIB', 'Taux_interet', 'Credit_prive', 'IDE']
available_model_vars = [v for v in model_vars if v in df_work.columns]

if 'PIB' in available_model_vars and len(available_model_vars) >= 2:
    # Préparer les variables (supprimer les NaN)
    model_data = df_work[available_model_vars].dropna()
    
    if len(model_data) > len(available_model_vars):  # Au moins n+1 observations pour n variables
        y = model_data['PIB'].values
        X_vars = []
        var_names = []
        # Ajouter toutes les variables sauf PIB
        for var in available_model_vars:
            if var != 'PIB' and var in model_data.columns:
                X_vars.append(model_data[var].values)
                # Noms en français
                name_map = {
                    'Taux_interet': 'Taux_intérêt',
                    'Credit_prive': 'Crédit_privé',
                    'IDE': 'IDE'
                }
                var_names.append(name_map.get(var, var))
        
        if len(X_vars) > 0:
            X = np.column_stack(X_vars)
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # Estimation OLS
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_pred = X_with_const @ beta
            residuals = y - y_pred
            
            # Statistiques
            n = len(y)
            k = X.shape[1] + 1
            ssr = np.sum(residuals**2)
            sst = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ssr / sst) if sst > 0 else 0
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else r_squared
            mse = ssr / (n - k) if n > k else ssr / n
            
            try:
                se_beta = np.sqrt(np.diag(mse * np.linalg.inv(X_with_const.T @ X_with_const)))
                t_stats = beta / se_beta
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
            except:
                se_beta = np.zeros(len(beta))
                t_stats = np.zeros(len(beta))
                p_values = np.ones(len(beta))
            
            print(f"\n   Modèle estimé: PIB = β₀ + β₁(Taux_intérêt) + β₂(Crédit_privé) + β₃(IDE)")
            print(f"   R² = {r_squared:.4f}, R² ajusté = {adj_r_squared:.4f}, n = {n}")
            
            # GRAPHIQUE 5: Diagnostics du modèle
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Diagnostics du Modèle Économétrique', fontsize=16, fontweight='bold')
            
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50, color='#2E86AB')
            axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Valeurs Prédites', fontweight='bold')
            axes[0, 0].set_ylabel('Résidus', fontweight='bold')
            axes[0, 0].set_title('Résidus vs Valeurs Prédites', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(residuals, marker='o', alpha=0.6, color='#F24236')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Observation', fontweight='bold')
            axes[0, 1].set_ylabel('Résidus', fontweight='bold')
            axes[0, 1].set_title('Résidus dans le Temps', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].hist(residuals, bins=min(20, len(residuals)//2), edgecolor='black', 
                          alpha=0.7, color='#06A77D')
            axes[1, 0].set_xlabel('Résidus', fontweight='bold')
            axes[1, 0].set_ylabel('Fréquence', fontweight='bold')
            axes[1, 0].set_title('Distribution des Résidus', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot (Normalité)', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('graphiques/diagnostics_modele.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("   [OK] Graphique 5: diagnostics_modele.png")
            plt.close()
            
            # GRAPHIQUE 6: Comparaison PIB observé vs prédit
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(y, y_pred, alpha=0.6, s=80, color='#2E86AB', edgecolors='black', linewidth=1)
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ligne parfaite')
            ax.set_xlabel('PIB Observé', fontweight='bold', fontsize=12)
            ax.set_ylabel('PIB Prédit', fontweight='bold', fontsize=12)
            ax.set_title(f'PIB Observé vs PIB Prédit (R² = {r_squared:.3f})', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graphiques/pib_observe_vs_predit.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("   [OK] Graphique 6: pib_observe_vs_predit.png")
            plt.close()
            
            # Sauvegarder les résultats
            import json
            results = {
                'R_squared': float(r_squared),
                'Adj_R_squared': float(adj_r_squared),
                'Coefficients': [float(b) for b in beta],
                'Variable_names': ['Constante'] + var_names,
                'Std_Errors': [float(se) for se in se_beta],
                't_stats': [float(t) for t in t_stats],
                'p_values': [float(p) for p in p_values],
                'n_observations': int(n)
            }
            
            with open('resultats_econometriques.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("   [OK] Resultats sauvegardes: resultats_econometriques.json")
    else:
        print("   Pas assez de données pour la modélisation économétrique")
else:
    print("   Variables manquantes pour la modélisation économétrique")

print(f"\n{'='*80}")
print("GÉNÉRATION TERMINÉE")
print(f"{'='*80}")
print("\nGraphiques générés dans le dossier 'graphiques/':")
if os.path.exists('graphiques'):
    for file in os.listdir('graphiques'):
        if file.endswith('.png'):
            print(f"  [OK] {file}")
