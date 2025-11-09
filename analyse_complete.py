"""
Analyse complète de conjoncture : Libéralisation Financière et Croissance Économique en Tunisie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# Lire les données
print("=" * 80)
print("ANALYSE DE CONJONCTURE : LIBÉRALISATION FINANCIÈRE ET CROISSANCE EN TUNISIE")
print("=" * 80)

df = pd.read_excel('liberalisation_analyses.xlsx')

# Nettoyer et préparer les données
print(f"\n1. EXPLORATION DES DONNÉES")
print(f"   Dimensions: {df.shape}")
print(f"   Colonnes: {df.columns.tolist()}")
print(f"\n   Premières observations:")
print(df.head(10))

# Identifier les colonnes clés (ajustez selon vos données)
# Colonnes typiques attendues
possible_cols = {
    'PIB': ['PIB', 'pib', 'GDP', 'gdp', 'PIB_par_habitant', 'PIB_reel'],
    'Taux_interet': ['Taux_interet', 'taux_interet', 'Interest_Rate', 'taux_d_interet'],
    'Credit_prive': ['Credit_prive', 'credit_prive', 'Private_Credit', 'credit_secteur_prive'],
    'IDE': ['IDE', 'ide', 'FDI', 'fdi', 'Investissement_direct_etranger'],
    'Inflation': ['Inflation', 'inflation', 'CPI', 'taux_inflation'],
    'Annee': ['Annee', 'annee', 'Year', 'year', 'Date', 'date']
}

# Fonction pour trouver les colonnes
def find_column(df, possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None

# Identifier les colonnes
col_pib = find_column(df, possible_cols['PIB'])
col_taux = find_column(df, possible_cols['Taux_interet'])
col_credit = find_column(df, possible_cols['Credit_prive'])
col_ide = find_column(df, possible_cols['IDE'])
col_inflation = find_column(df, possible_cols['Inflation'])
col_annee = find_column(df, possible_cols['Annee'])

print(f"\n   Colonnes identifiées:")
print(f"   - PIB: {col_pib}")
print(f"   - Taux d'intérêt: {col_taux}")
print(f"   - Crédit privé: {col_credit}")
print(f"   - IDE: {col_ide}")
print(f"   - Inflation: {col_inflation}")
print(f"   - Année: {col_annee}")

# Préparer les données pour l'analyse
if col_annee:
    if df[col_annee].dtype == 'object':
        df[col_annee] = pd.to_datetime(df[col_annee], errors='coerce')
    df = df.sort_values(col_annee)
    df = df.dropna(subset=[col_annee])

# Créer un dataframe de travail
data = {}
if col_annee:
    data['Annee'] = df[col_annee].values
if col_pib:
    data['PIB'] = pd.to_numeric(df[col_pib], errors='coerce').values
if col_taux:
    data['Taux_interet'] = pd.to_numeric(df[col_taux], errors='coerce').values
if col_credit:
    data['Credit_prive'] = pd.to_numeric(df[col_credit], errors='coerce').values
if col_ide:
    data['IDE'] = pd.to_numeric(df[col_ide], errors='coerce').values
if col_inflation:
    data['Inflation'] = pd.to_numeric(df[col_inflation], errors='coerce').values

df_work = pd.DataFrame(data)
df_work = df_work.dropna()

print(f"\n   Données nettoyées: {df_work.shape[0]} observations")

# 2. ANALYSE DESCRIPTIVE
print(f"\n2. ANALYSE DESCRIPTIVE")

# Statistiques descriptives
print(f"\n   Statistiques descriptives:")
print(df_work.describe())

# Graphiques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Évolution des Indicateurs Macroéconomiques en Tunisie', fontsize=16, fontweight='bold')

if 'Annee' in df_work.columns and 'PIB' in df_work.columns:
    axes[0, 0].plot(df_work['Annee'], df_work['PIB'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Évolution du PIB', fontweight='bold')
    axes[0, 0].set_xlabel('Année')
    axes[0, 0].set_ylabel('PIB')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

if 'Annee' in df_work.columns and 'Taux_interet' in df_work.columns:
    axes[0, 1].plot(df_work['Annee'], df_work['Taux_interet'], marker='s', color='orange', linewidth=2, markersize=4)
    axes[0, 1].set_title('Évolution du Taux d\'Intérêt', fontweight='bold')
    axes[0, 1].set_xlabel('Année')
    axes[0, 1].set_ylabel('Taux d\'Intérêt (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

if 'Annee' in df_work.columns and 'Credit_prive' in df_work.columns:
    axes[1, 0].plot(df_work['Annee'], df_work['Credit_prive'], marker='^', color='green', linewidth=2, markersize=4)
    axes[1, 0].set_title('Évolution du Crédit au Secteur Privé', fontweight='bold')
    axes[1, 0].set_xlabel('Année')
    axes[1, 0].set_ylabel('Crédit Privé')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

if 'Annee' in df_work.columns and 'IDE' in df_work.columns:
    axes[1, 1].plot(df_work['Annee'], df_work['IDE'], marker='d', color='red', linewidth=2, markersize=4)
    axes[1, 1].set_title('Évolution des Investissements Directs Étrangers (IDE)', fontweight='bold')
    axes[1, 1].set_xlabel('Année')
    axes[1, 1].set_ylabel('IDE')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('graphiques_evolution.png', dpi=300, bbox_inches='tight')
print("   Graphiques sauvegardés: graphiques_evolution.png")

# Matrice de corrélation
if all(col in df_work.columns for col in ['PIB', 'Taux_interet', 'Credit_prive', 'IDE']):
    corr_cols = ['PIB', 'Taux_interet', 'Credit_prive', 'IDE']
    corr_matrix = df_work[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation entre les Variables Clés', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('matrice_correlation.png', dpi=300, bbox_inches='tight')
    print("   Matrice de corrélation sauvegardée: matrice_correlation.png")

# 3. MODÉLISATION ÉCONOMÉTRIQUE
print(f"\n3. MODÉLISATION ÉCONOMÉTRIQUE")

if all(col in df_work.columns for col in ['PIB', 'Taux_interet', 'Credit_prive', 'IDE']):
    # Préparer les variables
    y = df_work['PIB'].values
    X = df_work[['Taux_interet', 'Credit_prive', 'IDE']].values
    
    # Ajouter une constante
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Modèle OLS
    print(f"\n   Modèle: PIB = β₀ + β₁(Taux_intérêt) + β₂(Crédit_privé) + β₃(IDE) + ε")
    
    # Estimation OLS
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    
    # Statistiques du modèle
    n = len(y)
    k = X.shape[1] + 1
    ssr = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ssr / sst)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    mse = ssr / (n - k)
    se_beta = np.sqrt(np.diag(mse * np.linalg.inv(X_with_const.T @ X_with_const)))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    
    print(f"\n   Résultats de la régression OLS:")
    print(f"   {'Variable':<20} {'Coefficient':<15} {'Std Error':<15} {'t-stat':<12} {'p-value':<10}")
    print(f"   {'-'*70}")
    print(f"   {'Constante':<20} {beta[0]:<15.4f} {se_beta[0]:<15.4f} {t_stats[0]:<12.4f} {p_values[0]:<10.4f}")
    print(f"   {'Taux_intérêt':<20} {beta[1]:<15.4f} {se_beta[1]:<15.4f} {t_stats[1]:<12.4f} {p_values[1]:<10.4f}")
    print(f"   {'Crédit_privé':<20} {beta[2]:<15.4f} {se_beta[2]:<15.4f} {t_stats[2]:<12.4f} {p_values[2]:<10.4f}")
    print(f"   {'IDE':<20} {beta[3]:<15.4f} {se_beta[3]:<15.4f} {t_stats[3]:<12.4f} {p_values[3]:<10.4f}")
    print(f"\n   R² = {r_squared:.4f}")
    print(f"   R² ajusté = {adj_r_squared:.4f}")
    print(f"   Nombre d'observations = {n}")
    
    # Tests de diagnostic
    print(f"\n   TESTS DE DIAGNOSTIC:")
    
    # Test de stationnarité (Dickey-Fuller)
    print(f"\n   a) Test de stationnarité (Dickey-Fuller augmenté):")
    for var_name, var_data in [('PIB', df_work['PIB']), 
                                ('Taux_intérêt', df_work['Taux_interet']),
                                ('Crédit_privé', df_work['Credit_prive']),
                                ('IDE', df_work['IDE'])]:
        try:
            adf_result = adfuller(var_data.dropna())
            print(f"      {var_name:20s}: ADF = {adf_result[0]:.4f}, p-value = {adf_result[1]:.4f}, "
                  f"{'Stationnaire' if adf_result[1] < 0.05 else 'Non-stationnaire'}")
        except:
            print(f"      {var_name:20s}: Test non disponible")
    
    # Test de multicolinéarité (VIF)
    print(f"\n   b) Test de multicolinéarité (VIF):")
    try:
        vif_data = pd.DataFrame({'Variable': ['Taux_intérêt', 'Crédit_privé', 'IDE']})
        vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_data['VIF'] = vif_values
        for idx, row in vif_data.iterrows():
            status = "Multicolinéarité élevée" if row['VIF'] > 10 else "Acceptable"
            print(f"      {row['Variable']:20s}: VIF = {row['VIF']:.4f} ({status})")
    except:
        print("      Test VIF non disponible")
    
    # Test d'hétéroscédasticité (White)
    print(f"\n   c) Test d'hétéroscédasticité (White):")
    try:
        white_test = het_white(residuals, X_with_const)
        print(f"      Statistique LM = {white_test[0]:.4f}, p-value = {white_test[1]:.4f}")
        print(f"      {'Hétéroscédasticité détectée' if white_test[1] < 0.05 else 'Homoscédasticité acceptée'}")
    except:
        print("      Test de White non disponible")
    
    # Test d'autocorrélation (Durbin-Watson)
    print(f"\n   d) Test d'autocorrélation (Durbin-Watson):")
    try:
        dw_stat = durbin_watson(residuals)
        print(f"      Statistique DW = {dw_stat:.4f}")
        if dw_stat < 1.5:
            print("      Autocorrélation positive détectée")
        elif dw_stat > 2.5:
            print("      Autocorrélation négative détectée")
        else:
            print("      Pas d'autocorrélation significative")
    except:
        print("      Test de Durbin-Watson non disponible")
    
    # Graphique des résidus
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diagnostics du Modèle Économétrique', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Valeurs Prédites')
    axes[0, 0].set_ylabel('Résidus')
    axes[0, 0].set_title('Résidus vs Valeurs Prédites', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(residuals, marker='o', alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Observation')
    axes[0, 1].set_ylabel('Résidus')
    axes[0, 1].set_title('Résidus dans le Temps', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Résidus')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution des Résidus', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normalité)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostics_modele.png', dpi=300, bbox_inches='tight')
    print("\n   Graphiques de diagnostic sauvegardés: diagnostics_modele.png")
    
    # Sauvegarder les résultats
    results_summary = {
        'R_squared': r_squared,
        'Adj_R_squared': adj_r_squared,
        'Coefficients': beta.tolist(),
        'Std_Errors': se_beta.tolist(),
        't_stats': t_stats.tolist(),
        'p_values': p_values.tolist()
    }
    
    import json
    with open('resultats_econometriques.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("   Résultats économétriques sauvegardés: resultats_econometriques.json")

else:
    print("   Variables manquantes pour la modélisation économétrique")

print(f"\n{'='*80}")
print("ANALYSE TERMINÉE")
print(f"{'='*80}")
print("\nFichiers générés:")
print("  - graphiques_evolution.png")
print("  - matrice_correlation.png")
print("  - diagnostics_modele.png")
print("  - resultats_econometriques.json")

