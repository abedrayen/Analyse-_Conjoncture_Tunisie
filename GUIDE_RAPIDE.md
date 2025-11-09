# Guide Rapide : Génération des Graphiques

## 🚀 Démarrage Rapide

### Étape 1 : Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 2 : Générer les graphiques
```bash
python generer_rapport_complet.py
```

### Étape 3 : Visualiser le rapport
Ouvrez `Rapport_Conjoncture_Tunisie.md` dans un visualiseur Markdown (VS Code, Typora, etc.)

## 📊 Graphiques Générés

Le script génère automatiquement **6 graphiques** dans le dossier `graphiques/` :

1. **evolution_indicators.png** - Évolution des 4 indicateurs principaux
2. **matrice_correlation.png** - Corrélations entre variables
3. **evolution_comparative.png** - Comparaison normalisée
4. **analyse_pib.png** - Analyse détaillée du PIB
5. **diagnostics_modele.png** - Diagnostics économétriques
6. **pib_observe_vs_predit.png** - Qualité de l'ajustement

## ⚠️ Résolution de Problèmes

### Erreur : "ModuleNotFoundError: No module named 'pandas'"
**Solution** : Installez les dépendances
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels openpyxl
```

### Erreur : "FileNotFoundError: liberalisation_analyses.xlsx"
**Solution** : Vérifiez que le fichier Excel est dans le même dossier que les scripts

### Les graphiques ne s'affichent pas dans le rapport
**Solution** : 
1. Vérifiez que le dossier `graphiques/` existe et contient les fichiers PNG
2. Utilisez un visualiseur Markdown qui supporte les images (VS Code, Typora, etc.)
3. Pour convertir en PDF, utilisez Pandoc ou un outil similaire

## 📝 Structure des Données Excel

Le script recherche automatiquement les colonnes suivantes (noms flexibles) :
- **PIB** : PIB, pib, GDP, gdp, croissance
- **Taux d'intérêt** : Taux_interet, taux_interet, Interest_Rate, taux
- **Crédit privé** : Credit_prive, credit_prive, Private_Credit, credit
- **IDE** : IDE, ide, FDI, fdi, Investissement_direct_etranger
- **Inflation** : Inflation, inflation, CPI, taux_inflation
- **Année** : Annee, annee, Year, year, Date, date

Si vos colonnes ont d'autres noms, modifiez le dictionnaire `possible_cols` dans `generer_rapport_complet.py`.

