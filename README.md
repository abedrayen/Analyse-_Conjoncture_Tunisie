# Analyse de Conjoncture : Libéralisation Financière et Croissance Économique en Tunisie

## 📋 Description

Ce projet contient une analyse complète de conjoncture (4 pages) sur le thème de la libéralisation financière et de la croissance économique en Tunisie. L'analyse combine :
- Analyse macroéconomique descriptive
- Modélisation économétrique (régression OLS, tests statistiques)
- Recommandations politiques stratégiques

## 📁 Fichiers du Projet

- **`Rapport_Conjoncture_Tunisie.md`** : Rapport complet en format Markdown (4 pages structurées)
- **`liberalisation_analyses.xlsx`** : Dataset avec les données économiques
- **`analyse_complete.py`** : Script Python pour l'analyse complète (descriptive + économétrique)
- **`requirements.txt`** : Dépendances Python nécessaires

## 🚀 Installation et Utilisation

### 1. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 2. Exécuter l'analyse complète avec graphiques

**Option 1 : Script complet avec graphiques intégrés (recommandé)**

```bash
python generer_rapport_complet.py
```

Ce script va :
- Lire les données depuis `liberalisation_analyses.xlsx`
- Effectuer l'analyse descriptive (statistiques, graphiques)
- Estimer le modèle économétrique (OLS)
- Effectuer les tests de diagnostic
- **Générer tous les graphiques dans le dossier `graphiques/`**
- Sauvegarder les résultats économétriques

**Option 2 : Script d'analyse original**

```bash
python analyse_complete.py
```

### 3. Fichiers générés

Après exécution de `generer_rapport_complet.py`, les fichiers suivants seront créés :

**Dans le dossier `graphiques/`** :
- `evolution_indicators.png` : Évolution des 4 indicateurs principaux (PIB, taux d'intérêt, crédit, IDE)
- `matrice_correlation.png` : Matrice de corrélation entre variables
- `evolution_comparative.png` : Évolution comparative normalisée
- `analyse_pib.png` : Analyse détaillée du PIB avec tendance et croissance
- `diagnostics_modele.png` : Graphiques de diagnostic du modèle économétrique
- `pib_observe_vs_predit.png` : Comparaison PIB observé vs prédit

**Fichiers de résultats** :
- `resultats_econometriques.json` : Résultats économétriques au format JSON

**Note** : Le rapport Markdown (`Rapport_Conjoncture_Tunisie.md`) référence automatiquement tous ces graphiques. Assurez-vous d'exécuter le script avant de visualiser le rapport pour que les graphiques soient disponibles.

## 📊 Structure du Rapport

Le rapport `Rapport_Conjoncture_Tunisie.md` est structuré en 4 sections principales :

1. **Introduction** : Contexte macroéconomique, réformes de libéralisation, problématique
2. **Analyse Descriptive et Conjoncturelle** : Indicateurs économiques, tendances, cycles
3. **Modélisation Économétrique** : Modèle OLS, tests de diagnostic, interprétation
4. **Discussion et Recommandations Politiques** : Bilan, recommandations stratégiques, scénarios futurs

## 🔧 Personnalisation

### Ajuster les colonnes du dataset

Si les noms de colonnes dans votre fichier Excel diffèrent, modifiez le dictionnaire `possible_cols` dans `analyse_complete.py` :

```python
possible_cols = {
    'PIB': ['PIB', 'pib', 'GDP', 'gdp', ...],
    'Taux_interet': ['Taux_interet', 'taux_interet', ...],
    # ... etc
}
```

### Modifier le modèle économétrique

Pour changer les variables du modèle, modifiez la section "MODÉLISATION ÉCONOMÉTRIQUE" dans `analyse_complete.py`.

## 📝 Notes

- Le rapport Markdown peut être converti en PDF avec des outils comme Pandoc ou des éditeurs Markdown
- Les graphiques générés sont en haute résolution (300 DPI) pour une utilisation dans des documents
- Les résultats économétriques sont sauvegardés en JSON pour une utilisation ultérieure

## 📧 Support

Pour toute question ou problème, vérifiez que :
1. Toutes les dépendances sont installées
2. Le fichier Excel est dans le même répertoire que les scripts
3. Les noms de colonnes correspondent aux attentes du script

---

**Auteur** : Abed Rayen 
**Date** : 2025

