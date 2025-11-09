"""
Script simplifié pour générer uniquement les graphiques à partir du dataset
"""

import sys
import os

# Vérifier si les dépendances sont installées
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("ERREUR : Dépendances manquantes")
    print(f"Message: {e}")
    print("\nVeuillez installer les dépendances avec:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Importer le script principal
try:
    from generer_rapport_complet import *
    print("Exécution du script de génération des graphiques...")
    print("=" * 80)
except Exception as e:
    print(f"Erreur lors de l'import: {e}")
    print("\nAssurez-vous que le fichier 'generer_rapport_complet.py' existe.")
    sys.exit(1)

