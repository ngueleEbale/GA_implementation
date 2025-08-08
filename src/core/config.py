# -*- coding: utf-8 -*-
"""
Fichier de configuration pour l'optimisation de réseau hydraulique
"""

# ------------------------------------------------------------------------
# PARAMÈTRES HYDRAULIQUES
# ------------------------------------------------------------------------
PRESSION_MIN = 204      # en mCE (~20 bar si 1 bar ~ 10.2 mCE)
PRESSION_MAX = 612      # en mCE (~60 bar)
DIAMETRE_MIN = 40       # mm
DIAMETRE_MAX = 400      # mm
VITESSE_MIN = 0.5       # m/s
VITESSE_MAX = 1.5       # m/s

# ------------------------------------------------------------------------
# PARAMÈTRES D'OPTIMISATION
# ------------------------------------------------------------------------
TAILLE_POPULATION = 100
NOMBRE_GENERATIONS = 100  # Augmenté pour une meilleure convergence
TAUX_CROISEMENT = 0.8
TAUX_MUTATION_INDIVIDU = 0.2  # Probabilité de mutation par individu
TAUX_MUTATION_GENE = 0.1      # Probabilité de mutation par gène

# ------------------------------------------------------------------------
# DIAMÈTRES DISPONIBLES (mm)
# ------------------------------------------------------------------------
DIAMETRES_DISPONIBLES = [40, 63, 75, 90, 110, 160, 200, 250, 315, 400]

# ------------------------------------------------------------------------
# PARAMÈTRES DE LOGGING
# ------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "optimisation.log"

# ------------------------------------------------------------------------
# PARAMÈTRES DE SAUVEGARDE
# ------------------------------------------------------------------------
SAUVEGARDE_PERIODIQUE = True
FREQUENCE_SAUVEGARDE = 10  # Toutes les N générations

# ------------------------------------------------------------------------
# PARAMÈTRES D'ARRÊT
# ------------------------------------------------------------------------
GENERATIONS_SANS_AMELIORATION_MAX = 10

# ------------------------------------------------------------------------
# PARAMÈTRES DE VISUALISATION
# ------------------------------------------------------------------------
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FORMAT = "png" 