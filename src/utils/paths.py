# -*- coding: utf-8 -*-
"""
Module pour gérer les chemins de manière centralisée
"""

import os

# Chemins de base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
EXAMPLES_DIR = os.path.join(DATA_DIR, 'examples')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Chemins pour les résultats
VISUALISATION_DIR = os.path.join(RESULTS_DIR, 'visualisation')
RAPPORTS_DIR = os.path.join(RESULTS_DIR, 'rapports')

# Chemins pour les données d'exemple
EXAMPLE_FILES = {
    'ebolowa': os.path.join(EXAMPLES_DIR, 'ebolowa_reseau.inp'),
    'temp': os.path.join(EXAMPLES_DIR, 'temp.inp'),
    'background': os.path.join(EXAMPLES_DIR, 'background.png')
}

# Créer les répertoires s'ils n'existent pas
def ensure_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    directories = [
        DATA_DIR,
        RESULTS_DIR,
        EXAMPLES_DIR,
        VISUALISATION_DIR,
        RAPPORTS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Fonctions utilitaires pour les chemins
def get_visualisation_path(filename):
    """Retourne le chemin complet pour un fichier de visualisation"""
    return os.path.join(VISUALISATION_DIR, filename)

def get_rapport_path(filename):
    """Retourne le chemin complet pour un fichier de rapport"""
    return os.path.join(RAPPORTS_DIR, filename)

def get_log_path(filename):
    """Retourne le chemin complet pour un fichier de log"""
    return os.path.join(LOGS_DIR, filename)

def get_example_path(filename):
    """Retourne le chemin complet pour un fichier d'exemple"""
    return os.path.join(EXAMPLES_DIR, filename) 