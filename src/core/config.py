# -*- coding: utf-8 -*-
"""
Configuration Centrale pour l'Optimisation de Réseaux Hydrauliques
================================================================

Ce module centralise tous les paramètres de configuration pour le système d'optimisation
de réseaux hydrauliques utilisant des algorithmes génétiques. Il définit les contraintes
hydrauliques, les paramètres d'optimisation, et les options de visualisation.

La configuration est organisée en sections thématiques pour faciliter la maintenance
et l'ajustement des paramètres selon les besoins spécifiques de chaque projet.

Author: Équipe d'Optimisation Hydraulique
Version: 2.0
Date: 2025
License: MIT

Sections:
---------
- PARAMÈTRES HYDRAULIQUES : Contraintes physiques du réseau
- PARAMÈTRES D'OPTIMISATION : Configuration des algorithmes génétiques
- DIAMÈTRES DISPONIBLES : Catalogue des diamètres commerciaux
- PARAMÈTRES DE LOGGING : Configuration du système de journalisation
- PARAMÈTRES DE SAUVEGARDE : Gestion des checkpoints d'optimisation
- PARAMÈTRES D'ARRÊT : Critères de convergence
- PARAMÈTRES DE VISUALISATION : Configuration des graphiques
"""

# ========================================================================
# PARAMÈTRES HYDRAULIQUES - CONTRAINTES PHYSIQUES DU RÉSEAU
# ========================================================================

# Contraintes de pression (mètres de Colonne d'Eau - mCE)
# Conversion: 1 bar ≈ 10.2 mCE, donc 20-60 bar ≈ 204-612 mCE
PRESSION_MIN = 204      # mCE - Pression minimale pour assurer le service (≈20 bar)
                        # Valeur typique pour réseaux urbains selon normes NF EN 805
PRESSION_MAX = 612      # mCE - Pression maximale pour éviter les surpressions (≈60 bar)
                        # Limite pour préserver l'intégrité des canalisations

# Contraintes de diamètre (millimètres)
DIAMETRE_MIN = 40       # mm - Diamètre minimal pour éviter les obstructions
                        # Valeur minimale selon normes de distribution urbaine
DIAMETRE_MAX = 400      # mm - Diamètre maximal disponible commercialement
                        # Limite pratique pour réseaux de distribution secondaire

# Contraintes de vitesse d'écoulement (mètres par seconde)
# Basées sur les recommandations hydrauliques pour éviter:
# - Vitesses trop faibles: dépôts, stagnation, qualité de l'eau
# - Vitesses trop élevées: érosion, coups de bélier, pertes de charge excessives
VITESSE_MIN = 0.5       # m/s - Vitesse minimale pour maintenir l'auto-curage
                        # Évite la sédimentation et les problèmes de qualité
VITESSE_MAX = 1.5       # m/s - Vitesse maximale pour limiter l'usure et les pertes
                        # Compromis entre efficacité énergétique et durabilité

# ========================================================================
# PARAMÈTRES D'OPTIMISATION - CONFIGURATION DES ALGORITHMES GÉNÉTIQUES
# ========================================================================

# Population et générations
TAILLE_POPULATION = 100     # Nombre d'individus par génération
                           # Compromis entre diversité génétique et temps de calcul
                           # Valeur typique: 50-200 pour problèmes de taille moyenne

NOMBRE_GENERATIONS = 100   # Nombre maximal de générations
                          # Augmenté pour permettre une convergence complète
                          # Peut être réduit si convergence précoce détectée

# Opérateurs génétiques
TAUX_CROISEMENT = 0.8      # Probabilité de croisement entre deux parents (80%)
                          # Valeur standard pour maintenir l'exploration génétique
                          # Plage recommandée: 0.6-0.9

TAUX_MUTATION_INDIVIDU = 0.2  # Probabilité qu'un individu subisse une mutation (20%)
                             # Assure la diversité génétique et évite la convergence prématurée
                             # Valeur adaptée aux problèmes discrets

TAUX_MUTATION_GENE = 0.1     # Probabilité de mutation pour chaque gène/conduite (10%)
                            # Contrôle la granularité des modifications
                            # Plus faible pour préserver les bonnes solutions partielles

# ========================================================================
# DIAMÈTRES DISPONIBLES - CATALOGUE COMMERCIAL NORMALISÉ
# ========================================================================

# Diamètres standards selon normes européennes (EN 545, EN 598)
# Série R10 et R20 pour canalisations en fonte ductile et PVC
# Progression géométrique pour optimiser les choix techniques et économiques
DIAMETRES_DISPONIBLES = [
    40,   # DN40  - Branchements individuels, extrémités de réseau
    63,   # DN63  - Petites dessertes résidentielles
    75,   # DN75  - Dessertes de quartiers résidentiels
    90,   # DN90  - Collecteurs secondaires
    110,  # DN110 - Artères de distribution
    160,  # DN160 - Conduites principales urbaines
    200,  # DN200 - Collecteurs principaux
    250,  # DN250 - Conduites maîtresses
    315,  # DN315 - Artères principales
    400   # DN400 - Conduites de transport/adduction
]
# Note: Diamètres en mm, correspondant aux diamètres nominaux (DN)
# Choix basé sur la disponibilité commerciale et les coûts optimisés

# ========================================================================
# PARAMÈTRES DE LOGGING - SYSTÈME DE JOURNALISATION
# ========================================================================

LOG_LEVEL = "INFO"          # Niveau de détail des logs
                           # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
                           # INFO capture les événements importants sans surcharge

LOG_FILE = "optimisation.log"  # Fichier de destination des logs
                              # Rotation automatique si le fichier devient trop volumineux
                              # Facilite le débogage et le suivi des optimisations

# ========================================================================
# PARAMÈTRES DE SAUVEGARDE - GESTION DES CHECKPOINTS
# ========================================================================

SAUVEGARDE_PERIODIQUE = True   # Active la sauvegarde automatique des états
                              # Permet la reprise après interruption
                              # Essentiel pour les optimisations longues

FREQUENCE_SAUVEGARDE = 10     # Sauvegarde toutes les N générations
                             # Compromis entre sécurité et performance
                             # Peut être ajusté selon la durée d'optimisation

# ========================================================================
# PARAMÈTRES D'ARRÊT - CRITÈRES DE CONVERGENCE
# ========================================================================

GENERATIONS_SANS_AMELIORATION_MAX = 10  # Arrêt si pas d'amélioration pendant N générations
                                       # Détection de convergence pour éviter les calculs inutiles
                                       # Valeur conservative pour s'assurer de la convergence

# ========================================================================
# PARAMÈTRES DE VISUALISATION - CONFIGURATION DES GRAPHIQUES
# ========================================================================

FIGURE_SIZE = (12, 8)      # Taille des figures en pouces (largeur, hauteur)
                          # Format adapté pour les rapports et présentations
                          # Ratio 3:2 pour un affichage optimal

DPI = 300                 # Résolution des images en points par pouce
                         # Qualité publication scientifique
                         # 300 DPI minimum pour impression professionnelle

SAVE_FORMAT = "png"       # Format de sauvegarde des graphiques
                         # PNG offre le meilleur compromis qualité/taille
                         # Transparence supportée, compression sans perte 