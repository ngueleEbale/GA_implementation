# -*- coding: utf-8 -*-
"""
Formules Hydrauliques et Constantes pour l'Optimisation de Réseaux
==================================================================

Ce module documente toutes les formules hydrauliques utilisées dans le système
d'optimisation. Il sert de référence technique pour comprendre les calculs
effectués et valider les résultats obtenus.

Les formules sont implémentées dans WNTR/EPANET mais documentées ici pour
faciliter la compréhension et la maintenance du code d'optimisation.

Author: Équipe d'Optimisation Hydraulique
Version: 3.0
Date: 2025
License: MIT
"""

import math

# ========================================================================
# CONSTANTES PHYSIQUES HYDRAULIQUES
# ========================================================================

# Constantes universelles
GRAVITY = 9.81  # Accélération gravitationnelle (m/s²)
WATER_DENSITY = 1000  # Densité de l'eau à 20°C (kg/m³)
KINEMATIC_VISCOSITY = 1.004e-6  # Viscosité cinématique de l'eau à 20°C (m²/s)

# Facteurs de conversion
BAR_TO_MCE = 10.2  # Conversion bar vers mètres de Colonne d'Eau
MCE_TO_BAR = 1/10.2  # Conversion inverse
MM_TO_M = 0.001  # Millimètres vers mètres
M_TO_MM = 1000  # Mètres vers millimètres

# ========================================================================
# FORMULES DE PERTES DE CHARGE
# ========================================================================

def hazen_williams_headloss(flow_rate, diameter, length, roughness_coeff=130):
    """
    Calcule les pertes de charge selon l'équation de Hazen-Williams.
    
    Cette formule empirique est largement utilisée pour les réseaux d'eau
    potable. Elle est particulièrement adaptée aux conduites en fonte,
    acier et PVC avec des vitesses modérées.
    
    Formule de Hazen-Williams:
    -------------------------
    ΔH = 10.67 × L × (Q^1.852) / (C^1.852 × D^4.87)
    
    Où:
    - ΔH = Perte de charge (m)
    - L = Longueur de la conduite (m)
    - Q = Débit volumique (m³/s)
    - C = Coefficient de rugosité de Hazen-Williams (sans dimension)
    - D = Diamètre intérieur de la conduite (m)
    
    Coefficients de rugosité typiques:
    - Fonte ductile neuve: C = 130-140
    - Acier neuf: C = 120-130
    - PVC/PE: C = 130-150
    - Fonte ancienne: C = 80-100
    
    Domaine de validité:
    - Vitesses: 0.3 à 3 m/s
    - Diamètres: > 50 mm
    - Eau à température ambiante
    
    Parameters:
    -----------
    flow_rate : float
        Débit volumique (m³/s)
    diameter : float
        Diamètre intérieur (m)
    length : float
        Longueur de la conduite (m)
    roughness_coeff : float, optional
        Coefficient C de Hazen-Williams (défaut: 130)
    
    Returns:
    --------
    float
        Perte de charge (m)
    
    Examples:
    ---------
    >>> # Conduite DN200, L=1000m, Q=0.1 m³/s
    >>> perte = hazen_williams_headloss(0.1, 0.2, 1000, 130)
    >>> print(f"Perte de charge: {perte:.2f} m")
    """
    if diameter <= 0 or flow_rate <= 0:
        return 0
    
    # Application de la formule de Hazen-Williams
    headloss = 10.67 * length * (abs(flow_rate) ** 1.852) / \
               (roughness_coeff ** 1.852 * diameter ** 4.87)
    
    return headloss

def darcy_weisbach_headloss(flow_rate, diameter, length, roughness=0.0015):
    """
    Calcule les pertes de charge selon l'équation de Darcy-Weisbach.
    
    Cette formule théorique est basée sur la mécanique des fluides et
    applicable à tous types d'écoulements. Elle nécessite le calcul
    du facteur de friction qui dépend du nombre de Reynolds.
    
    Formule de Darcy-Weisbach:
    -------------------------
    ΔH = f × (L/D) × (V²/2g)
    
    Où:
    - ΔH = Perte de charge (m)
    - f = Facteur de friction de Darcy (sans dimension)
    - L = Longueur de la conduite (m)
    - D = Diamètre intérieur (m)
    - V = Vitesse moyenne (m/s)
    - g = Accélération gravitationnelle (9.81 m/s²)
    
    Le facteur de friction f dépend du nombre de Reynolds (Re) et
    de la rugosité relative (ε/D):
    
    - Écoulement laminaire (Re < 2300): f = 64/Re
    - Écoulement turbulent: Équation de Colebrook-White
    
    Rugosités absolues typiques (ε):
    - Acier neuf: 0.045 mm
    - Fonte ductile: 0.25 mm
    - PVC: 0.0015 mm
    - Béton lisse: 0.3 mm
    
    Parameters:
    -----------
    flow_rate : float
        Débit volumique (m³/s)
    diameter : float
        Diamètre intérieur (m)
    length : float
        Longueur de la conduite (m)
    roughness : float, optional
        Rugosité absolue ε (m) (défaut: 0.0015 mm pour PVC)
    
    Returns:
    --------
    float
        Perte de charge (m)
    """
    if diameter <= 0 or flow_rate <= 0:
        return 0
    
    # Calcul de la vitesse moyenne
    area = math.pi * (diameter / 2) ** 2
    velocity = abs(flow_rate) / area
    
    # Calcul du nombre de Reynolds
    reynolds = velocity * diameter / KINEMATIC_VISCOSITY
    
    # Calcul du facteur de friction
    if reynolds < 2300:  # Écoulement laminaire
        friction_factor = 64 / reynolds
    else:  # Écoulement turbulent - Approximation de Swamee-Jain
        relative_roughness = roughness / diameter
        friction_factor = 0.25 / (math.log10(relative_roughness/3.7 + 5.74/reynolds**0.9))**2
    
    # Application de la formule de Darcy-Weisbach
    headloss = friction_factor * (length / diameter) * (velocity ** 2) / (2 * GRAVITY)
    
    return headloss

# ========================================================================
# CALCULS DE VITESSE ET DÉBIT
# ========================================================================

def calculate_velocity(flow_rate, diameter):
    """
    Calcule la vitesse d'écoulement dans une conduite circulaire.
    
    Formule fondamentale:
    --------------------
    V = Q / A = Q / (π × D²/4) = 4Q / (π × D²)
    
    Où:
    - V = Vitesse moyenne (m/s)
    - Q = Débit volumique (m³/s)
    - A = Section de passage (m²)
    - D = Diamètre intérieur (m)
    
    Vitesses recommandées pour réseaux d'eau potable:
    - Minimum: 0.5 m/s (éviter stagnation et dépôts)
    - Optimum: 0.8-1.2 m/s (bon compromis énergie/usure)
    - Maximum: 1.5-2.0 m/s (limiter érosion et coups de bélier)
    
    Parameters:
    -----------
    flow_rate : float
        Débit volumique (m³/s)
    diameter : float
        Diamètre intérieur (m)
    
    Returns:
    --------
    float
        Vitesse d'écoulement (m/s)
    """
    if diameter <= 0:
        return 0
    
    area = math.pi * (diameter / 2) ** 2
    velocity = abs(flow_rate) / area
    
    return velocity

def calculate_flow_rate(velocity, diameter):
    """
    Calcule le débit à partir de la vitesse et du diamètre.
    
    Formule:
    -------
    Q = V × A = V × π × D²/4
    
    Parameters:
    -----------
    velocity : float
        Vitesse d'écoulement (m/s)
    diameter : float
        Diamètre intérieur (m)
    
    Returns:
    --------
    float
        Débit volumique (m³/s)
    """
    area = math.pi * (diameter / 2) ** 2
    flow_rate = velocity * area
    
    return flow_rate

# ========================================================================
# CALCULS DE PRESSION ET ÉNERGIE
# ========================================================================

def pressure_to_head(pressure_bar):
    """
    Convertit une pression en bar vers une hauteur en mCE.
    
    Formule de conversion:
    ---------------------
    H = P × 10.2
    
    Où:
    - H = Hauteur d'eau (mCE)
    - P = Pression (bar)
    - 10.2 = Facteur de conversion (mCE/bar)
    
    Cette conversion est basée sur la relation:
    P = ρ × g × h / 10⁵
    
    Parameters:
    -----------
    pressure_bar : float
        Pression (bar)
    
    Returns:
    --------
    float
        Hauteur équivalente (mCE)
    """
    return pressure_bar * BAR_TO_MCE

def head_to_pressure(head_mce):
    """
    Convertit une hauteur en mCE vers une pression en bar.
    
    Parameters:
    -----------
    head_mce : float
        Hauteur d'eau (mCE)
    
    Returns:
    --------
    float
        Pression équivalente (bar)
    """
    return head_mce * MCE_TO_BAR

# ========================================================================
# FONCTIONS DE VALIDATION HYDRAULIQUE
# ========================================================================

def validate_pressure_range(pressure_mce, min_pressure=20, max_pressure=60):
    """
    Valide qu'une pression est dans la plage acceptable.
    
    Critères de validation:
    ----------------------
    - Pression minimale: Assurer le service aux usagers
    - Pression maximale: Préserver l'intégrité des canalisations
    
    Standards typiques:
    - Résidentiel: 15-40 mCE (1.5-4 bar)
    - Commercial: 20-50 mCE (2-5 bar)
    - Industriel: 30-60 mCE (3-6 bar)
    
    Parameters:
    -----------
    pressure_mce : float
        Pression à valider (mCE)
    min_pressure : float, optional
        Pression minimale acceptable (mCE)
    max_pressure : float, optional
        Pression maximale acceptable (mCE)
    
    Returns:
    --------
    bool
        True si la pression est acceptable
    """
    return min_pressure <= pressure_mce <= max_pressure

def validate_velocity_range(velocity, min_velocity=0.5, max_velocity=1.5):
    """
    Valide qu'une vitesse est dans la plage acceptable.
    
    Critères de validation:
    ----------------------
    - Vitesse minimale: Éviter stagnation et développement bactérien
    - Vitesse maximale: Limiter érosion et pertes de charge
    
    Parameters:
    -----------
    velocity : float
        Vitesse à valider (m/s)
    min_velocity : float, optional
        Vitesse minimale acceptable (m/s)
    max_velocity : float, optional
        Vitesse maximale acceptable (m/s)
    
    Returns:
    --------
    bool
        True si la vitesse est acceptable
    """
    return min_velocity <= abs(velocity) <= max_velocity

# ========================================================================
# NOTES TECHNIQUES ET RÉFÉRENCES
# ========================================================================

"""
RÉFÉRENCES NORMATIVES:
---------------------
- NF EN 805: Alimentation en eau - Exigences pour les réseaux extérieurs
- NF EN 14801: Réseaux d'évacuation et d'assainissement à l'extérieur
- Fascicule 71: Fourniture et pose de canalisations d'eau potable
- DTU 60.11: Réseaux d'eau chaude et froide sanitaires

LIMITES D'APPLICATION:
---------------------
- Température: 5-40°C (eau potable standard)
- Pression: 0.5-16 bar (réseaux de distribution)
- Diamètre: DN40-DN400 (distribution secondaire)
- Fluide: Eau potable (viscosité et densité standard)

PRÉCISION DES CALCULS:
---------------------
- Hazen-Williams: ±5% pour vitesses 0.3-3 m/s
- Darcy-Weisbach: ±2% (plus précis mais plus complexe)
- Conversions: Exactes selon définitions SI

CONSIDÉRATIONS PRATIQUES:
------------------------
- Vieillissement des conduites: Réduction du coefficient C
- Variations saisonnières: Température et viscosité
- Incertitudes de mesure: Débits et pressions
- Facteurs de sécurité: Marges de conception recommandées
"""