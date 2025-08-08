# -*- coding: utf-8 -*-
"""
Module d'Optimisation de Réseaux Hydrauliques par Algorithmes Génétiques
=========================================================================

Ce module implémente un système complet d'optimisation pour réseaux hydrauliques
utilisant des algorithmes génétiques mono et multi-objectif. Il permet d'optimiser
le dimensionnement des conduites d'un réseau de distribution d'eau en respectant
les contraintes hydrauliques et en minimisant les objectifs définis.

Le système utilise WNTR (Water Network Tool for Resilience) pour la simulation
hydraulique et DEAP (Distributed Evolutionary Algorithms in Python) pour
l'implémentation des algorithmes génétiques.

Fonctionnalités principales:
---------------------------
- Optimisation mono-objectif (minimisation des pertes de charge)
- Optimisation multi-objectif NSGA-II (pertes, vitesses, pressions)
- Génération automatique de visualisations et rapports
- Sauvegarde et reprise d'optimisations
- Gestion robuste des encodages de fichiers INP
- Export de réseaux optimisés au format EPANET

Classes principales:
-------------------
- OptimisationReseau: Classe principale d'optimisation
- AnalyseEconomique: Analyse économique des solutions (legacy)

Author: Équipe d'Optimisation Hydraulique
Version: 3.0
Date: 2025
License: MIT

Dependencies:
------------
- wntr: Simulation hydraulique EPANET
- deap: Algorithmes évolutionnaires
- numpy, pandas: Calculs numériques et manipulation de données
- matplotlib, seaborn: Visualisation
- json: Sérialisation des résultats
"""

# Imports système et calculs scientifiques
import copy
import os
import json
import random
import logging
import tempfile
import shutil
from datetime import datetime

# Imports calculs scientifiques et visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as mc
import matplotlib.colors as mcolors

# Imports spécialisés
import wntr  # Water Network Tool for Resilience - Simulation hydraulique
from deap import base, creator, tools, algorithms  # Algorithmes génétiques

# Configuration locale
from . import config

# ------------------------------------------------------------------------
# PARAMÈTRES GLOBAUX (importés depuis config.py)
# ------------------------------------------------------------------------
PRESSION_MIN = config.PRESSION_MIN
PRESSION_MAX = config.PRESSION_MAX
DIAMETRE_MIN = config.DIAMETRE_MIN
DIAMETRE_MAX = config.DIAMETRE_MAX
VITESSE_MIN = config.VITESSE_MIN
VITESSE_MAX = config.VITESSE_MAX

TAILLE_POPULATION = config.TAILLE_POPULATION
NOMBRE_GENERATIONS = config.NOMBRE_GENERATIONS
TAUX_CROISEMENT = config.TAUX_CROISEMENT
TAUX_MUTATION_INDIVIDU = config.TAUX_MUTATION_INDIVIDU
TAUX_MUTATION_GENE = config.TAUX_MUTATION_GENE

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# (Optionnel) Dossier(s) de sortie
os.makedirs("visualisation", exist_ok=True)
os.makedirs("rapports", exist_ok=True)

# Diamètres disponibles (mm)
DIAMETRES_DISPONIBLES = config.DIAMETRES_DISPONIBLES

# ========================================================================
# CLASSE PRINCIPALE D'OPTIMISATION
# ========================================================================

class OptimisationReseau:
    """
    Classe Principale d'Optimisation de Réseaux Hydrauliques
    ========================================================
    
    Cette classe implémente un système complet d'optimisation pour réseaux
    hydrauliques utilisant des algorithmes génétiques. Elle permet d'optimiser
    le dimensionnement des conduites en minimisant différents objectifs tout
    en respectant les contraintes hydrauliques.
    
    Fonctionnalités:
    ---------------
    - Optimisation mono-objectif (pertes de charge)
    - Optimisation multi-objectif NSGA-II (pertes, vitesses, pressions)
    - Simulation hydraulique via WNTR/EPANET
    - Génération automatique de visualisations et rapports
    - Sauvegarde/reprise d'optimisations
    - Export de réseaux optimisés
    
    Attributs principaux:
    --------------------
    - reseau: Modèle WNTR du réseau hydraulique
    - fichier_inp: Chemin vers le fichier INP source
    - diametres_initiaux: Diamètres d'origine des conduites
    - meilleure_solution: Solution optimale mono-objectif
    - solutions_pareto: Ensemble des solutions Pareto (multi-objectif)
    - historique_fitness: Historique de convergence
    
    Exemple d'utilisation:
    ---------------------
    >>> opt = OptimisationReseau("reseau.inp")
    >>> opt.executer_optimisation()  # Mono-objectif
    >>> opt.executer_optimisation_multi()  # Multi-objectif
    >>> opt.generer_rapports()
    >>> opt.generer_fichier_inp_optimise()
    """
    
    def __init__(self, fichier_inp):
        """
        Initialise le système d'optimisation avec un réseau hydraulique.
        
        Cette méthode charge un fichier INP EPANET, initialise les structures
        de données nécessaires et configure le système de logging. Elle gère
        automatiquement les problèmes d'encodage des fichiers INP.
        
        Processus d'initialisation:
        --------------------------
        1. Chargement du fichier INP avec gestion d'encodage robuste
        2. Extraction et sauvegarde des diamètres initiaux
        3. Validation de la cohérence du réseau
        4. Initialisation des structures de données d'optimisation
        
        Parameters:
        -----------
        fichier_inp : str
            Chemin vers le fichier INP EPANET contenant la définition du réseau.
            Le fichier doit contenir au minimum les sections [JUNCTIONS], 
            [PIPES], [RESERVOIRS] ou [TANKS], et [OPTIONS].
        
        Raises:
        -------
        FileNotFoundError
            Si le fichier INP spécifié n'existe pas
        UnicodeDecodeError
            Si aucun encodage supporté ne permet de lire le fichier
        ValueError
            Si le fichier INP est mal formé ou incomplet
        wntr.network.io.InvalidNetworkError
            Si le réseau contient des erreurs de définition
            
        Notes:
        ------
        - Supporte les encodages: UTF-8, Latin-1, CP1252, ISO-8859-1
        - Les diamètres sont automatiquement convertis en mm pour l'optimisation
        - Un fichier temporaire UTF-8 est créé si nécessaire puis supprimé
        - La validation hydraulique est effectuée automatiquement
        
        Examples:
        ---------
        >>> # Initialisation basique
        >>> opt = OptimisationReseau("mon_reseau.inp")
        
        >>> # Avec gestion d'erreur
        >>> try:
        ...     opt = OptimisationReseau("reseau_complexe.inp")
        ... except FileNotFoundError:
        ...     print("Fichier INP introuvable")
        ... except ValueError as e:
        ...     print(f"Erreur de format: {e}")
        """
        try:
            print("Chargement du réseau hydraulique...")
            
            # Stocker le chemin du fichier INP
            self.fichier_inp = fichier_inp
            
            # Tentative de chargement avec gestion d'encodage
            try:
                self.reseau = wntr.network.WaterNetworkModel(fichier_inp)
            except UnicodeDecodeError as e:
                print(f"Erreur d'encodage détectée: {e}")
                print("Tentative de chargement avec encodage latin-1...")
                
                # Créer une copie temporaire avec encodage correct
                import tempfile
                import shutil
                
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False, encoding='utf-8')
                
                # Lire le fichier original avec différents encodages
                encodings = ['latin-1', 'cp1252', 'iso-8859-1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(fichier_inp, 'r', encoding=encoding) as f:
                            content = f.read()
                        print(f"Fichier lu avec succès en {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    raise ValueError("Impossible de lire le fichier INP avec aucun encodage supporté")
                
                # Écrire le contenu en UTF-8
                temp_file.write(content)
                temp_file.close()
                
                # Charger le fichier temporaire
                self.reseau = wntr.network.WaterNetworkModel(temp_file.name)
                
                # Nettoyer le fichier temporaire
                os.unlink(temp_file.name)

            # Sauvegarde diamètres initiaux
            self.diametres_initiaux = []
            for nom_conduite in self.reseau.pipe_name_list:
                self.diametres_initiaux.append(self.reseau.get_link(nom_conduite).diameter)

            print(f"Réseau chargé : {len(self.reseau.node_name_list)} nœuds, "
                  f"{len(self.reseau.link_name_list)} conduites")

            # Validation du réseau
            self.valider_reseau()

            # Pour stocker les résultats
            self.meilleure_solution = None
            self.historique_fitness = []
            self.solutions_pareto   = []

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation : {e}")
            raise

    # --------------------------------------------------------------------
    # VALIDATION DU RÉSEAU
    # --------------------------------------------------------------------
    def valider_reseau(self):
        """Vérifie la cohérence du réseau"""
        try:
            if not self.reseau.pipe_name_list:
                raise ValueError("Aucune conduite trouvée dans le réseau")
            
            for nom_conduite in self.reseau.pipe_name_list:
                diametre = self.reseau.get_link(nom_conduite).diameter
                if diametre <= 0:
                    raise ValueError(f"Diamètre invalide pour {nom_conduite}: {diametre}")
                
                longueur = self.reseau.get_link(nom_conduite).length
                if longueur <= 0:
                    raise ValueError(f"Longueur invalide pour {nom_conduite}: {longueur}")
            
            logger.info("Validation du réseau réussie")
            
        except Exception as e:
            logger.error(f"Erreur de validation du réseau : {e}")
            raise

    # --------------------------------------------------------------------
    # GESTION DE LA POPULATION INITIALE
    # --------------------------------------------------------------------
    def initialiser_population(self):
        """Crée la population initiale en choisissant aléatoirement
        parmi les diamètres disponibles dans DIAMETRES_DISPONIBLES."""
        population = []
        nb_conduites = len(self.reseau.pipe_name_list)

        for _ in range(TAILLE_POPULATION):
            individu = [np.random.choice(DIAMETRES_DISPONIBLES)
                        for _ in range(nb_conduites)]
            population.append(individu)
        return population

    # --------------------------------------------------------------------
    # MUTATION DISCRÈTE
    # --------------------------------------------------------------------
    def mutation_discrete(self, individual, indpb=None):
        """
        Mutation discrète qui remplace aléatoirement des diamètres
        par d'autres diamètres valides de la liste DIAMETRES_DISPONIBLES.
        """
        if indpb is None:
            indpb = TAUX_MUTATION_GENE
            
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.choice(DIAMETRES_DISPONIBLES)
        return individual,

    # ====================================================================
    # ÉVALUATION MONO-OBJECTIF - FONCTION FITNESS PRINCIPALE
    # ====================================================================
    
    def evaluer_solution(self, solution):
        """
        Évalue une solution d'optimisation mono-objectif en calculant un score composite.
        
        Cette fonction constitue le cœur de l'algorithme génétique mono-objectif.
        Elle simule le comportement hydraulique du réseau avec les diamètres proposés
        et calcule un score basé sur les pertes de charge totales augmentées de
        pénalités pour les contraintes violées.
        
        Formule de score:
        ----------------
        Score = Σ(Pertes_de_charge) + Σ(Pénalités_contraintes)
        
        Où:
        - Pertes_de_charge: Somme des pertes de charge de toutes les conduites (m)
        - Pénalités_contraintes: Pénalités pour pressions et vitesses hors limites
        
        Processus d'évaluation:
        ----------------------
        1. Application des diamètres proposés au réseau
        2. Simulation hydraulique via EPANET
        3. Calcul des pertes de charge totales
        4. Calcul des pénalités pour contraintes violées
        5. Restauration des diamètres initiaux
        6. Retour du score composite
        
        Parameters:
        -----------
        solution : list of float
            Liste des diamètres en mm pour chaque conduite du réseau.
            La longueur doit correspondre au nombre de conduites.
            Chaque diamètre doit être dans DIAMETRES_DISPONIBLES.
        
        Returns:
        --------
        tuple of float
            Tuple contenant le score d'évaluation. Plus le score est faible,
            meilleure est la solution. Format requis par DEAP.
        
        Raises:
        -------
        Exception
            En cas d'erreur de simulation hydraulique ou de calcul.
            Retourne (inf,) pour signaler une solution invalide.
            
        Notes:
        ------
        - Les diamètres sont convertis de mm vers m pour WNTR
        - La simulation utilise le solveur EPANET intégré à WNTR
        - Les diamètres initiaux sont restaurés après chaque évaluation
        - Les erreurs de simulation invalident automatiquement la solution
        
        Formules hydrauliques utilisées:
        --------------------------------
        - Pertes de charge: Équation de Hazen-Williams ou Darcy-Weisbach
        - Pénalités: Fonction quadratique des dépassements de contraintes
        
        Examples:
        ---------
        >>> solution = [110, 160, 200, 110]  # Diamètres en mm
        >>> score = opt.evaluer_solution(solution)
        >>> print(f"Score: {score[0]:.2f}")
        """
        try:
            # Appliquer les diamètres
            for i, diametre_mm in enumerate(solution):
                self.reseau.get_link(self.reseau.pipe_name_list[i]).diameter = diametre_mm / 1000.0

            # Simulation
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # Calcul des pertes de charge totales
            pertes_total = self.calculer_pertes_total(resultats)
            # Pénalités
            penalites = self.calculer_penalites(resultats)

            # Réinit
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            score = pertes_total + penalites
            return (score,)

        except Exception as e:
            logger.error(f"Erreur evaluer_solution : {e}")
            return (float('inf'),)

    # --------------------------------------------------------------------
    # ÉVALUATION MULTI‐OBJECTIF
    # --------------------------------------------------------------------
    def evaluer_solution_multi(self, solution):
        """
        Retourne trois objectifs à minimiser :
        (pertes_de_charge, vitesses_moyennes, pressions_ecart).
        """
        try:
            # Appliquer diamètres
            for i, diametre_mm in enumerate(solution):
                self.reseau.get_link(self.reseau.pipe_name_list[i]).diameter = diametre_mm / 1000.0

            # Simulation
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # Objectifs
            pertes_total = self.calculer_pertes_total(resultats)
            vitesses_moy = self.calculer_vitesses_moyennes(resultats)
            pressions_ecart = self.calculer_ecart_pressions(resultats)

            # Pénalités séparées
            penalites_pertes = self.calculer_penalites_pertes(resultats)
            penalites_vitesses = self.calculer_penalites_vitesses(resultats)
            penalites_pressions = self.calculer_penalites_pressions(resultats)
            
            pertes_total += penalites_pertes
            vitesses_moy += penalites_vitesses
            pressions_ecart += penalites_pressions

            # Réinit
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            return (pertes_total, vitesses_moy, pressions_ecart)

        except Exception as e:
            logger.error(f"Erreur evaluer_solution_multi : {e}")
            return (float('inf'), float('inf'), float('inf'))

    # ====================================================================
    # CALCUL DES OBJECTIFS D'OPTIMISATION - MÉTRIQUES HYDRAULIQUES
    # ====================================================================
    
    def calculer_pertes_total(self, resultats):
        """
        Calcule la somme des pertes de charge totales du réseau hydraulique.
        
        Cette méthode constitue l'objectif principal de l'optimisation mono-objectif.
        Elle calcule la somme des pertes de charge de toutes les conduites du réseau
        sur toute la période de simulation.
        
        Formule hydraulique:
        -------------------
        Pertes_totales = Σ|ΔH_i| pour toutes les conduites i
        
        Où ΔH_i est la perte de charge dans la conduite i (m), calculée par EPANET
        selon l'équation de Hazen-Williams ou Darcy-Weisbach:
        
        Hazen-Williams: ΔH = 10.67 × L × (Q^1.852) / (C^1.852 × D^4.87)
        Où: L=longueur(m), Q=débit(m³/s), C=coefficient rugosité, D=diamètre(m)
        
        Parameters:
        -----------
        resultats : wntr.sim.results.SimulationResults
            Résultats de simulation WNTR contenant les données hydrauliques
            temporelles pour tous les éléments du réseau.
        
        Returns:
        --------
        float
            Somme des valeurs absolues des pertes de charge (m).
            Plus cette valeur est faible, plus le réseau est efficace.
        
        Notes:
        ------
        - Utilise la valeur absolue pour gérer les écoulements bidirectionnels
        - Les pertes sont calculées par EPANET selon la formule configurée
        - Retourne inf en cas d'erreur de simulation pour invalider la solution
        
        Examples:
        ---------
        >>> resultats = sim.run_sim()
        >>> pertes = opt.calculer_pertes_total(resultats)
        >>> print(f"Pertes totales: {pertes:.2f} m")
        """
        try:
            # Extraction des pertes de charge depuis les résultats WNTR
            # Format: [temps, conduites] - valeurs en mètres
            headloss = resultats.link['headloss'].values
            
            # Somme des valeurs absolues pour toutes les conduites et tous les pas de temps
            pertes_total = np.sum(np.abs(headloss))
            
            return pertes_total
        except Exception as e:
            logger.error(f"Erreur calcul pertes total : {e}")
            return float('inf')  # Solution invalide

    # ====================================================================
    # CALCUL DES OBJECTIFS MULTI-OBJECTIFS - MÉTRIQUES COMPLÉMENTAIRES
    # ====================================================================
    
    def calculer_vitesses_moyennes(self, resultats):
        """
        Calcule la vitesse d'écoulement moyenne pondérée du réseau.
        
        Cette méthode calcule la vitesse moyenne d'écoulement dans toutes les
        conduites du réseau. Elle constitue un des objectifs de l'optimisation
        multi-objectif pour équilibrer les vitesses et éviter les extrêmes.
        
        Formule hydraulique:
        -------------------
        V_moyenne = (1/n) × Σ(Q_i / A_i) pour toutes les conduites i
        
        Où:
        - Q_i = débit moyen dans la conduite i (m³/s)
        - A_i = section de la conduite i = π × (D_i/2)² (m²)
        - D_i = diamètre de la conduite i (m)
        - n = nombre de conduites
        
        Parameters:
        -----------
        resultats : wntr.sim.results.SimulationResults
            Résultats de simulation contenant les débits temporels.
        
        Returns:
        --------
        float
            Vitesse d'écoulement moyenne du réseau (m/s).
            Valeur optimale typique: 0.8-1.2 m/s
        
        Notes:
        ------
        - Utilise la valeur absolue des débits (écoulements bidirectionnels)
        - Les vitesses nulles (diamètre=0) sont ignorées
        - Objectif d'optimisation: minimiser l'écart aux vitesses optimales
        """
        try:
            # Extraction des débits depuis les résultats WNTR
            # Format: [temps, conduites] - valeurs en m³/s
            debits = resultats.link['flowrate'].values
            vitesses = []
            
            # Calcul de la vitesse pour chaque conduite
            for i, pipe_name in enumerate(self.reseau.pipe_name_list):
                pipe = self.reseau.get_link(pipe_name)
                diametre = pipe.diameter  # Diamètre en mètres
                
                # Débit moyen temporel pour cette conduite (valeur absolue)
                debit_moy = np.mean(np.abs(debits[:, i]))
                
                if diametre > 0:
                    # Formule: V = Q / A avec A = π × (D/2)²
                    section = np.pi * (diametre / 2) ** 2
                    vitesse = debit_moy / section
                    vitesses.append(vitesse)
                else:
                    vitesses.append(0)  # Conduite fermée ou inexistante
            
            # Moyenne des vitesses de toutes les conduites
            return np.mean(vitesses)
            
        except Exception as e:
            logger.error(f"Erreur calcul vitesses moyennes : {e}")
            return float('inf')
    
    def calculer_ecart_pressions(self, resultats):
        """
        Calcule l'écart-type des pressions pour mesurer l'uniformité du service.
        
        Cette méthode évalue l'homogénéité de la distribution des pressions dans
        le réseau. Un écart-type faible indique un service plus uniforme pour
        tous les usagers, ce qui constitue un objectif d'optimisation.
        
        Formule statistique:
        -------------------
        σ_p = √[(1/n) × Σ(P_i - P_moyenne)²] pour tous les nœuds i
        
        Où:
        - P_i = pression moyenne temporelle au nœud i (mCE)
        - P_moyenne = pression moyenne de tous les nœuds (mCE)
        - n = nombre de nœuds de demande
        
        Parameters:
        -----------
        resultats : wntr.sim.results.SimulationResults
            Résultats contenant les pressions temporelles aux nœuds.
        
        Returns:
        --------
        float
            Écart-type des pressions moyennes (mCE).
            Plus cette valeur est faible, plus le service est uniforme.
        
        Notes:
        ------
        - Calcule d'abord la moyenne temporelle pour chaque nœud
        - Puis l'écart-type spatial de ces moyennes
        - Objectif: minimiser les disparités de pression entre zones
        - Contribue à l'équité du service hydraulique
        """
        try:
            # Extraction des pressions depuis les résultats WNTR
            # Format: [temps, nœuds] - valeurs en mCE (mètres Colonne d'Eau)
            pressions = resultats.node['pressure'].values
            
            # Calcul de la pression moyenne temporelle pour chaque nœud
            pressions_moy = np.mean(pressions, axis=0)
            
            # Calcul de l'écart-type spatial des pressions moyennes
            ecart_type = np.std(pressions_moy)
            
            return ecart_type
            
        except Exception as e:
            logger.error(f"Erreur calcul écart pressions : {e}")
            return float('inf')

    # --------------------------------------------------------------------
    # CALCUL DES PÉNALITÉS
    # --------------------------------------------------------------------
    def calculer_penalites(self, resultats):
        """
        Pénalités si la pression ou la vitesse dépassent [PRESSION_MIN, PRESSION_MAX]
        ou [VITESSE_MIN, VITESSE_MAX].
        """
        penalites = 0

        # Facteur de pénalité basé sur les pertes de charge moyennes
        facteur_penalite = 1000  # Pénalités fixes

        # Pressions
        pressions = resultats.node['pressure'].values  # shape = (time_steps, nb_nodes)
        pressions_moy = pressions.mean(axis=0)
        for p in pressions_moy:
            if p < PRESSION_MIN:
                penalites += (PRESSION_MIN - p) * facteur_penalite
            elif p > PRESSION_MAX:
                penalites += (p - PRESSION_MAX) * facteur_penalite

        # Vitesses
        vitesses = resultats.link['velocity'].values
        vitesses_moy = vitesses.mean(axis=0)
        for v in vitesses_moy:
            if v < VITESSE_MIN:
                penalites += (VITESSE_MIN - v) * facteur_penalite
            elif v > VITESSE_MAX:
                penalites += (v - VITESSE_MAX) * facteur_penalite

        return penalites



    # --------------------------------------------------------------------
    # CALCUL ÉNERGIE
    # --------------------------------------------------------------------
    def calculer_energie(self, resultats):
        """
        Calcul de l'énergie consommée en utilisant les vraies durées de simulation.
        """
        try:
            # Calcul de la durée réelle de simulation
            temps_simulation = resultats.node['pressure'].index[-1] - resultats.node['pressure'].index[0]
            
            # Gestion des différents types de données temporelles
            if hasattr(temps_simulation, 'total_seconds'):
                # Si c'est un timedelta
                temps_h = temps_simulation.total_seconds() / 3600.0
            elif hasattr(temps_simulation, 'item'):
                # Si c'est un numpy.int64 ou autre type numpy
                temps_h = float(temps_simulation.item()) / 3600.0
            else:
                # Fallback : conversion directe
                temps_h = float(temps_simulation) / 3600.0
            
            debits = resultats.link['flowrate'].values
            headloss = resultats.link['headloss'].values
            rho, g = 1000, 9.81

            # Calcul des puissances instantanées (W)
            puissances = rho * g * debits * headloss
            
            # Énergie totale en kWh
            energie_kwh = np.sum(puissances) * temps_h / 1000.0
            return energie_kwh
            
        except Exception as e:
            print(f"Erreur calcul énergie : {e}")
            # Fallback vers l'ancienne méthode si erreur
            debits = resultats.link['flowrate'].values
            headloss = resultats.link['headloss'].values
            rho, g = 1000, 9.81
            puissances = rho * g * debits * headloss
            total_puissance = puissances.sum()
            n_steps = debits.shape[0]
            total_time_h = n_steps / 3600.0
            energie_kwh = (total_puissance * total_time_h) / 1000.0
            return energie_kwh

    # --------------------------------------------------------------------
    # PÉNALITÉS SPÉCIFIQUES MULTI-OBJECTIFS
    # --------------------------------------------------------------------
    def calculer_penalites_pertes(self, resultats):
        """Pénalités pour les pertes de charge excessives"""
        penalites = 0
        facteur_penalite = 1000
        
        headloss = resultats.link['headloss'].values
        pertes_moy = np.mean(np.abs(headloss))
        
        if pertes_moy > 50:  # Pénalité si pertes moyennes > 50m
            penalites += (pertes_moy - 50) * facteur_penalite
        
        return penalites
    
    def calculer_penalites_vitesses(self, resultats):
        """Pénalités pour les vitesses hors limites"""
        penalites = 0
        facteur_penalite = 1000
        
        debits = resultats.link['flowrate'].values
        for i, pipe_name in enumerate(self.reseau.pipe_name_list):
            pipe = self.reseau.get_link(pipe_name)
            diametre = pipe.diameter
            debit_moy = np.mean(np.abs(debits[:, i]))
            
            if diametre > 0:
                vitesse = debit_moy / (np.pi * (diametre/2)**2)
                if vitesse < VITESSE_MIN:
                    penalites += (VITESSE_MIN - vitesse) * facteur_penalite
                if vitesse > VITESSE_MAX:
                    penalites += (vitesse - VITESSE_MAX) * facteur_penalite
        
        return penalites
    
    def calculer_penalites_pressions(self, resultats):
        """Pénalités pour les pressions hors limites"""
        penalites = 0
        facteur_penalite = 1000
        
        pressions = resultats.node['pressure'].values
        pressions_moy = np.mean(pressions, axis=0)
        
        for p in pressions_moy:
            if p < PRESSION_MIN:
                penalites += (PRESSION_MIN - p) * facteur_penalite
            if p > PRESSION_MAX:
                penalites += (p - PRESSION_MAX) * facteur_penalite
        
        return penalites

    # --------------------------------------------------------------------
    # OPTIMISATION MONO‐OBJECTIF
    # ====================================================================
    # ALGORITHME GÉNÉTIQUE MONO-OBJECTIF - MÉTHODE PRINCIPALE
    # ====================================================================
    
    def executer_optimisation(self, callback=None, arret_demande=None):
        """
        Exécute l'optimisation mono-objectif du réseau hydraulique par algorithme génétique.
        
        Cette méthode implémente un algorithme génétique standard pour minimiser
        les pertes de charge totales du réseau tout en respectant les contraintes
        hydrauliques (pressions et vitesses). Elle utilise la bibliothèque DEAP
        pour l'implémentation des opérateurs évolutionnaires.
        
        Objectif d'optimisation:
        -----------------------
        Minimiser: Score = Σ(Pertes_de_charge) + Σ(Pénalités_contraintes)
        
        Algorithme génétique utilisé:
        -----------------------------
        - Population: TAILLE_POPULATION individus (défaut: 100)
        - Générations: NOMBRE_GENERATIONS maximum (défaut: 100)
        - Sélection: Tournoi de taille 5 (pression sélective équilibrée)
        - Croisement: Deux points (TAUX_CROISEMENT = 0.8)
        - Mutation: Discrète sur diamètres disponibles (TAUX_MUTATION = 0.2)
        - Arrêt: Convergence ou nombre max de générations
        
        Processus d'optimisation:
        ------------------------
        1. Initialisation d'une population aléatoire
        2. Évaluation de tous les individus
        3. Pour chaque génération:
           - Sélection des parents par tournoi
           - Reproduction par croisement deux points
           - Mutation discrète des descendants
           - Évaluation des nouveaux individus
           - Remplacement de la population
           - Vérification des critères d'arrêt
        4. Génération automatique du fichier INP optimisé
        
        Parameters:
        -----------
        callback : callable, optional
            Fonction appelée à chaque génération pour mise à jour GUI.
            Signature: callback(generation_number)
        arret_demande : callable, optional
            Fonction retournant True si l'utilisateur demande l'arrêt.
            Permet l'interruption interactive de l'optimisation.
        
        Returns:
        --------
        None
            Les résultats sont stockés dans les attributs de classe:
            - self.meilleure_solution: Meilleure configuration trouvée
            - self.historique_fitness: Évolution des scores par génération
        
        Raises:
        -------
        Exception
            En cas d'erreur durant l'optimisation (simulation, calcul, etc.)
            
        Side Effects:
        ------------
        - Met à jour self.meilleure_solution avec la meilleure configuration
        - Remplit self.historique_fitness avec l'évolution des scores
        - Génère automatiquement un fichier INP optimisé avec timestamp
        - Enregistre les logs d'optimisation
        
        Critères d'arrêt:
        -----------------
        - Nombre maximum de générations atteint
        - Convergence détectée (pas d'amélioration pendant N générations)
        - Arrêt demandé par l'utilisateur (via arret_demande)
        
        Notes techniques:
        ----------------
        - Utilise DEAP pour l'implémentation des opérateurs génétiques
        - Les diamètres sont encodés comme entiers (mm) dans les chromosomes
        - La fonction de fitness intègre pénalités pour contraintes violées
        - Sauvegarde périodique possible selon configuration
        
        Examples:
        ---------
        >>> opt = OptimisationReseau("reseau.inp")
        >>> # Optimisation simple
        >>> opt.executer_optimisation()
        
        >>> # Avec callback pour GUI
        >>> def update_progress(gen):
        ...     print(f"Génération {gen}")
        >>> opt.executer_optimisation(callback=update_progress)
        
        >>> # Avec possibilité d'arrêt
        >>> stop_flag = False
        >>> opt.executer_optimisation(arret_demande=lambda: stop_flag)
        """
        try:
            # Création des types DEAP
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()

            # Opérateurs
            toolbox.register("individual", tools.initIterate, creator.Individual,
                             lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", self.mutation_discrete)
            toolbox.register("select", tools.selTournament, tournsize=5)  # Augmenté pour plus de diversité

            # Population initiale
            population = toolbox.population(n=TAILLE_POPULATION)

            # Évaluation initiale
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            meilleur_score_global = float('inf')
            generations_sans_amelioration = 0

            print("\n=== Début optimisation MONO‐OBJECTIF ===\n")

            for gen in range(NOMBRE_GENERATIONS):
                # Vérification de l'arrêt demandé
                if arret_demande and arret_demande():
                    print("Arrêt de l'optimisation demandé par l'utilisateur.")
                    break
                offspring = algorithms.varAnd(
                    population, toolbox,
                    cxpb=TAUX_CROISEMENT,
                    mutpb=TAUX_MUTATION_INDIVIDU
                )

                # Évaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Sélection avec élitisme
                population_combinee = offspring + population
                population = toolbox.select(population_combinee, k=len(population)-2)
                
                # Élitisme : garder les 2 meilleurs individus
                meilleurs = tools.selBest(population_combinee, k=2)
                population.extend(meilleurs)

                # Statistiques
                fits = [ind.fitness.values[0] for ind in population]
                score_min = min(fits)
                score_moy = sum(fits) / len(fits)

                if score_min < meilleur_score_global:
                    meilleur_score_global = score_min
                    generations_sans_amelioration = 0
                    amelioration = "⭐"
                else:
                    generations_sans_amelioration += 1
                    amelioration = "  "

                # Calcul de métriques supplémentaires
                score_max = max(fits)
                ecart_type = np.std(fits)
                diversite = len(set([round(f, 2) for f in fits])) / len(fits)
                
                print(f"Génération {gen+1} | Min: {score_min:.2f} | Moy: {score_moy:.2f} | Max: {score_max:.2f} | σ: {ecart_type:.2f} | Div: {diversite:.2f} {amelioration}")
                self.historique_fitness.append(score_min)

                if callback:
                    callback(gen)

                # Sauvegarde périodique
                if config.SAUVEGARDE_PERIODIQUE and (gen + 1) % config.FREQUENCE_SAUVEGARDE == 0:
                    self.sauvegarder_etat(gen + 1, population, score_min)
                
                if generations_sans_amelioration >= config.GENERATIONS_SANS_AMELIORATION_MAX:
                    print(f"Arrêt anticipé ({config.GENERATIONS_SANS_AMELIORATION_MAX} gen sans amélioration).")
                    break

            print(f"Meilleur score final : {meilleur_score_global:.2f}\n")

            # Meilleure solution
            self.meilleure_solution = tools.selBest(population, k=1)[0]
            
            # Générer le fichier INP optimisé
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nom_fichier = f"reseau_optimise_mono_{timestamp}.inp"
                self.generer_fichier_inp_optimise(self.meilleure_solution, nom_fichier)
            except Exception as e:
                logger.warning(f"Impossible de générer le fichier INP optimisé : {e}")
            
            # Log des métriques finales
            logger.info(f"Optimisation terminée - Meilleur score: {meilleur_score_global:.2f}")
            logger.info(f"Nombre de générations sans amélioration: {generations_sans_amelioration}")

        except Exception as e:
            logger.error(f"Erreur executer_optimisation : {e}")
            raise

    # --------------------------------------------------------------------
    # SAUVEGARDE D'ÉTAT
    # --------------------------------------------------------------------
    def sauvegarder_etat(self, generation, population, meilleur_score):
        """Sauvegarde l'état actuel de l'optimisation"""
        try:
            etat = {
                'generation': generation,
                'meilleur_score': meilleur_score,
                'historique_fitness': self.historique_fitness.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Sauvegarde de la meilleure solution si disponible
            if self.meilleure_solution:
                etat['meilleure_solution'] = list(self.meilleure_solution)
            
            filename = f'checkpoint_gen_{generation}.json'
            with open(filename, 'w') as f:
                json.dump(etat, f, indent=2)
            
            logger.info(f"État sauvegardé dans {filename}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde d'état : {e}")

    # --------------------------------------------------------------------
    # REPRODUCTION GÉNÉTIQUE (pour multi‐objectif)
    # --------------------------------------------------------------------
    def reproduction_genetique(self, population, toolbox, cxpb=0.8, mutpb=0.2):
        """Fonction manuelle qui clone, croise et mute."""
        offspring = [copy.deepcopy(ind) for ind in population]

        for child1, child2 in zip(offspring[0::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    # --------------------------------------------------------------------
    # OPTIMISATION MULTI‐OBJECTIF
    # --------------------------------------------------------------------
    def executer_optimisation_multi(self, callback=None, arret_demande=None):
        """
        Minimisation de (pertes_de_charge, vitesses_moyennes, pressions_ecart) + pénalités
        """
        try:
            if not hasattr(creator, "FitnessMulti"):
                creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
                creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()

            toolbox.register("individual", tools.initIterate, creator.IndividualMulti,
                             lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution_multi)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", self.mutation_discrete)
            toolbox.register("select", tools.selNSGA2)

            population = toolbox.population(n=TAILLE_POPULATION)

            # Évaluation initiale
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Tri initial
            population = toolbox.select(population, len(population))

            print("\n=== Début optimisation MULTI‐OBJECTIF (NSGA‐II) ===\n")

            for gen in range(NOMBRE_GENERATIONS):
                # Vérification de l'arrêt demandé
                if arret_demande and arret_demande():
                    print("Arrêt de l'optimisation multi-objectif demandé par l'utilisateur.")
                    break
                # Utilisation de varAnd() pour NSGA-II
                offspring = algorithms.varAnd(
                    population, toolbox,
                    cxpb=TAUX_CROISEMENT,
                    mutpb=TAUX_MUTATION_INDIVIDU
                )

                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                population = toolbox.select(population + offspring, k=len(population))

                # Récup front non dominé
                fronts = tools.sortNondominated(population, len(population), first_front_only=True)
                pareto = fronts[0] if fronts else []
                print(f"Génération {gen+1} : {len(pareto)} solutions sur le front Pareto")

                if callback:
                    callback(gen)

            final_front = tools.sortNondominated(population, k=len(population), first_front_only=True)
            self.solutions_pareto = final_front[0] if final_front else []

            print(f"Front de Pareto final : {len(self.solutions_pareto)} solutions\n")
            
            # Générer le fichier INP optimisé pour la meilleure solution du front de Pareto
            if self.solutions_pareto:
                try:
                    # Choisir la première solution du front de Pareto comme solution représentative
                    solution_representative = self.solutions_pareto[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nom_fichier = f"reseau_optimise_multi_{timestamp}.inp"
                    self.generer_fichier_inp_optimise(solution_representative, nom_fichier)
                    
                    # Optionnel : générer plusieurs solutions représentatives
                    if len(self.solutions_pareto) >= 3:
                        # Solution avec meilleure performance sur le premier objectif (pertes)
                        meilleure_pertes = min(self.solutions_pareto, key=lambda x: x.fitness.values[0])
                        nom_fichier_pertes = f"reseau_optimise_multi_meilleure_pertes_{timestamp}.inp"
                        self.generer_fichier_inp_optimise(meilleure_pertes, nom_fichier_pertes)
                        
                        # Solution avec meilleure performance sur le deuxième objectif (vitesses)
                        meilleure_vitesses = min(self.solutions_pareto, key=lambda x: x.fitness.values[1])
                        nom_fichier_vitesses = f"reseau_optimise_multi_meilleure_vitesses_{timestamp}.inp"
                        self.generer_fichier_inp_optimise(meilleure_vitesses, nom_fichier_vitesses)
                        
                        # Solution avec meilleure performance sur le troisième objectif (pressions)
                        meilleure_pressions = min(self.solutions_pareto, key=lambda x: x.fitness.values[2])
                        nom_fichier_pressions = f"reseau_optimise_multi_meilleure_pressions_{timestamp}.inp"
                        self.generer_fichier_inp_optimise(meilleure_pressions, nom_fichier_pressions)
                        
                        print(f"📄 4 fichiers INP multi-objectif générés avec différentes optimisations")
                        
                except Exception as e:
                    logger.warning(f"Impossible de générer les fichiers INP optimisés multi-objectif : {e}")

        except Exception as e:
            print(f"Erreur executer_optimisation_multi : {e}")
            raise
    # --------------------------------------------------------------------
    # MÉTHODES DE VISUALISATION
    # --------------------------------------------------------------------
    def _plot_convergence(self):
        """Graphique de convergence (mono‐objectif)"""
        if not self.historique_fitness:
            print("Aucun historique de fitness à tracer.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.historique_fitness, marker='o')
        plt.title('Convergence (mono‐objectif)')
        plt.xlabel('Génération')
        plt.ylabel('Meilleure fitness (score)')
        plt.grid(True)
        plt.savefig('data/results/visualisation/convergence.png')  # <--- Dans dossier "visualisation"
        plt.close()
        print("Graphique de convergence sauvegardé : data/results/visualisation/convergence.png")

    def _plot_distribution_pressions(self):
        """Distribution des pressions pour la meilleure solution (mono‐objectif)."""
        if not self.meilleure_solution:
            print("Pas de meilleure solution pour tracer la distribution des pressions.")
            return
        try:
            # 1) Appliquer la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()
            pressions = resultats.node['pressure'].mean()  # moyenne sur le temps (chaque nœud)

            plt.figure(figsize=(12, 6))
            # Histogramme
            plt.subplot(1,2,1)
            plt.hist(pressions, bins=20, color='skyblue', edgecolor='black')
            plt.axvline(PRESSION_MIN, color='r', linestyle='--', label=f'Min ({PRESSION_MIN})')
            plt.axvline(PRESSION_MAX, color='r', linestyle='--', label=f'Max ({PRESSION_MAX})')
            plt.title('Distribution des pressions')
            plt.xlabel('Pression (m)')  # <--- SI c'est en mètres
            plt.ylabel('Nombre de nœuds')
            plt.legend()

            # Boxplot
            plt.subplot(1,2,2)
            plt.boxplot(pressions)
            plt.title('Boxplot des pressions')
            plt.ylabel('Pression (m)')

            plt.tight_layout()
            plt.savefig('data/results/visualisation/distribution_pressions.png')
            plt.close()

            # Réinit.
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("Distribution des pressions : data/results/visualisation/distribution_pressions.png")

        except Exception as e:
            print(f"Erreur dans _plot_distribution_pressions : {e}")

    def _plot_carte_chaleur_pertes(self):
        """
        Carte de chaleur des pertes de charge améliorée avec visualisation des nœuds.
        Version améliorée avec meilleures couleurs, annotations et légendes.
        """
        if not self.meilleure_solution:
            print("Pas de meilleure solution pour tracer la carte de chaleur.")
            return
        try:
            # 1) Appliquer la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # 2) Calcul des pertes de charge réelles (formule de Darcy-Weisbach simplifiée)
            pertes = {}
            vitesses = {}
            debits = {}
            
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                debit = resultats.link['flowrate'][pipe_name].mean()
                headloss = resultats.link['headloss'][pipe_name].mean()
                longu = pipe.length
                diam = pipe.diameter
                
                # Calcul de la vitesse
                if diam > 0:
                    vitesse = abs(debit) / (np.pi * (diam/2)**2)
                    vitesses[pipe_name] = vitesse
                else:
                    vitesses[pipe_name] = 0
                
                debits[pipe_name] = abs(debit)
                
                # Utiliser les pertes de charge réelles calculées par EPANET
                pertes[pipe_name] = abs(headloss) if headloss is not None else 0

            # 3) Construire la liste de segments avec informations enrichies
            segments = []
            values_pertes = []
            values_vitesses = []
            values_debits = []
            pipe_names = []
            
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                start_node = self.reseau.get_node(pipe.start_node_name)
                end_node = self.reseau.get_node(pipe.end_node_name)

                x1, y1 = start_node.coordinates
                x2, y2 = end_node.coordinates

                seg = [(x1, y1), (x2, y2)]
                segments.append(seg)
                values_pertes.append(pertes[pipe_name])
                values_vitesses.append(vitesses[pipe_name])
                values_debits.append(debits[pipe_name])
                pipe_names.append(pipe_name)

            # 4) Créer une figure avec plusieurs sous-graphiques
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Analyse Complète du Réseau Hydraulique Optimisé', fontsize=16, fontweight='bold')

            # 4.1) Carte de chaleur des pertes de charge
            cmap_pertes = plt.cm.Reds
            norm_pertes = mcolors.Normalize(vmin=min(values_pertes), vmax=max(values_pertes))
            lc_pertes = mc.LineCollection(segments, cmap=cmap_pertes, norm=norm_pertes, linewidths=6)
            lc_pertes.set_array(np.array(values_pertes))

            ax1.add_collection(lc_pertes)
            
            # Ajouter les nœuds avec taille variable selon la pression
            pressions_nodes = []
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                pression = resultats.node['pressure'][node_name].mean()
                pressions_nodes.append(pression)
                ax1.scatter(node.coordinates[0], node.coordinates[1], 
                           c='navy', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            # Ajuster l'échelle
            all_x = [c[0] for seg in segments for c in seg]
            all_y = [c[1] for seg in segments for c in seg]
            ax1.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax1.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_pertes, ax=ax1, label='Pertes de charge (m)', shrink=0.8)
            ax1.set_title('Pertes de Charge par Conduite', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Coordonnée X (m)')
            ax1.set_ylabel('Coordonnée Y (m)')
            ax1.grid(True, alpha=0.3)

            # 4.2) Carte de chaleur des vitesses
            cmap_vitesses = plt.cm.Blues
            norm_vitesses = mcolors.Normalize(vmin=min(values_vitesses), vmax=max(values_vitesses))
            lc_vitesses = mc.LineCollection(segments, cmap=cmap_vitesses, norm=norm_vitesses, linewidths=6)
            lc_vitesses.set_array(np.array(values_vitesses))

            ax2.add_collection(lc_vitesses)
            
            # Ajouter les nœuds
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                ax2.scatter(node.coordinates[0], node.coordinates[1], 
                           c='darkred', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            ax2.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax2.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_vitesses, ax=ax2, label='Vitesse (m/s)', shrink=0.8)
            ax2.set_title('Vitesses d\'Écoulement par Conduite', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Coordonnée X (m)')
            ax2.set_ylabel('Coordonnée Y (m)')
            ax2.grid(True, alpha=0.3)

            # 4.3) Carte de chaleur des débits
            cmap_debits = plt.cm.Greens
            norm_debits = mcolors.Normalize(vmin=min(values_debits), vmax=max(values_debits))
            lc_debits = mc.LineCollection(segments, cmap=cmap_debits, norm=norm_debits, linewidths=6)
            lc_debits.set_array(np.array(values_debits))

            ax3.add_collection(lc_debits)
            
            # Ajouter les nœuds
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                ax3.scatter(node.coordinates[0], node.coordinates[1], 
                           c='purple', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            ax3.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax3.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_debits, ax=ax3, label='Débit (m³/s)', shrink=0.8)
            ax3.set_title('Débits par Conduite', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Coordonnée X (m)')
            ax3.set_ylabel('Coordonnée Y (m)')
            ax3.grid(True, alpha=0.3)

            # 4.4) Graphique de synthèse - Distribution des pressions aux nœuds
            ax4.hist(pressions_nodes, bins=15, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
            ax4.axvline(np.mean(pressions_nodes), color='red', linestyle='--', linewidth=2, 
                       label=f'Moyenne: {np.mean(pressions_nodes):.1f} m')
            ax4.axvline(np.median(pressions_nodes), color='orange', linestyle='--', linewidth=2,
                       label=f'Médiane: {np.median(pressions_nodes):.1f} m')
            
            ax4.set_title('Distribution des Pressions aux Nœuds', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Pression (m)')
            ax4.set_ylabel('Nombre de nœuds')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 5) Ajouter des statistiques en texte
            stats_text = f"""
Statistiques du réseau optimisé:
• Pertes de charge moyennes: {np.mean(values_pertes):.3f} m
• Vitesse moyenne: {np.mean(values_vitesses):.2f} m/s
• Débit total: {np.sum(values_debits):.2f} m³/s
• Pression moyenne: {np.mean(pressions_nodes):.1f} m
• Nombre de conduites: {len(segments)}
• Nombre de nœuds: {len(self.reseau.node_name_list)}
            """
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.savefig('data/results/visualisation/carte_chaleur_pertes_amelioree.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 6) Créer aussi une version interactive avec Plotly
            self._creer_carte_interactive(segments, values_pertes, values_vitesses, values_debits, pipe_names)

            # 7) Réinitialiser
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("✅ Carte de chaleur améliorée : data/results/visualisation/carte_chaleur_pertes_amelioree.png")
            print("✅ Carte interactive : data/results/visualisation/carte_interactive.html")

        except Exception as e:
            print(f"❌ Erreur lors de la carte de chaleur des pertes : {e}")
            import traceback
            traceback.print_exc()

    def _plot_carte_chaleur_amelioree(self):
        """
        Version améliorée de la carte de chaleur avec visualisations multiples.
        """
        if not self.meilleure_solution:
            print("Pas de meilleure solution pour tracer la carte de chaleur.")
            return
        try:
            # 1) Appliquer la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # 2) Calcul des métriques
            pertes = {}
            vitesses = {}
            debits = {}
            
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                debit = resultats.link['flowrate'][pipe_name].mean()
                headloss = resultats.link['headloss'][pipe_name].mean()
                longu = pipe.length
                diam = pipe.diameter
                
                # Vitesse
                if diam > 0:
                    vitesse = abs(debit) / (np.pi * (diam/2)**2)
                    vitesses[pipe_name] = vitesse
                else:
                    vitesses[pipe_name] = 0
                
                debits[pipe_name] = abs(debit)
                pertes[pipe_name] = abs(headloss) if headloss is not None else 0

            # 3) Préparer les données
            segments = []
            values_pertes = []
            values_vitesses = []
            values_debits = []
            
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                start_node = self.reseau.get_node(pipe.start_node_name)
                end_node = self.reseau.get_node(pipe.end_node_name)

                x1, y1 = start_node.coordinates
                x2, y2 = end_node.coordinates

                seg = [(x1, y1), (x2, y2)]
                segments.append(seg)
                values_pertes.append(pertes[pipe_name])
                values_vitesses.append(vitesses[pipe_name])
                values_debits.append(debits[pipe_name])

            # 4) Créer la figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Analyse Complète du Réseau Hydraulique Optimisé', fontsize=16, fontweight='bold')

            # 4.1) Pertes de charge
            cmap_pertes = plt.cm.Reds
            norm_pertes = mcolors.Normalize(vmin=min(values_pertes), vmax=max(values_pertes))
            lc_pertes = mc.LineCollection(segments, cmap=cmap_pertes, norm=norm_pertes, linewidths=6)
            lc_pertes.set_array(np.array(values_pertes))

            ax1.add_collection(lc_pertes)
            
            # Nœuds
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                ax1.scatter(node.coordinates[0], node.coordinates[1], 
                           c='navy', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            all_x = [c[0] for seg in segments for c in seg]
            all_y = [c[1] for seg in segments for c in seg]
            ax1.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax1.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_pertes, ax=ax1, label='Pertes de charge (m)', shrink=0.8)
            ax1.set_title('Pertes de Charge par Conduite', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Coordonnée X (m)')
            ax1.set_ylabel('Coordonnée Y (m)')
            ax1.grid(True, alpha=0.3)

            # 4.2) Vitesses
            cmap_vitesses = plt.cm.Blues
            norm_vitesses = mcolors.Normalize(vmin=min(values_vitesses), vmax=max(values_vitesses))
            lc_vitesses = mc.LineCollection(segments, cmap=cmap_vitesses, norm=norm_vitesses, linewidths=6)
            lc_vitesses.set_array(np.array(values_vitesses))

            ax2.add_collection(lc_vitesses)
            
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                ax2.scatter(node.coordinates[0], node.coordinates[1], 
                           c='darkred', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            ax2.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax2.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_vitesses, ax=ax2, label='Vitesse (m/s)', shrink=0.8)
            ax2.set_title('Vitesses d\'Écoulement par Conduite', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Coordonnée X (m)')
            ax2.set_ylabel('Coordonnée Y (m)')
            ax2.grid(True, alpha=0.3)

            # 4.3) Débits
            cmap_debits = plt.cm.Greens
            norm_debits = mcolors.Normalize(vmin=min(values_debits), vmax=max(values_debits))
            lc_debits = mc.LineCollection(segments, cmap=cmap_debits, norm=norm_debits, linewidths=6)
            lc_debits.set_array(np.array(values_debits))

            ax3.add_collection(lc_debits)
            
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                ax3.scatter(node.coordinates[0], node.coordinates[1], 
                           c='purple', s=50, marker='o', alpha=0.8, edgecolors='white', linewidth=1)

            ax3.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax3.set_ylim(min(all_y)*0.95, max(all_y)*1.05)
            
            plt.colorbar(lc_debits, ax=ax3, label='Débit (m³/s)', shrink=0.8)
            ax3.set_title('Débits par Conduite', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Coordonnée X (m)')
            ax3.set_ylabel('Coordonnée Y (m)')
            ax3.grid(True, alpha=0.3)

            # 4.4) Distribution des pertes
            ax4.hist(values_pertes, bins=15, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
            ax4.axvline(np.mean(values_pertes), color='red', linestyle='--', linewidth=2, 
                       label=f'Moyenne: {np.mean(values_pertes):.3f} m')
            ax4.axvline(np.median(values_pertes), color='orange', linestyle='--', linewidth=2,
                       label=f'Médiane: {np.median(values_pertes):.3f} m')
            
            ax4.set_title('Distribution des Pertes de Charge', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Pertes de charge (m)')
            ax4.set_ylabel('Nombre de conduites')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 5) Statistiques
            stats_text = f"""
Statistiques du réseau optimisé:
• Pertes moyennes: {np.mean(values_pertes):.3f} m
• Vitesse moyenne: {np.mean(values_vitesses):.2f} m/s
• Débit total: {np.sum(values_debits):.2f} m³/s
• Conduites: {len(segments)}
• Nœuds: {len(self.reseau.node_name_list)}
            """
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.savefig('data/results/visualisation/carte_chaleur_amelioree.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 6) Réinitialiser
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("✅ Carte de chaleur améliorée : data/results/visualisation/carte_chaleur_amelioree.png")

        except Exception as e:
            print(f"❌ Erreur lors de la carte de chaleur améliorée : {e}")
            import traceback
            traceback.print_exc()

    def _creer_carte_interactive(self, segments, values_pertes, values_vitesses, values_debits, pipe_names):
        """
        Créer une carte interactive avec Plotly pour une exploration dynamique.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Créer les données pour les traces
            traces_pertes = []
            traces_vitesses = []
            traces_debits = []
            
            for i, seg in enumerate(segments):
                x_coords = [seg[0][0], seg[1][0]]
                y_coords = [seg[0][1], seg[1][1]]
                
                # Trace pour les pertes
                traces_pertes.append(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(width=8, color=values_pertes[i]),
                    name=f'Conduite {pipe_names[i]}',
                    hovertemplate=f'<b>Conduite:</b> {pipe_names[i]}<br>' +
                                 f'<b>Pertes:</b> {values_pertes[i]:.3f} m<br>' +
                                 f'<b>Vitesse:</b> {values_vitesses[i]:.2f} m/s<br>' +
                                 f'<b>Débit:</b> {values_debits[i]:.3f} m³/s<extra></extra>',
                    showlegend=False
                ))
                
                # Trace pour les vitesses
                traces_vitesses.append(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(width=8, color=values_vitesses[i]),
                    name=f'Conduite {pipe_names[i]}',
                    hovertemplate=f'<b>Conduite:</b> {pipe_names[i]}<br>' +
                                 f'<b>Vitesse:</b> {values_vitesses[i]:.2f} m/s<extra></extra>',
                    showlegend=False
                ))
                
                # Trace pour les débits
                traces_debits.append(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(width=8, color=values_debits[i]),
                    name=f'Conduite {pipe_names[i]}',
                    hovertemplate=f'<b>Conduite:</b> {pipe_names[i]}<br>' +
                                 f'<b>Débit:</b> {values_debits[i]:.3f} m³/s<extra></extra>',
                    showlegend=False
                ))

            # Créer la figure avec sous-graphiques
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Pertes de Charge', 'Vitesses d\'Écoulement', 
                              'Débits', 'Statistiques'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "bar"}]]
            )

            # Ajouter les traces
            for trace in traces_pertes:
                fig.add_trace(trace, row=1, col=1)
            
            for trace in traces_vitesses:
                fig.add_trace(trace, row=1, col=2)
            
            for trace in traces_debits:
                fig.add_trace(trace, row=2, col=1)

            # Ajouter un histogramme des pertes
            fig.add_trace(
                go.Histogram(x=values_pertes, nbinsx=15, name='Distribution des pertes',
                           marker_color='lightcoral'),
                row=2, col=2
            )

            # Mettre à jour la mise en page
            fig.update_layout(
                title_text="Analyse Interactive du Réseau Hydraulique",
                title_x=0.5,
                showlegend=False,
                height=800,
                width=1200
            )

            # Mettre à jour les axes
            fig.update_xaxes(title_text="Coordonnée X (m)", row=1, col=1)
            fig.update_yaxes(title_text="Coordonnée Y (m)", row=1, col=1)
            fig.update_xaxes(title_text="Coordonnée X (m)", row=1, col=2)
            fig.update_yaxes(title_text="Coordonnée Y (m)", row=1, col=2)
            fig.update_xaxes(title_text="Coordonnée X (m)", row=2, col=1)
            fig.update_yaxes(title_text="Coordonnée Y (m)", row=2, col=1)
            fig.update_xaxes(title_text="Pertes de charge (m)", row=2, col=2)
            fig.update_yaxes(title_text="Nombre de conduites", row=2, col=2)

            # Sauvegarder
            fig.write_html('data/results/visualisation/carte_interactive.html')
            
        except ImportError:
            print("⚠️ Plotly non disponible - carte interactive non générée")
        except Exception as e:
            print(f"❌ Erreur lors de la création de la carte interactive : {e}")

    def _plot_evolution_temporelle(self):
        """Évolution temporelle (pression moyenne)."""
        if not self.meilleure_solution:
            return
        try:
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            df_p = resultats.node['pressure']
            # Moyenne de la pression à chaque pas de temps
            pression_moy = df_p.mean(axis=1)

            plt.figure(figsize=(10,6))
            plt.plot(pression_moy.index, pression_moy.values, marker='o', linestyle='-')
            plt.title("Évolution temporelle de la pression moyenne")
            plt.xlabel("Temps (pas de simulation)")
            plt.ylabel("Pression (m)")
            plt.grid(True)
            plt.savefig("data/results/visualisation/evolution_temporelle.png", dpi=300)
            plt.close()

            # Réinit
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("Évolution temporelle : data/results/visualisation/evolution_temporelle.png")

        except Exception as e:
            print(f"Erreur dans _plot_evolution_temporelle : {e}")

    def _plot_pareto(self):
        """Trace le front de Pareto (pertes vs vitesses vs pressions)."""
        if not self.solutions_pareto:
            print("Aucune solution Pareto disponible.")
            return
        try:
            pertes = [ind.fitness.values[0] for ind in self.solutions_pareto]
            vitesses = [ind.fitness.values[1] for ind in self.solutions_pareto]
            pressions = [ind.fitness.values[2] for ind in self.solutions_pareto]

            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(pertes, vitesses, pressions, c=pressions, cmap='viridis', s=50)
            ax.set_xlabel("Pertes de charge (m)")
            ax.set_ylabel("Vitesses moyennes (m/s)")
            ax.set_zlabel("Écart pressions (m)")
            ax.set_title("Front de Pareto - (Pertes vs Vitesses vs Pressions)")
            
            plt.colorbar(scatter, ax=ax, label='Écart pressions (m)')
            plt.tight_layout()
            plt.savefig("data/results/visualisation/front_pareto.png", dpi=300)
            plt.close()

            print("Front de Pareto : data/results/visualisation/front_pareto.png")

        except Exception as e:
            print(f"Erreur dans _plot_pareto : {e}")

    # --------------------------------------------------------------------
    # COMPARAISON AVANT/APRÈS
    # --------------------------------------------------------------------
    def generer_comparaison(self):
        """Comparaison avant/après pour la meilleure solution (mono‐objectif)."""
        if not self.meilleure_solution:
            return
        try:
            # Simulation initiale
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
            sim_init = wntr.sim.EpanetSimulator(self.reseau)
            resultats_initiaux = sim_init.run_sim()

            pressions_initiales = resultats_initiaux.node['pressure'].mean().values
            pertes_init = resultats_initiaux.link['flowrate'].sum().sum()
            energie_init = self.calculer_energie(resultats_initiaux)

            # Simulation optimisée
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            sim_opt = wntr.sim.EpanetSimulator(self.reseau)
            resultats_opt = sim_opt.run_sim()

            pressions_optim = resultats_opt.node['pressure'].mean().values
            pertes_opt = resultats_opt.link['flowrate'].sum().sum()
            energie_opt = self.calculer_energie(resultats_opt)

            dict_init = {
                'pressions': pressions_initiales,
                'pertes_totales': pertes_init,
                'energie_totale': energie_init
            }
            dict_opt = {
                'pressions': pressions_optim,
                'pertes_totales': pertes_opt,
                'energie_totale': energie_opt
            }

            self._plot_comparaison(dict_init, dict_opt)

            # Réinit
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

        except Exception as e:
            print(f"Erreur lors de la comparaison : {e}")

    def _plot_comparaison(self, resultats_initiaux, resultats_optimises):
        """Graphiques comparatifs avant/après."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1) Distribution des pressions
            ax1.hist(resultats_initiaux['pressions'], bins=30, alpha=0.5,
                     label='Initial', color='blue')
            ax1.hist(resultats_optimises['pressions'], bins=30, alpha=0.5,
                     label='Optimisé', color='green')
            ax1.set_title('Distribution des pressions')
            ax1.set_xlabel('Pression (m)')
            ax1.set_ylabel('Fréquence')
            ax1.legend()
            ax1.grid(True)

            # 2) Pertes totales
            labels = ['Initial', 'Optimisé']
            pertes_vals = [resultats_initiaux['pertes_totales'],
                           resultats_optimises['pertes_totales']]
            ax2.bar(labels, pertes_vals, color=['blue', 'green'])
            ax2.set_title('Pertes totales')
            ax2.set_ylabel('Débit total (m³/h)')
            ax2.grid(True)

            # 3) Énergie
            energie_vals = [resultats_initiaux['energie_totale'],
                            resultats_optimises['energie_totale']]
            ax3.bar(labels, energie_vals, color=['blue', 'green'])
            ax3.set_title('Consommation énergétique')
            ax3.set_ylabel('Énergie (kWh)')
            ax3.grid(True)

            # 4) Améliorations en %
            ameliorations = {
                'Pressions': (
                    (np.mean(resultats_optimises['pressions']) -
                     np.mean(resultats_initiaux['pressions']))
                    / np.mean(resultats_initiaux['pressions']) * 100
                ),
                'Pertes': (
                    (resultats_optimises['pertes_totales'] -
                     resultats_initiaux['pertes_totales'])
                    / resultats_initiaux['pertes_totales'] * 100
                ),
                'Énergie': (
                    (resultats_optimises['energie_totale'] -
                     resultats_initiaux['energie_totale'])
                    / resultats_initiaux['energie_totale'] * 100
                )
            }
            parametres = list(ameliorations.keys())
            valeurs = list(ameliorations.values())
            colors = ['green' if v < 0 else 'red' for v in valeurs]

            ax4.bar(parametres, valeurs, color=colors)
            ax4.set_title("Pourcentage d'amélioration (valeurs négatives = réduction)")
            ax4.set_ylabel('Amélioration (%)')
            ax4.grid(True)

            plt.tight_layout()
            fig.suptitle('Comparaison avant/après optimisation', fontsize=16, y=1.02)
            plt.savefig('data/results/visualisation/comparaison_resultats.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Petit rapport texte
            rapport = f"""
            --- Rapport de comparaison ---

            Pression moyenne initiale  : {np.mean(resultats_initiaux['pressions']):.2f} m
            Pression moyenne optimisée : {np.mean(resultats_optimises['pressions']):.2f} m

            Pertes initiales  : {resultats_initiaux['pertes_totales']:.2f} m³/h
            Pertes optimisées : {resultats_optimises['pertes_totales']:.2f} m³/h

            Énergie initiale  : {resultats_initiaux['energie_totale']:.2f} kWh
            Énergie optimisée : {resultats_optimises['energie_totale']:.2f} kWh

            Améliorations (en %) :
            - Pressions : {ameliorations['Pressions']:.2f} %
            - Pertes    : {ameliorations['Pertes']:.2f} %
            - Énergie   : {ameliorations['Énergie']:.2f} %
            """
            with open('data/results/rapports/rapport_comparaison.txt', 'w') as f:
                f.write(rapport)

            print("Comparaison avant/après : data/results/visualisation/comparaison_resultats.png\n"
                  "Rapport texte : data/results/rapports/rapport_comparaison.txt")

        except Exception as e:
            print(f"Erreur dans _plot_comparaison : {e}")

    # --------------------------------------------------------------------
    # EXPORTS DES RAPPORTS
    # --------------------------------------------------------------------
    def generer_rapports(self):
        """Génération des rapports CSV (mono‐objectif)."""
        self._export_resume_optimisation()
        self._export_indicateurs_performance()
        self._export_parametres_critiques()

    def generer_rapports_multi_objectif(self):
        """Génération des rapports CSV pour l'optimisation multi-objectif."""
        self._export_resume_multi_objectif()
        self._export_indicateurs_multi_objectif()
        self._export_solutions_pareto()

    def _export_resume_multi_objectif(self):
        """Export du résumé multi-objectif."""
        if not self.solutions_pareto:
            return
            
        pertes = [sol.fitness.values[0] for sol in self.solutions_pareto]
        vitesses = [sol.fitness.values[1] for sol in self.solutions_pareto]
        pressions = [sol.fitness.values[2] for sol in self.solutions_pareto]
        
        df_resume = pd.DataFrame({
            'Métrique': [
                'Nombre de solutions Pareto',
                'Pertes minimum (m)',
                'Pertes maximum (m)',
                'Pertes moyennes (m)',
                'Vitesse minimum (m/s)',
                'Vitesse maximum (m/s)',
                'Vitesse moyenne (m/s)',
                'Écart pressions minimum (m)',
                'Écart pressions maximum (m)',
                'Écart pressions moyen (m)'
            ],
            'Valeur': [
                len(self.solutions_pareto),
                min(pertes),
                max(pertes),
                np.mean(pertes),
                min(vitesses),
                max(vitesses),
                np.mean(vitesses),
                min(pressions),
                max(pressions),
                np.mean(pressions)
            ]
        })
        df_resume.to_csv('data/results/rapports/resume_multi_objectif.csv', index=False)
        print("Résumé multi-objectif : data/results/rapports/resume_multi_objectif.csv")

    def _export_indicateurs_multi_objectif(self):
        """Export des indicateurs multi-objectif."""
        if not self.solutions_pareto:
            return
            
        pertes = [sol.fitness.values[0] for sol in self.solutions_pareto]
        vitesses = [sol.fitness.values[1] for sol in self.solutions_pareto]
        pressions = [sol.fitness.values[2] for sol in self.solutions_pareto]
        
        # Calculer des indicateurs de qualité
        spread_pertes = max(pertes) - min(pertes)
        spread_vitesses = max(vitesses) - min(vitesses)
        spread_pressions = max(pressions) - min(pressions)
        
        # Calculer l'hypervolume approximatif (simplifié)
        hypervolume = spread_pertes * spread_vitesses * spread_pressions
        
        data = [
            {'Indicateur': 'Étendue pertes (m)', 'Valeur': spread_pertes},
            {'Indicateur': 'Étendue vitesses (m/s)', 'Valeur': spread_vitesses},
            {'Indicateur': 'Étendue pressions (m)', 'Valeur': spread_pressions},
            {'Indicateur': 'Hypervolume approximatif', 'Valeur': hypervolume},
            {'Indicateur': 'Diversité des solutions', 'Valeur': len(self.solutions_pareto)},
            {'Indicateur': 'Ratio pertes/vitesses min', 'Valeur': min(pertes) / min(vitesses) if min(vitesses) > 0 else 0},
            {'Indicateur': 'Ratio pertes/vitesses max', 'Valeur': max(pertes) / max(vitesses) if max(vitesses) > 0 else 0}
        ]
        
        df = pd.DataFrame(data)
        df.to_csv('data/results/rapports/indicateurs_multi_objectif.csv', index=False)
        print("Indicateurs multi-objectif : data/results/rapports/indicateurs_multi_objectif.csv")

    def _export_solutions_pareto(self):
        """Export des solutions du front de Pareto."""
        if not self.solutions_pareto:
            return
            
        solutions_data = []
        for i, sol in enumerate(self.solutions_pareto):
            solution_data = {
                'Solution': i + 1,
                'Pertes (m)': sol.fitness.values[0],
                'Vitesses (m/s)': sol.fitness.values[1],
                'Écart Pressions (m)': sol.fitness.values[2]
            }
            
            # Ajouter les diamètres
            for j, diametre in enumerate(sol):
                nom_conduite = self.reseau.pipe_name_list[j]
                solution_data[f'Diamètre_{nom_conduite} (mm)'] = diametre
            
            solutions_data.append(solution_data)
        
        df = pd.DataFrame(solutions_data)
        df.to_csv('data/results/rapports/solutions_pareto.csv', index=False)
        print("Solutions Pareto : data/results/rapports/solutions_pareto.csv")

    def _export_resume_optimisation(self):
        """Export du tableau récapitulatif (mono‐objectif)."""
        if not self.meilleure_solution:
            return
        reseau_opt = copy.deepcopy(self.reseau)
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_opt.pipe_name_list[i]
            reseau_opt.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(reseau_opt)
        resultats = sim.run_sim()

        df_resume = pd.DataFrame({
            'Métrique': [
                'Pertes totales (m)',
                'Pression moyenne (m)',
                'Pression minimale (m)',
                'Pression maximale (m)',
                'Vitesse moyenne (m/s)',
                'Vitesse minimale (m/s)',
                'Vitesse maximale (m/s)'
            ],
            'Valeur': [
                self.calculer_pertes_total(resultats),
                resultats.node['pressure'].mean().mean(),
                resultats.node['pressure'].min().min(),
                resultats.node['pressure'].max().max(),
                resultats.link['velocity'].mean().mean(),
                resultats.link['velocity'].min().min(),
                resultats.link['velocity'].max().max()
            ]
        })
        df_resume.to_csv('data/results/rapports/resume_optimisation.csv', index=False)
        print("Résumé optimisation : data/results/rapports/resume_optimisation.csv")

    def _export_indicateurs_performance(self):
        """Export des indicateurs (exemple)."""
        data = [
            {'Zone': 'Nord', 'Pression moyenne': 8.5, 'Nombre de noeuds': 10},
            {'Zone': 'Sud',  'Pression moyenne': 7.2, 'Nombre de noeuds': 8},
        ]
        df = pd.DataFrame(data)
        df.to_csv('data/results/rapports/indicateurs_performance.csv', index=False)
        print("Indicateurs performance : data/results/rapports/indicateurs_performance.csv")

    def _export_parametres_critiques(self):
        """Export des éléments où les contraintes ne sont pas respectées."""
        if not self.meilleure_solution:
            return
        reseau_opt = copy.deepcopy(self.reseau)
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_opt.pipe_name_list[i]
            reseau_opt.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(reseau_opt)
        res = sim.run_sim()

        pressions_moy = res.node['pressure'].mean()
        vitesses_moy  = res.link['velocity'].mean()

        parametres_critiques = []

        for node_name in pressions_moy.index:
            p = pressions_moy[node_name]
            if p < PRESSION_MIN or p > PRESSION_MAX:
                parametres_critiques.append({
                    'Element': node_name,
                    'Type': 'Nœud',
                    'Paramètre': 'Pression (m)',
                    'Valeur': p
                })

        for link_name in vitesses_moy.index:
            v = vitesses_moy[link_name]
            if v < VITESSE_MIN or v > VITESSE_MAX:
                parametres_critiques.append({
                    'Element': link_name,
                    'Type': 'Conduite',
                    'Paramètre': 'Vitesse (m/s)',
                    'Valeur': v
                })

        df = pd.DataFrame(parametres_critiques)
        df.to_csv('data/results/rapports/parametres_critiques.csv', index=False)
        print("Paramètres critiques : data/results/rapports/parametres_critiques.csv")

    def exporter_resultats_json(self):
        """Export en JSON (mono‐objectif)."""
        if not self.meilleure_solution:
            return
        reseau_opt = copy.deepcopy(self.reseau)
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_opt.pipe_name_list[i]
            reseau_opt.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(reseau_opt)
        resultats = sim.run_sim()

        pressions = {
            n: float(resultats.node['pressure'][n].mean())
            for n in reseau_opt.node_name_list
        }
        vitesses = {
            l: float(resultats.link['velocity'][l].mean())
            for l in reseau_opt.link_name_list
        }

        data_json = {
            "date_optimisation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parametres": {
                "pression_min": PRESSION_MIN,
                "pression_max": PRESSION_MAX,
                "diametre_min": DIAMETRE_MIN,
                "diametre_max": DIAMETRE_MAX,
                "vitesse_min": VITESSE_MIN,
                "vitesse_max": VITESSE_MAX
            },
            "meilleure_solution": {
                "fitness": float(self.meilleure_solution.fitness.values[0]),
                "diametres": {
                    pipe: int(d) 
                    for pipe, d in zip(reseau_opt.pipe_name_list, self.meilleure_solution)
                },
                "pressions": pressions,
                "vitesses": vitesses
            },
            "statistiques": {
                "pertes_total": self.calculer_pertes_total(resultats),
                "historique_convergence": [float(f) for f in self.historique_fitness]
            }
        }

        with open('data/results/rapports/resultats_optimisation.json', 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=4, ensure_ascii=False)

        print("Résultats JSON : data/results/rapports/resultats_optimisation.json")

    def _plot_visualisations_ameliorees(self):
        """
        Génère toutes les visualisations améliorées en une seule fois.
        """
        if not self.meilleure_solution:
            print("Pas de meilleure solution pour générer les visualisations.")
            return
        try:
            print("🎨 Génération des visualisations améliorées...")
            
            # Appliquer la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # 1. Convergence améliorée avec plus de détails
            self._plot_convergence_amelioree()
            
            # 2. Distribution des pressions améliorée
            self._plot_distribution_pressions_amelioree(resultats)
            
            # 3. Évolution temporelle améliorée
            self._plot_evolution_temporelle_amelioree(resultats)
            
            # 4. Carte de chaleur améliorée
            self._plot_carte_chaleur_amelioree()
            
            # 5. Analyse comparative améliorée
            self._plot_comparaison_amelioree(resultats)
            
            # Réinitialiser
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
                
            print("✅ Toutes les visualisations améliorées ont été générées !")
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des visualisations : {e}")
            import traceback
            traceback.print_exc()

    def _plot_visualisations_multi_objectif(self):
        """
        Génère les visualisations spécifiques à l'optimisation multi-objectif.
        """
        if not self.solutions_pareto:
            print("Pas de solutions Pareto pour générer les visualisations multi-objectif.")
            return
        try:
            print("🎨 Génération des visualisations multi-objectif...")
            
            # 1. Front de Pareto (déjà généré par _plot_pareto())
            print("✅ Front de Pareto déjà généré")
            
            # 2. Analyse des solutions Pareto
            self._plot_analyse_solutions_pareto()
            
            # 3. Distribution des objectifs
            self._plot_distribution_objectifs()
            
            # 4. Sélection de solutions représentatives
            self._plot_solutions_representatives()
            
            print("✅ Toutes les visualisations multi-objectif ont été générées !")
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des visualisations multi-objectif : {e}")
            import traceback
            traceback.print_exc()

    def _plot_analyse_solutions_pareto(self):
        """Analyse des solutions du front de Pareto."""
        if not self.solutions_pareto:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Extraire les valeurs des objectifs
        pertes = [sol.fitness.values[0] for sol in self.solutions_pareto]
        vitesses = [sol.fitness.values[1] for sol in self.solutions_pareto]
        pressions = [sol.fitness.values[2] for sol in self.solutions_pareto]
        
        # Graphique principal : Front de Pareto
        plt.subplot(2, 2, (1, 3))
        scatter = plt.scatter(pertes, vitesses, alpha=0.6, s=30, c=pressions, cmap='viridis', edgecolors='black', linewidth=0.5)
        plt.xlabel('Pertes de charge (m)')
        plt.ylabel('Vitesses moyennes (m/s)')
        plt.title('Front de Pareto - Solutions Multi-objectif')
        plt.colorbar(scatter, label='Écart pressions (m)')
        plt.grid(True, alpha=0.3)
        
        # Statistiques
        plt.subplot(2, 2, 2)
        stats_data = [
            ['Nombre solutions', len(self.solutions_pareto)],
            ['Pertes min', f"{min(pertes):.1f}"],
            ['Pertes max', f"{max(pertes):.1f}"],
            ['Vitesse min', f"{min(vitesses):.2f}"],
            ['Vitesse max', f"{max(vitesses):.2f}"]
        ]
        
        plt.axis('off')
        table = plt.table(cellText=stats_data, colLabels=['Métrique', 'Valeur'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Statistiques du Front de Pareto', fontweight='bold', fontsize=12)
        
        # Distribution des pertes
        plt.subplot(2, 2, 4)
        plt.hist(pertes, bins=20, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        plt.xlabel('Pertes de charge (m)')
        plt.ylabel('Nombre de solutions')
        plt.title('Distribution des Pertes', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/visualisation/analyse_solutions_pareto.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Analyse des solutions Pareto : data/results/visualisation/analyse_solutions_pareto.png")

    def _plot_distribution_objectifs(self):
        """Distribution des valeurs des objectifs."""
        if not self.solutions_pareto:
            return
            
        pertes = [sol.fitness.values[0] for sol in self.solutions_pareto]
        vitesses = [sol.fitness.values[1] for sol in self.solutions_pareto]
        pressions = [sol.fitness.values[2] for sol in self.solutions_pareto]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution des pertes
        ax1.hist(pertes, bins=25, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.axvline(np.mean(pertes), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(pertes):.1f}')
        ax1.set_xlabel('Pertes de charge (m)')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Pertes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution des vitesses
        ax2.hist(vitesses, bins=25, color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.axvline(np.mean(vitesses), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(vitesses):.2f}')
        ax2.set_xlabel('Vitesses moyennes (m/s)')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des Vitesses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/visualisation/distribution_objectifs.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Distribution des objectifs : data/results/visualisation/distribution_objectifs.png")

    def _plot_solutions_representatives(self):
        """Sélection et visualisation de solutions représentatives."""
        if not self.solutions_pareto:
            return
            
        # Sélectionner quelques solutions représentatives
        n_solutions = min(5, len(self.solutions_pareto))
        indices = np.linspace(0, len(self.solutions_pareto)-1, n_solutions, dtype=int)
        solutions_representatives = [self.solutions_pareto[i] for i in indices]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, sol in enumerate(solutions_representatives):
            if i >= len(axes):
                break
                
            # Appliquer la solution
            for j, diametre in enumerate(sol):
                nom_conduite = self.reseau.pipe_name_list[j]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            
            # Simuler
            try:
                sim = wntr.sim.EpanetSimulator(self.reseau)
                resultats = sim.run_sim()
                
                # Extraire les pressions moyennes
                pressions = [float(resultats.node['pressure'][n].mean()) 
                           for n in self.reseau.node_name_list]
                
                # Graphique des pressions
                axes[i].hist(pressions, bins=15, color='lightblue', alpha=0.7, edgecolor='navy')
                axes[i].set_xlabel('Pression (mCE)')
                axes[i].set_ylabel('Nombre de nœuds')
                axes[i].set_title(f'Solution {i+1}\nPertes: {sol.fitness.values[0]:.1f}\nVitesses: {sol.fitness.values[1]:.2f}')
                axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Erreur simulation\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Solution {i+1} - Erreur')
            
            # Réinitialiser
            for j, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[j]
        
        # Masquer les axes inutilisés
        for i in range(len(solutions_representatives), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/results/visualisation/solutions_representatives.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Solutions représentatives : data/results/visualisation/solutions_representatives.png")

    def _plot_convergence_amelioree(self):
        """Graphique de convergence amélioré avec plus de détails."""
        if not self.historique_fitness:
            print("Aucun historique de fitness à tracer.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Graphique principal
        plt.subplot(2, 2, (1, 3))
        plt.plot(self.historique_fitness, marker='o', markersize=4, linewidth=2, 
                color='blue', alpha=0.7, label='Meilleure fitness')
        
        # Ligne de tendance
        x = np.arange(len(self.historique_fitness))
        z = np.polyfit(x, self.historique_fitness, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Tendance')
        
        plt.title('Convergence de l\'Optimisation', fontweight='bold', fontsize=14)
        plt.xlabel('Génération')
        plt.ylabel('Fitness (pertes + pénalités)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Statistiques
        plt.subplot(2, 2, 2)
        stats_data = [
            ['Valeur initiale', self.historique_fitness[0]],
            ['Valeur finale', self.historique_fitness[-1]],
            ['Amélioration', self.historique_fitness[0] - self.historique_fitness[-1]],
            ['% Amélioration', ((self.historique_fitness[0] - self.historique_fitness[-1]) / self.historique_fitness[0]) * 100]
        ]
        
        plt.axis('off')
        table = plt.table(cellText=stats_data, colLabels=['Métrique', 'Valeur'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Statistiques de Convergence', fontweight='bold', fontsize=12)
        
        # Distribution des valeurs
        plt.subplot(2, 2, 4)
        plt.hist(self.historique_fitness, bins=20, color='lightblue', alpha=0.7, edgecolor='navy')
        plt.axvline(np.mean(self.historique_fitness), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(self.historique_fitness):.2f}')
        plt.title('Distribution des Valeurs de Fitness', fontweight='bold', fontsize=12)
        plt.xlabel('Fitness')
        plt.ylabel('Fréquence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/visualisation/convergence_amelioree.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Convergence améliorée : data/results/visualisation/convergence_amelioree.png")

    def _plot_distribution_pressions_amelioree(self, resultats):
        """Distribution des pressions améliorée avec plus d'informations."""
        try:
            pressions = []
            for node_name in self.reseau.node_name_list:
                pression = resultats.node['pressure'][node_name].mean()
                pressions.append(pression)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Analyse Complète des Pressions du Réseau', fontsize=16, fontweight='bold')
            
            # Histogramme principal
            ax1.hist(pressions, bins=20, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
            ax1.axvline(np.mean(pressions), color='red', linestyle='--', linewidth=2,
                       label=f'Moyenne: {np.mean(pressions):.1f} m')
            ax1.axvline(np.median(pressions), color='orange', linestyle='--', linewidth=2,
                       label=f'Médiane: {np.median(pressions):.1f} m')
            ax1.set_title('Distribution des Pressions', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Pression (m)')
            ax1.set_ylabel('Nombre de nœuds')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(pressions, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax2.set_title('Box Plot des Pressions', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Pression (m)')
            ax2.grid(True, alpha=0.3)
            
            # Évolution temporelle des pressions moyennes
            pressions_temporelles = resultats.node['pressure'].mean(axis=1)
            ax3.plot(pressions_temporelles.index, pressions_temporelles.values, 
                    linewidth=2, color='purple', alpha=0.8)
            ax3.set_title('Évolution Temporelle - Pression Moyenne', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Temps')
            ax3.set_ylabel('Pression moyenne (m)')
            ax3.grid(True, alpha=0.3)
            
            # Statistiques détaillées
            stats_text = f"""
Statistiques des pressions:
• Minimum: {min(pressions):.1f} m
• Maximum: {max(pressions):.1f} m
• Moyenne: {np.mean(pressions):.1f} m
• Médiane: {np.median(pressions):.1f} m
• Écart-type: {np.std(pressions):.1f} m
• Nœuds < 20m: {sum(1 for p in pressions if p < 20)}
• Nœuds > 60m: {sum(1 for p in pressions if p > 60)}
            """
            
            ax4.axis('off')
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('data/results/visualisation/distribution_pressions_amelioree.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Distribution des pressions améliorée : data/results/visualisation/distribution_pressions_amelioree.png")
            
        except Exception as e:
            print(f"❌ Erreur dans la distribution des pressions améliorée : {e}")

    def _plot_evolution_temporelle_amelioree(self, resultats):
        """Évolution temporelle améliorée avec plusieurs métriques."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Évolution Temporelle du Réseau Hydraulique', fontsize=16, fontweight='bold')
            
            # Pression moyenne
            df_p = resultats.node['pressure']
            pression_moy = df_p.mean(axis=1)
            ax1.plot(pression_moy.index, pression_moy.values, marker='o', linestyle='-', 
                    linewidth=2, markersize=4, color='blue', alpha=0.8)
            ax1.set_title('Pression Moyenne', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('Pression (m)')
            ax1.grid(True, alpha=0.3)
            
            # Débit total
            df_q = resultats.link['flowrate']
            debit_total = df_q.sum(axis=1)
            ax2.plot(debit_total.index, debit_total.values, marker='s', linestyle='-',
                    linewidth=2, markersize=4, color='green', alpha=0.8)
            ax2.set_title('Débit Total du Réseau', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Temps')
            ax2.set_ylabel('Débit total (m³/s)')
            ax2.grid(True, alpha=0.3)
            
            # Pertes de charge totales
            df_h = resultats.link['headloss']
            pertes_total = df_h.sum(axis=1)
            ax3.plot(pertes_total.index, pertes_total.values, marker='^', linestyle='-',
                    linewidth=2, markersize=4, color='red', alpha=0.8)
            ax3.set_title('Pertes de Charge Totales', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Temps')
            ax3.set_ylabel('Pertes totales (m)')
            ax3.grid(True, alpha=0.3)
            
            # Vitesse moyenne
            vitesses = []
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                debit = resultats.link['flowrate'][pipe_name]
                diam = pipe.diameter
                if diam > 0:
                    vitesse = abs(debit) / (np.pi * (diam/2)**2)
                    vitesses.append(vitesse)
            
            if vitesses:
                vitesse_moy = np.mean(vitesses, axis=0)
                ax4.plot(vitesse_moy.index, vitesse_moy.values, marker='d', linestyle='-',
                        linewidth=2, markersize=4, color='purple', alpha=0.8)
                ax4.set_title('Vitesse Moyenne d\'Écoulement', fontweight='bold', fontsize=12)
                ax4.set_xlabel('Temps')
                ax4.set_ylabel('Vitesse moyenne (m/s)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data/results/visualisation/evolution_temporelle_amelioree.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Évolution temporelle améliorée : data/results/visualisation/evolution_temporelle_amelioree.png")
            
        except Exception as e:
            print(f"❌ Erreur dans l'évolution temporelle améliorée : {e}")

    def _plot_comparaison_amelioree(self, resultats_optimises):
        """Comparaison améliorée avant/après optimisation."""
        try:
            # Simuler le réseau initial
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
            
            sim_init = wntr.sim.EpanetSimulator(self.reseau)
            resultats_initiaux = sim_init.run_sim()
            
            # Réappliquer la solution optimisée
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comparaison Avant/Après Optimisation', fontsize=16, fontweight='bold')
            
            # Pressions
            pressions_init = [resultats_initiaux.node['pressure'][node].mean() for node in self.reseau.node_name_list]
            pressions_opt = [resultats_optimises.node['pressure'][node].mean() for node in self.reseau.node_name_list]
            
            ax1.hist(pressions_init, bins=20, alpha=0.5, label='Initial', color='blue', edgecolor='navy')
            ax1.hist(pressions_opt, bins=20, alpha=0.5, label='Optimisé', color='green', edgecolor='darkgreen')
            ax1.set_title('Distribution des Pressions', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Pression (m)')
            ax1.set_ylabel('Fréquence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pertes de charge
            pertes_init = resultats_initiaux.link['headloss'].sum().sum()
            pertes_opt = resultats_optimises.link['headloss'].sum().sum()
            
            labels = ['Initial', 'Optimisé']
            pertes_vals = [abs(pertes_init), abs(pertes_opt)]
            bars = ax2.bar(labels, pertes_vals, color=['lightcoral', 'lightgreen'], alpha=0.8)
            ax2.set_title('Pertes de Charge Totales', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Pertes totales (m)')
            ax2.grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, pertes_vals):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Débits
            debits_init = resultats_initiaux.link['flowrate'].sum().sum()
            debits_opt = resultats_optimises.link['flowrate'].sum().sum()
            
            debit_vals = [abs(debits_init), abs(debits_opt)]
            bars = ax3.bar(labels, debit_vals, color=['lightblue', 'lightyellow'], alpha=0.8)
            ax3.set_title('Débits Totaux', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Débit total (m³/s)')
            ax3.grid(True, alpha=0.3)
            
            for bar, val in zip(bars, debit_vals):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Améliorations en pourcentage
            ameliorations = [
                ((np.mean(pressions_opt) - np.mean(pressions_init)) / np.mean(pressions_init)) * 100,
                ((pertes_opt - pertes_init) / pertes_init) * 100 if pertes_init != 0 else 0,
                ((debits_opt - debits_init) / debits_init) * 100 if debits_init != 0 else 0
            ]
            
            labels_amel = ['Pression', 'Pertes', 'Débits']
            colors = ['green' if x > 0 else 'red' for x in ameliorations]
            bars = ax4.bar(labels_amel, ameliorations, color=colors, alpha=0.7)
            ax4.set_title('Améliorations (%)', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Amélioration (%)')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            for bar, val in zip(bars, ameliorations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                        f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('data/results/visualisation/comparaison_amelioree.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Comparaison améliorée : data/results/visualisation/comparaison_amelioree.png")
            
        except Exception as e:
            print(f"❌ Erreur dans la comparaison améliorée : {e}")

    def generer_fichier_inp_optimise(self, meilleure_solution=None, nom_fichier="reseau_optimise.inp"):
        """
        Génère un fichier INP avec les diamètres optimisés.
        
        Args:
            meilleure_solution: Liste des diamètres optimaux. Si None, utilise la meilleure solution actuelle.
            nom_fichier: Nom du fichier INP à générer
        """
        try:
            import os
            import shutil
            from datetime import datetime
            
            # Utiliser la meilleure solution si pas fournie
            if meilleure_solution is None:
                if hasattr(self, 'meilleure_solution') and self.meilleure_solution:
                    meilleure_solution = self.meilleure_solution
                else:
                    logger.error("Aucune solution optimisée disponible")
                    return None
            
            # Créer le répertoire de sortie
            output_dir = "data/results/reseaux_optimises"
            os.makedirs(output_dir, exist_ok=True)
            
            # Chemin complet du fichier de sortie
            output_path = os.path.join(output_dir, nom_fichier)
            
            # Copier le fichier INP original
            shutil.copy2(self.fichier_inp, output_path)
            
            # Lire le contenu du fichier avec gestion d'encodage
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            contenu = None
            
            for encoding in encodings:
                try:
                    with open(output_path, 'r', encoding=encoding) as f:
                        contenu = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if contenu is None:
                raise ValueError("Impossible de lire le fichier INP avec aucun encodage supporté")
            
            # Obtenir la liste des conduites
            conduites = list(self.reseau.pipe_name_list)
            
            # Remplacer les diamètres dans la section [PIPES]
            lignes = contenu.split('\n')
            dans_section_pipes = False
            nouvelles_lignes = []
            
            for ligne in lignes:
                ligne_stripped = ligne.strip()
                
                # Détecter le début de la section [PIPES]
                if ligne_stripped.upper() == '[PIPES]':
                    dans_section_pipes = True
                    nouvelles_lignes.append(ligne)
                    continue
                
                # Détecter la fin de la section [PIPES]
                elif ligne_stripped.startswith('[') and dans_section_pipes:
                    dans_section_pipes = False
                    nouvelles_lignes.append(ligne)
                    continue
                
                # Traiter les lignes dans la section [PIPES]
                elif dans_section_pipes and ligne_stripped and not ligne_stripped.startswith(';'):
                    elements = ligne_stripped.split()
                    if len(elements) >= 6:  # Format: ID Noeud1 Noeud2 Longueur Diametre Rugosite
                        pipe_id = elements[0]
                        
                        # Trouver l'index de cette conduite
                        try:
                            pipe_index = conduites.index(pipe_id)
                            nouveau_diametre = meilleure_solution[pipe_index]
                            
                            # Remplacer le diamètre (élément à l'index 4)
                            elements[4] = str(nouveau_diametre)
                            nouvelle_ligne = '\t'.join(elements)
                            nouvelles_lignes.append(nouvelle_ligne)
                            
                        except (ValueError, IndexError):
                            # Si la conduite n'est pas trouvée, garder la ligne originale
                            nouvelles_lignes.append(ligne)
                    else:
                        nouvelles_lignes.append(ligne)
                else:
                    nouvelles_lignes.append(ligne)
            
            # Ajouter un commentaire d'en-tête
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_comment = f"""; Réseau hydraulique optimisé
; Généré le : {timestamp}
; Algorithme : Génétique (NSGA-II)
; Objectifs optimisés : Pertes de charge, Vitesses, Pressions
; Nombre de conduites optimisées : {len(conduites)}

"""
            
            # Écrire le fichier modifié
            contenu_final = header_comment + '\n'.join(nouvelles_lignes)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(contenu_final)
            
            # Générer un résumé des modifications
            self._generer_resume_optimisation_inp(conduites, meilleure_solution, output_dir)
            
            logger.info(f"Fichier INP optimisé généré : {output_path}")
            print(f"📄 Fichier INP optimisé : {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du fichier INP optimisé : {e}")
            print(f"❌ Erreur génération INP : {e}")
            return None

    def _generer_resume_optimisation_inp(self, conduites, diametres_optimises, output_dir):
        """Génère un résumé des modifications apportées au réseau."""
        try:
            import pandas as pd
            
            # Obtenir les diamètres originaux
            diametres_originaux = []
            for pipe_name in conduites:
                pipe = self.reseau.get_link(pipe_name)
                diametres_originaux.append(pipe.diameter * 1000)  # Conversion en mm
            
            # Créer un DataFrame de comparaison
            comparaison = pd.DataFrame({
                'Conduite': conduites,
                'Diametre_Original_mm': diametres_originaux,
                'Diametre_Optimise_mm': diametres_optimises,
                'Difference_mm': [opt - orig for opt, orig in zip(diametres_optimises, diametres_originaux)],
                'Ratio_Optimisation': [opt/orig if orig > 0 else 1 for opt, orig in zip(diametres_optimises, diametres_originaux)]
            })
            
            # Statistiques globales
            stats = {
                'Nombre_conduites': len(conduites),
                'Diametre_moyen_original': np.mean(diametres_originaux),
                'Diametre_moyen_optimise': np.mean(diametres_optimises),
                'Augmentation_moyenne_mm': np.mean([opt - orig for opt, orig in zip(diametres_optimises, diametres_originaux)]),
                'Conduites_agrandies': sum(1 for opt, orig in zip(diametres_optimises, diametres_originaux) if opt > orig),
                'Conduites_reduites': sum(1 for opt, orig in zip(diametres_optimises, diametres_originaux) if opt < orig),
                'Conduites_inchangees': sum(1 for opt, orig in zip(diametres_optimises, diametres_originaux) if opt == orig)
            }
            
            # Sauvegarder la comparaison détaillée
            comparaison_path = os.path.join(output_dir, "comparaison_diametres.csv")
            comparaison.to_csv(comparaison_path, index=False)
            
            # Sauvegarder les statistiques
            stats_path = os.path.join(output_dir, "statistiques_optimisation.csv")
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            
            print(f"📊 Comparaison des diamètres : {comparaison_path}")
            print(f"📈 Statistiques d'optimisation : {stats_path}")
            
            # Afficher un résumé
            print(f"\n🔧 Résumé de l'optimisation :")
            print(f"   • {stats['Conduites_agrandies']} conduites agrandies")
            print(f"   • {stats['Conduites_reduites']} conduites réduites")
            print(f"   • {stats['Conduites_inchangees']} conduites inchangées")
            print(f"   • Diamètre moyen : {stats['Diametre_moyen_original']:.1f} → {stats['Diametre_moyen_optimise']:.1f} mm")
            
        except Exception as e:
            logger.error(f"Erreur génération résumé INP : {e}")
            print(f"⚠️ Erreur résumé : {e}")


# ------------------------------------------------------------------------
# Classe d'analyse économique (exemple)
# ------------------------------------------------------------------------
class AnalyseEconomique:
    def __init__(self, reseau):
        self.reseau = reseau
        self.taux_actualisation = 0.08
        self.duree_projet = 20

    def calculer_couts_investissement(self):
        couts = {
            'equipements': {
                'capteurs_pression': 1500000,
                'vannes_regulation': 2500000,
                'systeme_controle': 3500000
            },
            'installation': {
                'main_oeuvre': 1200000,
                'transport': 800000,
                'formation': 1000000
            },
            'etudes': {
                'audit_initial': 1500000,
                'conception': 2000000,
                'supervision': 1500000
            }
        }
        total_investissement = sum(sum(c.values()) for c in couts.values())
        return couts, total_investissement

    def calculer_economies_annuelles(self, resultats_optimisation):
        economies = {
            'reduction_pertes': {
                'volume_economise': resultats_optimisation.get('reduction_pertes', 5000),
                'prix_eau': 500
            },
            'energie': {
                'kwh_economises': resultats_optimisation.get('reduction_energie', 2000),
                'prix_kwh': 100
            },
            'maintenance': {
                'reduction_interventions': resultats_optimisation.get('reduction_maintenance', 10),
                'cout_intervention': 200000
            }
        }
        economies['reduction_pertes']['economie'] = (
            economies['reduction_pertes']['volume_economise'] *
            economies['reduction_pertes']['prix_eau']
        )
        economies['energie']['economie'] = (
            economies['energie']['kwh_economises'] *
            economies['energie']['prix_kwh']
        )
        economies['maintenance']['economie'] = (
            economies['maintenance']['reduction_interventions'] *
            economies['maintenance']['cout_intervention']
        )
        total_eco = sum(poste['economie'] for poste in economies.values())
        return economies, total_eco

    def calculer_van(self, investissement, economies_annuelles):
        flux = -investissement
        for annee in range(1, self.duree_projet + 1):
            flux += economies_annuelles / ((1 + self.taux_actualisation)**annee)
        return flux

    def calculer_tri(self, investissement, economies_annuelles):
        def npv(rate):
            return sum([-investissement] +
                       [economies_annuelles / (1 + rate)**t
                        for t in range(1, self.duree_projet + 1)])
        rate_low, rate_high = 0, 1
        while rate_high - rate_low > 1e-4:
            rate_mid = (rate_low + rate_high) / 2
            if npv(rate_mid) > 0:
                rate_low = rate_mid
            else:
                rate_high = rate_mid
        return rate_mid

    def analyse_sensibilite(self, investissement, economies):
        variations = [-20, -10, 0, 10, 20]
        result = {
            'variation_investissement': [],
            'variation_economies': []
        }
        for var in variations:
            inv_mod = investissement * (1 + var/100.0)
            van_inv = self.calculer_van(inv_mod, economies)
            tri_inv = self.calculer_tri(inv_mod, economies)
            result['variation_investissement'].append({
                'variation_%': var,
                'VAN': van_inv,
                'TRI': tri_inv
            })

            eco_mod = economies * (1 + var/100.0)
            van_eco = self.calculer_van(investissement, eco_mod)
            tri_eco = self.calculer_tri(investissement, eco_mod)
            result['variation_economies'].append({
                'variation_%': var,
                'VAN': van_eco,
                'TRI': tri_eco
            })
        return result

    def exporter_rapport_excel(self, rapport):
        writer = pd.ExcelWriter('data/results/rapports/rapport_economique.xlsx', engine='xlsxwriter')

        detail_inv = rapport['investissement']['detail']
        df_inv = pd.DataFrame(detail_inv).T
        df_inv['Sous-total'] = df_inv.sum(axis=1)
        df_inv.to_excel(writer, sheet_name='Investissements')

        detail_eco = rapport['economies_annuelles']['detail']
        df_eco = []
        for k, v in detail_eco.items():
            df_eco.append({'Type': k, 'Economie_FCFA': v['economie']})
        df_eco = pd.DataFrame(df_eco)
        df_eco.to_excel(writer, sheet_name='Economies', index=False)

        ind = [{
            'VAN': rapport['indicateurs_rentabilite']['VAN'],
            'TRI': rapport['indicateurs_rentabilite']['TRI'],
            'Temps_retour (années)': rapport['indicateurs_rentabilite']['temps_retour']
        }]
        df_ind = pd.DataFrame(ind)
        df_ind.to_excel(writer, sheet_name='Indicateurs', index=False)

        sensi = rapport['analyse_sensibilite']
        df_sens_inv = pd.DataFrame(sensi['variation_investissement'])
        df_sens_eco = pd.DataFrame(sensi['variation_economies'])
        df_sens_inv.to_excel(writer, sheet_name='Sensib_Invest', index=False)
        df_sens_eco.to_excel(writer, sheet_name='Sensib_Eco', index=False)

        writer.close()

    def generer_rapport_economique(self, resultats_optimisation):
        couts, total_investissement = self.calculer_couts_investissement()
        economies, total_economies = self.calculer_economies_annuelles(resultats_optimisation)

        van = self.calculer_van(total_investissement, total_economies)
        tri = self.calculer_tri(total_investissement, total_economies)
        temps_retour = total_investissement / total_economies if total_economies != 0 else None

        rapport = {
            'date_analyse': datetime.now().strftime('%Y-%m-%d'),
            'investissement': {
                'detail': couts,
                'total': total_investissement
            },
            'economies_annuelles': {
                'detail': economies,
                'total': total_economies
            },
            'indicateurs_rentabilite': {
                'VAN': van,
                'TRI': tri,
                'temps_retour': temps_retour
            },
            'analyse_sensibilite': self.analyse_sensibilite(total_investissement, total_economies)
        }

        self.exporter_rapport_excel(rapport)
        return rapport


# ------------------------------------------------------------------------
# (Optionnel) Vous pourriez mettre ici du code de test si vous voulez
# lancer "optimisation.py" directement, mais si vous utilisez gui.py,
# ce n'est pas nécessaire.
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("Ceci est 'optimisation.py'. Utilisez 'gui.py' pour l'IHM ou appelez les méthodes ici si besoin.")

