#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface en Ligne de Commande pour l'Optimisation de Réseaux Hydrauliques
==========================================================================

Ce module fournit une interface en ligne de commande complète et professionnelle
pour l'optimisation de réseaux hydrauliques. Il permet d'exécuter les optimisations
mono et multi-objectif depuis le terminal avec une configuration flexible.

L'interface CLI est particulièrement adaptée pour:
- L'automatisation et les scripts batch
- L'exécution sur serveurs sans interface graphique
- L'intégration dans des workflows de calcul
- Les optimisations de longue durée en arrière-plan

Fonctionnalités:
---------------
- Optimisation mono-objectif (pertes de charge)
- Optimisation multi-objectif NSGA-II (pertes, vitesses, pressions)
- Configuration flexible via arguments ou fichier JSON
- Génération automatique de rapports et visualisations
- Support des formats de sortie multiples
- Logging détaillé et gestion d'erreurs robuste

Usage typique:
-------------
python optimisation_cli.py reseau.inp --mono --generations 100
python optimisation_cli.py reseau.inp --multi --config params.json

Author: Équipe d'Optimisation Hydraulique
Version: 3.0
Date: 2025
License: MIT
"""

# Imports système et arguments
import argparse
import sys
import os
import json
from datetime import datetime

# Configuration des chemins d'import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports logique métier
from core import optimisation, config


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Optimisation de réseau hydraulique par algorithme génétique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s ebolowa_reseau.inp --mono
  %(prog)s ebolowa_reseau.inp --multi --generations 50
  %(prog)s ebolowa_reseau.inp --config config.json
        """
    )
    
    # Arguments obligatoires
    parser.add_argument('fichier_inp', 
                       help='Fichier .inp du réseau à optimiser')
    
    # Type d'optimisation
    group_opt = parser.add_mutually_exclusive_group(required=True)
    group_opt.add_argument('--mono', action='store_true',
                          help='Optimisation mono-objectif (coût)')
    group_opt.add_argument('--multi', action='store_true',
                          help='Optimisation multi-objectif (coût + énergie)')
    
    # Paramètres d'optimisation
    parser.add_argument('--generations', type=int, default=config.NOMBRE_GENERATIONS,
                       help=f'Nombre de générations (défaut: {config.NOMBRE_GENERATIONS})')
    parser.add_argument('--population', type=int, default=config.TAILLE_POPULATION,
                       help=f'Taille de population (défaut: {config.TAILLE_POPULATION})')
    parser.add_argument('--croisement', type=float, default=config.TAUX_CROISEMENT,
                       help=f'Taux de croisement (défaut: {config.TAUX_CROISEMENT})')
    parser.add_argument('--mutation', type=float, default=config.TAUX_MUTATION_INDIVIDU,
                       help=f'Taux de mutation (défaut: {config.TAUX_MUTATION_INDIVIDU})')
    
    # Paramètres hydrauliques
    parser.add_argument('--pression-min', type=float, default=config.PRESSION_MIN,
                       help=f'Pression minimale en mCE (défaut: {config.PRESSION_MIN})')
    parser.add_argument('--pression-max', type=float, default=config.PRESSION_MAX,
                       help=f'Pression maximale en mCE (défaut: {config.PRESSION_MAX})')
    parser.add_argument('--diametre-min', type=int, default=config.DIAMETRE_MIN,
                       help=f'Diamètre minimal en mm (défaut: {config.DIAMETRE_MIN})')
    parser.add_argument('--diametre-max', type=int, default=config.DIAMETRE_MAX,
                       help=f'Diamètre maximal en mm (défaut: {config.DIAMETRE_MAX})')
    parser.add_argument('--vitesse-min', type=float, default=config.VITESSE_MIN,
                       help=f'Vitesse minimale en m/s (défaut: {config.VITESSE_MIN})')
    parser.add_argument('--vitesse-max', type=float, default=config.VITESSE_MAX,
                       help=f'Vitesse maximale en m/s (défaut: {config.VITESSE_MAX})')
    
    # Options de sortie
    parser.add_argument('--config', type=str,
                       help='Fichier de configuration JSON à charger')
    parser.add_argument('--save-config', type=str,
                       help='Sauvegarder la configuration dans un fichier JSON')
    parser.add_argument('--no-visualisations', action='store_true',
                       help='Ne pas générer les visualisations')
    parser.add_argument('--no-rapports', action='store_true',
                       help='Ne pas générer les rapports')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Répertoire de sortie (défaut: répertoire courant)')
    
    # Options de debug
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbeux')
    parser.add_argument('--debug', action='store_true',
                       help='Mode debug')
    
    return parser.parse_args()


def charger_configuration(fichier_config):
    """Charge une configuration depuis un fichier JSON"""
    try:
        with open(fichier_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"✅ Configuration chargée depuis {fichier_config}")
        return config_data
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        return None


def sauvegarder_configuration(config_data, fichier_config):
    """Sauvegarde la configuration dans un fichier JSON"""
    try:
        with open(fichier_config, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration sauvegardée dans {fichier_config}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde de la configuration: {e}")


def appliquer_configuration(config_data):
    """Applique une configuration aux paramètres globaux"""
    if not config_data:
        return
    
    # Paramètres d'optimisation
    if 'generations' in config_data:
        optimisation.NOMBRE_GENERATIONS = config_data['generations']
    if 'population' in config_data:
        optimisation.TAILLE_POPULATION = config_data['population']
    if 'croisement' in config_data:
        optimisation.TAUX_CROISEMENT = config_data['croisement']
    if 'mutation' in config_data:
        optimisation.TAUX_MUTATION_INDIVIDU = config_data['mutation']
    
    # Paramètres hydrauliques
    if 'pression_min' in config_data:
        optimisation.PRESSION_MIN = config_data['pression_min']
    if 'pression_max' in config_data:
        optimisation.PRESSION_MAX = config_data['pression_max']
    if 'diametre_min' in config_data:
        optimisation.DIAMETRE_MIN = config_data['diametre_min']
    if 'diametre_max' in config_data:
        optimisation.DIAMETRE_MAX = config_data['diametre_max']
    if 'vitesse_min' in config_data:
        optimisation.VITESSE_MIN = config_data['vitesse_min']
    if 'vitesse_max' in config_data:
        optimisation.VITESSE_MAX = config_data['vitesse_max']


def appliquer_arguments(args):
    """Applique les arguments de ligne de commande aux paramètres"""
    # Paramètres d'optimisation
    optimisation.NOMBRE_GENERATIONS = args.generations
    optimisation.TAILLE_POPULATION = args.population
    optimisation.TAUX_CROISEMENT = args.croisement
    optimisation.TAUX_MUTATION_INDIVIDU = args.mutation
    
    # Paramètres hydrauliques
    optimisation.PRESSION_MIN = args.pression_min
    optimisation.PRESSION_MAX = args.pression_max
    optimisation.DIAMETRE_MIN = args.diametre_min
    optimisation.DIAMETRE_MAX = args.diametre_max
    optimisation.VITESSE_MIN = args.vitesse_min
    optimisation.VITESSE_MAX = args.vitesse_max


def afficher_parametres():
    """Affiche les paramètres actuels"""
    print("\n" + "="*60)
    print("PARAMÈTRES D'OPTIMISATION")
    print("="*60)
    print(f"Taille population: {optimisation.TAILLE_POPULATION}")
    print(f"Nombre générations: {optimisation.NOMBRE_GENERATIONS}")
    print(f"Taux croisement: {optimisation.TAUX_CROISEMENT}")
    print(f"Taux mutation: {optimisation.TAUX_MUTATION_INDIVIDU}")
    
    print("\nPARAMÈTRES HYDRAULIQUES")
    print("-" * 30)
    print(f"Pression min/max: {optimisation.PRESSION_MIN}/{optimisation.PRESSION_MAX} mCE")
    print(f"Diamètre min/max: {optimisation.DIAMETRE_MIN}/{optimisation.DIAMETRE_MAX} mm")
    print(f"Vitesse min/max: {optimisation.VITESSE_MIN}/{optimisation.VITESSE_MAX} m/s")
    print("="*60 + "\n")


def callback_generations(generation):
    """Callback pour afficher la progression"""
    print(f"Génération {generation}/{optimisation.NOMBRE_GENERATIONS} "
          f"({generation/optimisation.NOMBRE_GENERATIONS*100:.1f}%)")


def main():
    """Fonction principale"""
    print("🚀 Optimisation de Réseau Hydraulique - Interface CLI")
    print("=" * 60)
    
    # Parse des arguments
    args = parse_arguments()
    
    # Vérification du fichier INP
    if not os.path.exists(args.fichier_inp):
        print(f"❌ Erreur: Le fichier {args.fichier_inp} n'existe pas")
        sys.exit(1)
    
    # Chargement de la configuration si spécifiée
    if args.config:
        config_data = charger_configuration(args.config)
        if config_data:
            appliquer_configuration(config_data)
    
    # Application des arguments de ligne de commande
    appliquer_arguments(args)
    
    # Affichage des paramètres
    afficher_parametres()
    
    # Création du répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualisation"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "rapports"), exist_ok=True)
    
    try:
        # Initialisation de l'optimiseur
        print("📊 Initialisation de l'optimiseur...")
        opt = optimisation.OptimisationReseau(args.fichier_inp)
        
        # Sauvegarde de la configuration si demandée
        if args.save_config:
            config_data = {
                'generations': optimisation.NOMBRE_GENERATIONS,
                'population': optimisation.TAILLE_POPULATION,
                'croisement': optimisation.TAUX_CROISEMENT,
                'mutation': optimisation.TAUX_MUTATION_INDIVIDU,
                'pression_min': optimisation.PRESSION_MIN,
                'pression_max': optimisation.PRESSION_MAX,
                'diametre_min': optimisation.DIAMETRE_MIN,
                'diametre_max': optimisation.DIAMETRE_MAX,
                'vitesse_min': optimisation.VITESSE_MIN,
                'vitesse_max': optimisation.VITESSE_MAX,
                'date_sauvegarde': datetime.now().isoformat()
            }
            sauvegarder_configuration(config_data, args.save_config)
        
        # Lancement de l'optimisation
        if args.mono:
            print("🎯 Lancement de l'optimisation MONO-OBJECTIF...")
            opt.executer_optimisation(callback=callback_generations)
        else:
            print("🎯 Lancement de l'optimisation MULTI-OBJECTIF...")
            opt.executer_optimisation_multi(callback=callback_generations)
        
        # Génération des visualisations
        if not args.no_visualisations:
            print("📈 Génération des visualisations...")
            opt._plot_visualisations_ameliorees()
        
        # Génération des rapports
        if not args.no_rapports:
            print("📋 Génération des rapports...")
            opt.generer_rapports()
        
        print("\n✅ Optimisation terminée avec succès !")
        print(f"📁 Résultats disponibles dans: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimisation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'optimisation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
