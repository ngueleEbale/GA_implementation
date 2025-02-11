import wntr
import numpy as np
from optimisation import OptimisationReseau
from rapports import ReportGenerator
import logging
from datetime import datetime
import os

# Configuration du logging
def setup_logging():
    """Configure le système de logging"""
    # Création du dossier logs s'il n'existe pas
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Configuration du logger
    log_filename = f'logs/optimisation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def main():
    """Fonction principale d'exécution"""
    
    # Paramètres d'optimisation
    parametres = {
        'population_size': 100,
        'nb_generations': 5,
        'taux_croisement': 0.7,
        'taux_mutation': 0.2,
        'pression_min': 204,  # m (minimum absolu) 20 bar
        'pression_min_service': 500,  # m (minimum de service)
        'pression_max': 612,  # m 60bar
        'vitesse_min': 0.5,  # m/s
        'vitesse_max': 1.5,  # m/s 
        'diametres_disponibles': [
            90, 110, 160, 200, 250, 315, 400  # Diamètres standards en mm
        ],
        # Nouveaux paramètres de pénalités
        'penalite_pression': 1e4,
        'penalite_vitesse': 1e3,
        'penalite_diametre': 1e2
    }

    try:
        # Configuration du logging
        setup_logging()
        logging.info("=== DÉBUT DE L'OPTIMISATION DU RÉSEAU D'EBOLOWA ===")

        # 1. Initialisation
        logging.info("Chargement du réseau...")
        optimiseur = OptimisationReseau('ebolowa_reseau.inp', parametres)
        
        # 2. Exécution de l'optimisation
        logging.info("Démarrage de l'optimisation...")
        progression = lambda x: logging.info(f"Progression : {x}%")
        resultats = optimiseur.executer_optimisation(callback_progression=progression)
        
        # 3. Génération des rapports
        logging.info("Génération des rapports et visualisations...")
        reporter = ReportGenerator(optimiseur, resultats)
        
        # Création des dossiers de sortie si nécessaire
        for dossier in ['rapports', 'visualisations', 'resultats']:
            if not os.path.exists(dossier):
                os.makedirs(dossier)
        
        # Génération des différents rapports
        reporter.generer_visualisations()
        reporter.generer_rapports_csv()
        reporter.generer_rapport_json()
        reporter.generer_analyse_economique()
        
        # 4. Affichage du résumé
        logging.info("\n=== RÉSUMÉ DES RÉSULTATS ===")
        reporter.afficher_resume()
        
        # 5. Sauvegarde des résultats
        #logging.info("Sauvegarde des résultats...")
        #optimiseur.sauvegarder_resultats()
        
        logging.info("=== OPTIMISATION TERMINÉE AVEC SUCCÈS ===")
        
        # Affichage de l'arborescence des fichiers générés
        print("\nFichiers générés :")
        print("├── logs/")
        print("│   └── optimisation_YYYYMMDD_HHMMSS.log")
        print("├── rapports/")
        print("│   ├── resume_optimisation.csv")
        print("│   ├── indicateurs_performance.csv")
        print("│   ├── parametres_critiques.csv")
        print("│   └── analyse_economique.xlsx")
        print("├── visualisations/")
        print("│   ├── convergence.png")
        print("│   ├── distribution_pressions.png")
        print("│   ├── carte_chaleur_pertes.png")
        print("│   ├── comparaison.png")
        print("│   └── front_pareto.png")
        print("└── resultats/")
        print("    └── resultats_optimisation.json")

    except Exception as e:
        logging.error(f"Erreur lors de l'optimisation : {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()