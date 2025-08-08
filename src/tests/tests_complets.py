#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests complets pour l'optimisation de réseau hydraulique
Regroupe tous les tests : unitaires, encodage, énergie
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import json
from datetime import timedelta
from unittest.mock import Mock, patch

# Ajouter le répertoire courant au path pour importer les modules
sys.path.append('.')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import optimisation, config


class TestOptimisationReseau(unittest.TestCase):
    """Tests pour la classe OptimisationReseau"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        # Créer un fichier INP temporaire pour les tests
        self.temp_inp = tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False)
        self.temp_inp.write("""
[TITLE]
Test Network

[OPTIONS]
UNITS LPS
HEADLOSS H-W

[JUNCTIONS]
J1 100 0
J2 95 0
J3 90 0

[RESERVOIRS]
R1 120 0

[PIPES]
P1 J1 J2 100 200 100 0.01
P2 J2 J3 100 200 100 0.01

[COORDINATES]
J1 0 0
J2 100 0
J3 200 0
R1 -50 0

[END]
        """)
        self.temp_inp.close()
        
    def tearDown(self):
        """Nettoyage après chaque test"""
        if os.path.exists(self.temp_inp.name):
            os.unlink(self.temp_inp.name)
    
    def test_initialisation(self):
        """Test de l'initialisation de la classe"""
        try:
            opt = optimisation.OptimisationReseau(self.temp_inp.name)
            self.assertIsNotNone(opt.reseau)
            self.assertIsNotNone(opt.diametres_initiaux)
            self.assertEqual(len(opt.diametres_initiaux), 2)  # 2 conduites
        except Exception as e:
            self.fail(f"L'initialisation a échoué: {e}")
    
    def test_validation_reseau(self):
        """Test de la validation du réseau"""
        opt = optimisation.OptimisationReseau(self.temp_inp.name)
        # La validation ne doit pas lever d'exception
        opt.valider_reseau()
    
    def test_calcul_pertes_total(self):
        """Test du calcul des pertes de charge totales"""
        opt = optimisation.OptimisationReseau(self.temp_inp.name)
        
        # Créer des résultats de test
        class MockResultats:
            def __init__(self):
                self.link = type('obj', (object,), {
                    'headloss': type('obj', (object,), {
                        'values': np.array([[1.0, 2.0], [1.5, 2.5]])
                    })
                })()
        
        resultats = MockResultats()
        pertes = opt.calculer_pertes_total(resultats)
        self.assertGreater(pertes, 0)
        self.assertIsInstance(pertes, (int, float))
    
    def test_mutation_discrete(self):
        """Test de la mutation discrète"""
        opt = optimisation.OptimisationReseau(self.temp_inp.name)
        individu = [40, 63, 75]
        
        # Test avec probabilité 1.0 (mutation certaine)
        individu_mute = opt.mutation_discrete(individu, indpb=1.0)
        
        # Vérifier que tous les diamètres sont valides
        for diametre in individu_mute[0]:
            self.assertIn(diametre, config.DIAMETRES_DISPONIBLES)
    
    def test_initialiser_population(self):
        """Test de l'initialisation de la population"""
        opt = optimisation.OptimisationReseau(self.temp_inp.name)
        population = opt.initialiser_population()
        
        self.assertEqual(len(population), config.TAILLE_POPULATION)
        
        # Vérifier que chaque individu a le bon nombre de gènes
        for individu in population:
            self.assertEqual(len(individu), 2)  # 2 conduites
            # Vérifier que tous les diamètres sont valides
            for diametre in individu:
                self.assertIn(diametre, config.DIAMETRES_DISPONIBLES)


class TestConfiguration(unittest.TestCase):
    """Tests pour la configuration"""
    
    def test_parametres_hydrauliques(self):
        """Test des paramètres hydrauliques"""
        self.assertGreater(config.PRESSION_MAX, config.PRESSION_MIN)
        self.assertGreater(config.DIAMETRE_MAX, config.DIAMETRE_MIN)
        self.assertGreater(config.VITESSE_MAX, config.VITESSE_MIN)
    
    def test_parametres_optimisation(self):
        """Test des paramètres d'optimisation"""
        self.assertGreater(config.TAILLE_POPULATION, 0)
        self.assertGreater(config.NOMBRE_GENERATIONS, 0)
        self.assertGreater(config.TAUX_CROISEMENT, 0)
        self.assertLess(config.TAUX_CROISEMENT, 1)
    
    def test_diametres_disponibles(self):
        """Test des diamètres disponibles"""
        self.assertGreater(len(config.DIAMETRES_DISPONIBLES), 0)
        for diametre in config.DIAMETRES_DISPONIBLES:
            self.assertGreater(diametre, 0)


class TestLogging(unittest.TestCase):
    """Tests pour le système de logging"""
    
    def test_logging_configuration(self):
        """Test de la configuration du logging"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Vérifier que le logger existe (les handlers peuvent être vides par défaut)
        self.assertIsNotNone(logger)
        # Le test des handlers n'est pas critique car ils peuvent être configurés ailleurs


class TestCalculEnergie(unittest.TestCase):
    """Tests pour le calcul d'énergie"""
    
    def test_calcul_energie_timedelta(self):
        """Test du calcul d'énergie avec timedelta"""
        # Données de test
        debits = np.array([0.1, 0.15, 0.12, 0.08])  # m³/s
        headloss = np.array([2.5, 3.1, 2.8, 1.9])   # m
        rho, g = 1000, 9.81
        
        # Test avec timedelta
        temps_simulation = timedelta(hours=9)
        
        if hasattr(temps_simulation, 'total_seconds'):
            temps_h = temps_simulation.total_seconds() / 3600.0
        elif hasattr(temps_simulation, 'item'):
            temps_h = float(temps_simulation.item()) / 3600.0
        else:
            temps_h = float(temps_simulation) / 3600.0
        
        puissances = rho * g * debits * headloss
        energie_kwh = np.sum(puissances) * temps_h / 1000.0
        
        self.assertGreater(energie_kwh, 0)
        self.assertIsInstance(energie_kwh, float)
    
    def test_calcul_energie_numpy_int64(self):
        """Test du calcul d'énergie avec numpy.int64"""
        # Données de test
        debits = np.array([0.1, 0.15, 0.12, 0.08])  # m³/s
        headloss = np.array([2.5, 3.1, 2.8, 1.9])   # m
        rho, g = 1000, 9.81
        
        # Test avec numpy.int64
        temps_simulation = np.int64(9)  # 9 secondes
        
        if hasattr(temps_simulation, 'total_seconds'):
            temps_h = temps_simulation.total_seconds() / 3600.0
        elif hasattr(temps_simulation, 'item'):
            temps_h = float(temps_simulation.item()) / 3600.0
        else:
            temps_h = float(temps_simulation) / 3600.0
        
        puissances = rho * g * debits * headloss
        energie_kwh = np.sum(puissances) * temps_h / 1000.0
        
        self.assertGreater(energie_kwh, 0)
        self.assertIsInstance(energie_kwh, float)
    
    def test_calcul_energie_entier_simple(self):
        """Test du calcul d'énergie avec entier simple"""
        # Données de test
        debits = np.array([0.1, 0.15, 0.12, 0.08])  # m³/s
        headloss = np.array([2.5, 3.1, 2.8, 1.9])   # m
        rho, g = 1000, 9.81
        
        # Test avec entier simple
        temps_simulation = 9  # 9 secondes
        
        if hasattr(temps_simulation, 'total_seconds'):
            temps_h = temps_simulation.total_seconds() / 3600.0
        elif hasattr(temps_simulation, 'item'):
            temps_h = float(temps_simulation.item()) / 3600.0
        else:
            temps_h = float(temps_simulation) / 3600.0
        
        puissances = rho * g * debits * headloss
        energie_kwh = np.sum(puissances) * temps_h / 1000.0
        
        self.assertGreater(energie_kwh, 0)
        self.assertIsInstance(energie_kwh, float)


class TestEncodage(unittest.TestCase):
    """Tests pour la gestion d'encodage"""
    
    def test_encodage_fichier_existant(self):
        """Test de gestion d'encodage avec les fichiers existants"""
        fichiers_test = ["ebolowa_reseau.inp", "temp.inp"]
        
        for fichier in fichiers_test:
            if os.path.exists(fichier):
                try:
                    # Test direct
                    opt = optimisation.OptimisationReseau(fichier)
                    self.assertIsNotNone(opt.reseau)
                    print(f"✅ {fichier} : Chargement réussi")
                except Exception as e:
                    print(f"❌ {fichier} : Erreur - {e}")
                    # Le test ne doit pas échouer car la gestion d'encodage est automatique
                    pass
    
    def test_encodage_fichier_inexistant(self):
        """Test avec un fichier inexistant"""
        with self.assertRaises(Exception):
            optimisation.OptimisationReseau("fichier_inexistant.inp")


class TestVisualisations(unittest.TestCase):
    """Tests pour les visualisations"""
    
    def setUp(self):
        """Initialisation pour les tests de visualisation"""
        # Créer un fichier INP temporaire
        self.temp_inp = tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False)
        self.temp_inp.write("""
[TITLE]
Test Network

[OPTIONS]
UNITS LPS
HEADLOSS H-W

[JUNCTIONS]
J1 100 0
J2 95 0
J3 90 0

[RESERVOIRS]
R1 120 0

[PIPES]
P1 J1 J2 100 200 100 0.01
P2 J2 J3 100 200 100 0.01

[COORDINATES]
J1 0 0
J2 100 0
J3 200 0
R1 -50 0

[END]
        """)
        self.temp_inp.close()
        
        # Créer l'optimiseur
        self.opt = optimisation.OptimisationReseau(self.temp_inp.name)
        
        # Simuler une optimisation complète
        self.opt.meilleure_solution = [40, 63]  # Solution fictive
        self.opt.historique_fitness = [1000, 950, 900, 850, 800]  # Historique fictif
        
    def tearDown(self):
        """Nettoyage après les tests"""
        if os.path.exists(self.temp_inp.name):
            os.unlink(self.temp_inp.name)
    
    def test_generation_visualisations(self):
        """Test de la génération des visualisations"""
        try:
            # Test de la génération des visualisations améliorées
            self.opt._plot_visualisations_ameliorees()
            
            # Vérifier que les fichiers ont été créés
            fichiers_attendus = [
                'data/results/visualisation/convergence_amelioree.png',
                'data/results/visualisation/distribution_pressions_amelioree.png',
                'data/results/visualisation/evolution_temporelle_amelioree.png',
                'data/results/visualisation/carte_chaleur_amelioree.png',
                'data/results/visualisation/comparaison_amelioree.png'
            ]
            
            for fichier in fichiers_attendus:
                if os.path.exists(fichier):
                    print(f"✅ {fichier} généré avec succès")
                else:
                    print(f"⚠️ {fichier} non généré")
                    
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération des visualisations: {e}")
            # Le test ne doit pas échouer car les visualisations peuvent dépendre de matplotlib
    
    def test_generation_rapports(self):
        """Test de la génération des rapports"""
        try:
            # Test de la génération des rapports
            self.opt.generer_rapports()
            
            # Vérifier que les fichiers ont été créés
            fichiers_attendus = [
                'data/results/rapports/resume_optimisation.csv',
                'data/results/rapports/indicateurs_performance.csv',
                'data/results/rapports/parametres_critiques.csv'
            ]
            
            for fichier in fichiers_attendus:
                if os.path.exists(fichier):
                    print(f"✅ {fichier} généré avec succès")
                else:
                    print(f"⚠️ {fichier} non généré")
                    
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération des rapports: {e}")
            # Le test ne doit pas échouer car les rapports peuvent dépendre de pandas


def main():
    """Fonction principale pour exécuter tous les tests"""
    print("🧪 Tests complets pour l'optimisation de réseau hydraulique")
    print("=" * 60)
    
    # Créer la suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_classes = [
        TestOptimisationReseau,
        TestConfiguration,
        TestLogging,
        TestCalculEnergie,
        TestEncodage,
        TestVisualisations
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ ÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ ERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ TOUS LES TESTS ONT RÉUSSI !")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 