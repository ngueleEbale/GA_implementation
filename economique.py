import pandas as pd
from optimisation import OptimisationReseau as opt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from datetime import datetime

import wntr
import json
import os

class AnalyseEconomique:
    def __init__(self, optimiseur, resultats):
        """
        Initialisation de l'analyse économique
        
        Args:
            optimiseur: Instance de OptimisationReseau
            resultats: Résultats de l'optimisation
        """
        self.optimiseur = optimiseur
        self.resultats = resultats
        self.taux_actualisation = 0.08  # 8% taux d'actualisation
        self.duree_projet = 20  # années
        
    def calculer_couts_investissement(self):
        """Calcul des coûts d'investissement initiaux"""
        # Coûts des conduites
        cout_conduites = 0
        for i, diametre in enumerate(self.resultats['meilleure_solution']):
            nom_conduite = self.optimiseur.reseau.pipe_name_list[i]
            longueur = self.optimiseur.reseau.get_link(nom_conduite).length
            cout_unitaire = self.get_cout_unitaire(diametre)
            cout_conduites += cout_unitaire * longueur
            
        # Autres coûts
        couts = {
            'Conduites': cout_conduites,
            'Installation': cout_conduites * 0.2,  # 20% du coût des conduites
            'Études et ingénierie': cout_conduites * 0.1,  # 10% du coût des conduites
            'Équipements annexes': cout_conduites * 0.15,  # 15% du coût des conduites
            'Imprévus': cout_conduites * 0.1  # 10% du coût des conduites
        }
        
        return couts

    def calculer_couts_exploitation(self):
        """Calcul des coûts d'exploitation annuels"""
        # Simulation pour obtenir les paramètres hydrauliques
        sim = wntr.sim.EpanetSimulator(self.optimiseur.reseau)
        resultats_sim = sim.run_sim()
        
        # Calcul de l'énergie consommée

        energie = opt.calculer_energie(opt,resultats=resultats_sim)
        
        couts = {
            'Énergie': energie * 100,  # 100 FCFA/kWh
            'Maintenance': self.resultats['fitness_finale'] * 0.02,  # 2% du coût d'investissement
            'Personnel': 5000000,  # Coût fixe annuel
            'Produits traitement': 2000000  # Coût fixe annuel
        }
        
        return couts

    def calculer_van(self):
        """Calcul de la Valeur Actuelle Nette"""
        couts_inv = sum(self.calculer_couts_investissement().values())
        couts_exp = sum(self.calculer_couts_exploitation().values())
        
        van = -couts_inv
        for annee in range(1, self.duree_projet + 1):
            van += couts_exp / (1 + self.taux_actualisation)**annee
            
        return van

    def calculer_tri(self):
        """Calcul du Taux de Rentabilité Interne"""
        couts_inv = sum(self.calculer_couts_investissement().values())
        couts_exp = sum(self.calculer_couts_exploitation().values())
        
        def npv(rate):
            return -couts_inv + sum([couts_exp / (1 + rate)**t 
                                for t in range(1, self.duree_projet + 1)])
        
        # Recherche du TRI par dichotomie
        taux_min, taux_max = -0.5, 0.5
        precision = 0.0001
        
        while (taux_max - taux_min) > precision:
            taux = (taux_min + taux_max) / 2
            if npv(taux) > 0:
                taux_min = taux
            else:
                taux_max = taux
                
        return taux

    def get_cout_unitaire(self, diametre):
        """
        Retourne le coût unitaire pour un diamètre donné
        
        Args:
            diametre (float): Diamètre de la conduite en mm
            
        Returns:
            float: Coût unitaire en FCFA/m
        """
        # Coûts unitaires des conduites (FCFA/m) pour le contexte camerounais
        couts_unitaires = {
            100: 10000,    # DN100
            150: 15000,    # DN150
            200: 20000,    # DN200
            250: 25000,    # DN250
            300: 30000,    # DN300
            400: 40000,    # DN400
            500: 50000,    # DN500
            600: 60000,    # DN600
            800: 80000     # DN800
        }
        
        # Si le diamètre n'est pas dans la liste, on prend le plus proche
        if diametre not in couts_unitaires:
            diametres = list(couts_unitaires.keys())
            diametre_proche = min(diametres, key=lambda x: abs(x - diametre))
            return couts_unitaires[diametre_proche]
            
        return couts_unitaires[diametre]

    def generer_rapport(self):
        """Génération du rapport économique complet"""
        try:
            # Calculs économiques
            couts_inv = self.calculer_couts_investissement()
            couts_exp = self.calculer_couts_exploitation()
            van = self.calculer_van()
            tri = self.calculer_tri()
            
            # Création du rapport Excel
            with pd.ExcelWriter('rapports/analyse_economique.xlsx', engine='openpyxl') as writer:
                # Feuille 1 : Résumé
                resume = pd.DataFrame({
                    'Indicateur': [
                        'Coût total d\'investissement (FCFA)',
                        'Coûts d\'exploitation annuels (FCFA)',
                        'VAN (FCFA)',
                        'TRI (%)',
                        'Durée du projet (années)',
                        'Taux d\'actualisation (%)'
                    ],
                    'Valeur': [
                        sum(couts_inv.values()),
                        sum(couts_exp.values()),
                        van,
                        tri * 100,
                        self.duree_projet,
                        self.taux_actualisation * 100
                    ]
                })
                resume.to_excel(writer, sheet_name='Résumé', index=False)
                
                # Feuille 2 : Détail des coûts d'investissement
                pd.DataFrame({
                    'Poste': couts_inv.keys(),
                    'Montant (FCFA)': couts_inv.values()
                }).to_excel(writer, sheet_name='Coûts investissement', index=False)
                
                # Feuille 3 : Détail des coûts d'exploitation
                pd.DataFrame({
                    'Poste': couts_exp.keys(),
                    'Montant annuel (FCFA)': couts_exp.values()
                }).to_excel(writer, sheet_name='Coûts exploitation', index=False)
                
                # Feuille 4 : Flux de trésorerie
                flux = [-sum(couts_inv.values())]  # Année 0
                flux.extend([-sum(couts_exp.values())] * self.duree_projet)  # Années 1 à n
                
                pd.DataFrame({
                    'Année': range(self.duree_projet + 1),
                    'Flux (FCFA)': flux,
                    'Flux actualisé (FCFA)': [f / (1 + self.taux_actualisation)**t 
                                            for t, f in enumerate(flux)]
                }).to_excel(writer, sheet_name='Flux de trésorerie', index=False)

        except Exception as e:
            logging.error(f"Erreur lors de la génération du rapport économique : {str(e)}")
            raise