import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from datetime import datetime
import wntr
import json
import os
from economique import AnalyseEconomique
class ReportGenerator:
    def __init__(self, optimiseur, resultats):
        """
        Initialisation du générateur de rapports
        
        Args:
            optimiseur: Instance de OptimisationReseau
            resultats: Résultats de l'optimisation
        """
        self.optimiseur = optimiseur
        self.resultats = resultats
        self.reseau = optimiseur.reseau
        
        # Création des dossiers de sortie
        for dossier in ['rapports', 'visualisations']:
            if not os.path.exists(dossier):
                os.makedirs(dossier)

    def generer_visualisations(self):
        """Génère toutes les visualisations"""
        logging.info("Génération des visualisations...")
        
        self._plot_convergence()
        self._plot_distribution_pressions()
        self._plot_carte_chaleur_pertes()
        self._plot_comparaison()
        self._plot_evolution_temporelle()

    def _plot_convergence(self):
        """Graphique amélioré de convergence de l'algorithme génétique"""
        metriques = self.resultats['metriques_convergence']
        
        plt.figure(figsize=(12, 8))
        plt.plot(metriques['generation'], metriques['min_fitness'], 
                label='Minimum', color='green')
        plt.plot(metriques['generation'], metriques['avg_fitness'], 
                label='Moyenne', color='blue')
        plt.fill_between(metriques['generation'],
                        np.array(metriques['avg_fitness']) - np.array(metriques['std_fitness']),
                        np.array(metriques['avg_fitness']) + np.array(metriques['std_fitness']),
                        alpha=0.2, color='blue')
        
        plt.title('Convergence de l\'optimisation')
        plt.xlabel('Génération')
        plt.ylabel('Fitness (coût + pénalités)')
        plt.grid(True)
        plt.legend()
        plt.savefig('visualisations/convergence.png')
        plt.close()

    def _plot_distribution_pressions(self):
        """Distribution des pressions dans le réseau"""
        # Simulation avec la solution optimale
        for i, diametre in enumerate(self.resultats['meilleure_solution']):
            nom_conduite = self.reseau.pipe_name_list[i]
            self.reseau.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats_sim = sim.run_sim()
        pressions = resultats_sim.node['pressure'].mean()

        plt.figure(figsize=(12, 6))
        
        # Histogramme
        plt.subplot(121)
        sns.histplot(pressions, bins=20)
        plt.axvline(self.optimiseur.parametres['pression_min'], 
                   color='r', linestyle='--', label='Min')
        plt.axvline(self.optimiseur.parametres['pression_max'], 
                   color='r', linestyle='--', label='Max')
        plt.title('Distribution des pressions')
        plt.xlabel('Pression (m)')
        plt.ylabel('Nombre de nœuds')
        plt.legend()

        # Boxplot
        plt.subplot(122)
        plt.boxplot(pressions)
        plt.title('Boxplot des pressions')
        plt.ylabel('Pression (m)')

        plt.tight_layout()
        plt.savefig('visualisations/distribution_pressions.png')
        plt.close()

    def _plot_carte_chaleur_pertes(self):
        """Carte de chaleur des pertes de charge"""
        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats_sim = sim.run_sim()
        
        # Calcul des pertes de charge
        pertes = resultats_sim.link['headloss'].mean()
        
        # Création de la carte
        plt.figure(figsize=(15, 10))
        
        # Récupération des coordonnées
        x_coords = []
        y_coords = []
        colors = []
        
        for pipe_name in self.reseau.pipe_name_list:
            pipe = self.reseau.get_link(pipe_name)
            start_node = self.reseau.get_node(pipe.start_node_name)
            end_node = self.reseau.get_node(pipe.end_node_name)
            
            x_coords.extend([start_node.coordinates[0], end_node.coordinates[0]])
            y_coords.extend([start_node.coordinates[1], end_node.coordinates[1]])
            colors.extend([pertes[pipe_name], pertes[pipe_name]])

        plt.scatter(x_coords, y_coords, c=colors, cmap='YlOrRd', s=50)
        plt.colorbar(label='Pertes de charge (m)')
        
        # Connexions entre les nœuds
        for pipe_name in self.reseau.pipe_name_list:
            pipe = self.reseau.get_link(pipe_name)
            start_node = self.reseau.get_node(pipe.start_node_name)
            end_node = self.reseau.get_node(pipe.end_node_name)
            plt.plot([start_node.coordinates[0], end_node.coordinates[0]],
                    [start_node.coordinates[1], end_node.coordinates[1]],
                    'gray', alpha=0.3)

        plt.title('Carte des pertes de charge')
        plt.xlabel('Coordonnée X')
        plt.ylabel('Coordonnée Y')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('visualisations/carte_chaleur_pertes.png')
        plt.close()

    def _plot_comparaison(self):
        """Comparaison avant/après optimisation"""
        try:
            # Simulation état initial en utilisant le fichier INP original
            reseau_initial = wntr.network.WaterNetworkModel(self.optimiseur.fichier_inp)
            sim_initiale = wntr.sim.EpanetSimulator(reseau_initial)
            resultats_initiaux = sim_initiale.run_sim()

            # Simulation état optimisé
            reseau_optimal = wntr.network.WaterNetworkModel(self.optimiseur.fichier_inp)
            for i, diametre in enumerate(self.resultats['meilleure_solution']):
                nom_conduite = reseau_optimal.pipe_name_list[i]
                reseau_optimal.get_link(nom_conduite).diameter = diametre/1000
                
            sim_optimale = wntr.sim.EpanetSimulator(reseau_optimal)
            resultats_optimaux = sim_optimale.run_sim()

            # Création du graphique
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Pressions
            ax1.hist(resultats_initiaux.node['pressure'].mean(), bins=20, alpha=0.5, 
                    label='Initial')
            ax1.hist(resultats_optimaux.node['pressure'].mean(), bins=20, alpha=0.5, 
                    label='Optimisé')
            ax1.set_title('Distribution des pressions')
            ax1.set_xlabel('Pression (m)')
            ax1.legend()

            # 2. Vitesses
            ax2.hist(resultats_initiaux.link['velocity'].mean(), bins=20, alpha=0.5, 
                    label='Initial')
            ax2.hist(resultats_optimaux.link['velocity'].mean(), bins=20, alpha=0.5, 
                    label='Optimisé')
            ax2.set_title('Distribution des vitesses')
            ax2.set_xlabel('Vitesse (m/s)')
            ax2.legend()

            # 3. Pertes de charge
            ax3.hist(resultats_initiaux.link['headloss'].mean(), bins=20, alpha=0.5, 
                    label='Initial')
            ax3.hist(resultats_optimaux.link['headloss'].mean(), bins=20, alpha=0.5, 
                    label='Optimisé')
            ax3.set_title('Distribution des pertes de charge')
            ax3.set_xlabel('Pertes (m)')
            ax3.legend()

            # 4. Améliorations en pourcentage
            ameliorations = {
                'Pression moyenne': ((resultats_optimaux.node['pressure'].mean().mean() - 
                                    resultats_initiaux.node['pressure'].mean().mean()) / 
                                resultats_initiaux.node['pressure'].mean().mean() * 100),
                'Vitesse moyenne': ((resultats_optimaux.link['velocity'].mean().mean() - 
                                resultats_initiaux.link['velocity'].mean().mean()) / 
                                resultats_initiaux.link['velocity'].mean().mean() * 100),
                'Pertes moyennes': ((resultats_optimaux.link['headloss'].mean().mean() - 
                                resultats_initiaux.link['headloss'].mean().mean()) / 
                                resultats_initiaux.link['headloss'].mean().mean() * 100)
            }
            
            ax4.bar(ameliorations.keys(), ameliorations.values())
            ax4.set_title('Améliorations relatives (%)')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig('visualisations/comparaison.png')
            plt.close()

        except Exception as e:
            logging.error(f"Erreur lors de la création du graphique de comparaison: {str(e)}")
            raise
        
    def _plot_evolution_temporelle(self):
        """Évolution temporelle des paramètres clés"""
        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats_sim = sim.run_sim()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Pressions
        resultats_sim.node['pressure'].plot(ax=ax1)
        ax1.set_title('Évolution des pressions')
        ax1.set_xlabel('Temps')
        ax1.set_ylabel('Pression (m)')
        
        # Vitesses
        resultats_sim.link['velocity'].plot(ax=ax2)
        ax2.set_title('Évolution des vitesses')
        ax2.set_xlabel('Temps')
        ax2.set_ylabel('Vitesse (m/s)')
        
        # Débits
        resultats_sim.link['flowrate'].plot(ax=ax3)
        ax3.set_title('Évolution des débits')
        ax3.set_xlabel('Temps')
        ax3.set_ylabel('Débit (m³/s)')
        
        plt.tight_layout()
        plt.savefig('visualisations/evolution_temporelle.png')
        plt.close()

    def generer_rapports_csv(self):
        """Génère tous les rapports CSV"""
        self._export_resume_optimisation()
        self._export_indicateurs_performance()
        self._export_parametres_critiques()

    def _export_resume_optimisation(self):
        """Export du résumé de l'optimisation"""
        resume = {
            'Paramètre': [
                'Coût total (FCFA)',
                'Nombre de générations',
                'Fitness finale',
                'Temps de calcul (s)',
                'Nombre de solutions évaluées'
            ],
            'Valeur': [
                self.resultats['fitness_finale'],
                self.resultats['nb_generations'],
                self.resultats['fitness_finale'],
                0,  # À implémenter : temps de calcul
                self.resultats['nb_generations'] * 
                self.optimiseur.parametres['population_size']
            ]
        }
        
        df_resume = pd.DataFrame(resume)
        df_resume.to_csv('rapports/resume_optimisation.csv', index=False)

    def _export_indicateurs_performance(self):
        """Export des indicateurs de performance"""
        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats_sim = sim.run_sim()
        
        indicateurs = {
            'Indicateur': [
                'Pression moyenne (m)',
                'Pression minimale (m)',
                'Pression maximale (m)',
                'Vitesse moyenne (m/s)',
                'Vitesse minimale (m/s)',
                'Vitesse maximale (m/s)',
                'Pertes totales (m)',
                'Coût total (FCFA)'
            ],
            'Valeur': [
                resultats_sim.node['pressure'].mean().mean(),
                resultats_sim.node['pressure'].min().min(),
                resultats_sim.node['pressure'].max().max(),
                resultats_sim.link['velocity'].mean().mean(),
                resultats_sim.link['velocity'].min().min(),
                resultats_sim.link['velocity'].max().max(),
                resultats_sim.link['headloss'].sum().sum(),
                self.resultats['fitness_finale']
            ]
        }
        
        df_indicateurs = pd.DataFrame(indicateurs)
        df_indicateurs.to_csv('rapports/indicateurs_performance.csv', index=False)

    def _export_parametres_critiques(self):
        """Export des paramètres critiques"""
        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats_sim = sim.run_sim()
        
        # Points critiques : pressions
        pressions = resultats_sim.node['pressure'].mean()
        points_critiques = []
        
        for noeud in self.reseau.node_name_list:
            pression = pressions[noeud]
            if (pression < self.optimiseur.parametres['pression_min'] or 
                pression > self.optimiseur.parametres['pression_max']):
                points_critiques.append({
                    'Element': noeud,
                    'Type': 'Nœud',
                    'Paramètre': 'Pression',
                    'Valeur': float(pression),
                    'Limite min': self.optimiseur.parametres['pression_min'],
                    'Limite max': self.optimiseur.parametres['pression_max']
                })
        
        # Points critiques : vitesses
        vitesses = resultats_sim.link['velocity'].mean()
        
        for conduite in self.reseau.pipe_name_list:
            vitesse = vitesses[conduite]
            if (vitesse < self.optimiseur.parametres['vitesse_min'] or 
                vitesse > self.optimiseur.parametres['vitesse_max']):
                points_critiques.append({
                    'Element': conduite,
                    'Type': 'Conduite',
                    'Paramètre': 'Vitesse',
                    'Valeur': float(vitesse),
                    'Limite min': self.optimiseur.parametres['vitesse_min'],
                    'Limite max': self.optimiseur.parametres['vitesse_max']
                })
        
        df_critiques = pd.DataFrame(points_critiques)
        df_critiques.to_csv('rapports/parametres_critiques.csv', index=False)

    def generer_analyse_economique(self):
        """Génère l'analyse économique"""
        analyse = AnalyseEconomique(self.optimiseur, self.resultats)
        analyse.generer_rapport()

    def afficher_resume(self):
        """Affiche un résumé des résultats"""
        print("\nRÉSUMÉ DE L'OPTIMISATION")
        print("-" * 50)
        print(f"Coût final : {self.resultats['fitness_finale']:,.2f} FCFA")
        print(f"Nombre de générations : {self.resultats['nb_generations']}")
        print(f"Nombre de solutions évaluées : {self.resultats['nb_generations'] * self.optimiseur.parametres['population_size']}")
        #print
    
    def generer_rapport_json(self):
        """Génère un rapport détaillé au format JSON"""
        try:
            # Simulation avec la solution optimale
            reseau_optimal = wntr.network.WaterNetworkModel(self.optimiseur.fichier_inp)
            for i, diametre in enumerate(self.resultats['meilleure_solution']):
                nom_conduite = reseau_optimal.pipe_name_list[i]
                reseau_optimal.get_link(nom_conduite).diameter = diametre/1000
                
            sim = wntr.sim.EpanetSimulator(reseau_optimal)
            resultats_sim = sim.run_sim()

            # Calcul des statistiques hydrauliques
            pressions = resultats_sim.node['pressure'].mean()
            vitesses = resultats_sim.link['velocity'].mean()
            pertes = resultats_sim.link['headloss'].sum().sum()

            # Création du rapport
            rapport = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parametres_optimisation": self.optimiseur.parametres,
                "resultats": {
                    "meilleure_solution": {
                        "diametres": {
                            pipe: float(diametre) 
                            for pipe, diametre in zip(
                                self.optimiseur.reseau.pipe_name_list,
                                self.resultats['meilleure_solution']
                            )
                        },
                        "fitness": float(self.resultats['fitness_finale'])
                    },
                    "statistiques_hydrauliques": {
                        "pression_moyenne": float(pressions.mean()),
                        "pression_min": float(pressions.min()),
                        "pression_max": float(pressions.max()),
                        "vitesse_moyenne": float(vitesses.mean()),
                        "vitesse_min": float(vitesses.min()),
                        "vitesse_max": float(vitesses.max()),
                        "pertes_totales": float(pertes)
                    },
                    "performance": {
                        "nb_generations": self.resultats['nb_generations'],
                        "historique_fitness": [float(f) for f in self.resultats['historique_fitness']]
                    }
                }
            }

            # Sauvegarde du rapport
            os.makedirs('resultats', exist_ok=True)
            with open('resultats/rapport_optimisation.json', 'w', encoding='utf-8') as f:
                json.dump(rapport, f, indent=4, ensure_ascii=False)

            logging.info("Rapport JSON généré avec succès")

        except Exception as e:
            logging.error(f"Erreur lors de la génération du rapport JSON : {str(e)}")
            raise

   