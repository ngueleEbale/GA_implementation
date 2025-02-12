# -*- coding: utf-8 -*-

# Import des bibliothèques nécessaires
import copy
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import json
from datetime import datetime

# Définition des constantes
PRESSION_MIN = 20   # 20 Pression minimale en bar
PRESSION_MAX = 612  # 60 Pression maximale en bar
DIAMETRE_MIN = 40  # Diamètre minimal en mm
DIAMETRE_MAX = 250 # Diamètre maximal en mm
VITESSE_MIN = 0.5  # Vitesse minimale en m/s
VITESSE_MAX = 1.5  # Vitesse maximale en m/s

# Paramètres de l'algorithme génétique
TAILLE_POPULATION = 100
NOMBRE_GENERATIONS = 5
TAUX_CROISEMENT = 0.8
TAUX_MUTATION = 0.2

# Coûts spécifiques au Cameroun (à adapter selon les données réelles)
COUT_DIAMETRE = {
    40: 5000,   # FCFA par mètre
    63: 7000,
    75: 8500,
    90: 10000,
    110: 12000,
    160: 15000,
    200: 18000,
    250: 22000
}

class OptimisationReseau:
    def __init__(self, fichier_inp):
        """Initialisation de l'optimiseur de réseau"""
        try:
            # Initialisation des attributs de classe
            self.meilleure_solution = None
            self.historique_fitness = []
            self.solutions_pareto = []
            
            # Chargement direct du réseau
            print("Chargement du réseau hydraulique...")
            self.reseau = wntr.network.WaterNetworkModel(fichier_inp)
            
            # Sauvegarde des diamètres initiaux
            self.diametres_initiaux = []
            for nom_conduite in self.reseau.pipe_name_list:
                self.diametres_initiaux.append(self.reseau.get_link(nom_conduite).diameter)
            
            print(f"Réseau chargé avec succès:")
            print(f"- {len(self.reseau.node_name_list)} nœuds")
            print(f"- {len(self.reseau.link_name_list)} conduites")
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation: {str(e)}")
            raise

    def initialiser_population(self):
        """Création de la population initiale"""
        population = []
        nb_conduites = len(self.reseau.pipe_name_list)
        
        for _ in range(TAILLE_POPULATION):
            # Génération aléatoire des diamètres pour chaque conduite
            individu = [np.random.choice(list(COUT_DIAMETRE.keys())) 
                        for _ in range(nb_conduites)]
            population.append(individu)
        
        return population

    def evaluer_solution(self, solution):
        """
        Évaluation d'une solution
        Formule: Score = Coût total + Pénalités contraintes
        """
        try:
            # Application des diamètres de la solution sur le réseau existant
            for i, diametre in enumerate(solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000  # conversion mm -> m

            # Simulation hydraulique
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()
            
            # Calcul des coûts
            cout_total = self.calculer_cout_total(solution)
            
            # Vérification des contraintes
            penalites = self.calculer_penalites(resultats)
            
            # Réinitialisation des diamètres à leur valeur d'origine
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
            
            score_final = cout_total + penalites
            return (score_final,)
            
        except Exception as e:
            print(f"Erreur critique lors de l'évaluation: {str(e)}")
            return (float('inf'),)

    def calculer_cout_total(self, solution):
        """Calcul du coût total de la solution"""
        cout = 0
        for i, diametre in enumerate(solution):
            nom_conduite = self.reseau.pipe_name_list[i]
            longueur = self.reseau.get_link(nom_conduite).length
            cout += COUT_DIAMETRE[diametre] * longueur
        return cout

    def calculer_penalites(self, resultats):
        """
        Calcul des pénalités pour violation des contraintes
        Retourne un score de pénalité
        """
        penalites = 0
        
        # Pénalités pour les pressions
        pressions = resultats.node['pressure'].values
        pressions_mean = pressions.mean(axis=0)  # Moyenne sur la période
        for p in pressions_mean:
            if p < PRESSION_MIN:
                penalites += (PRESSION_MIN - p) * 1e6
            elif p > PRESSION_MAX:
                penalites += (p - PRESSION_MAX) * 1e6
        
        # Pénalités pour les vitesses
        vitesses = resultats.link['velocity'].values
        vitesses_mean = vitesses.mean(axis=0)  # Moyenne sur la période
        for v in vitesses_mean:
            if v < VITESSE_MIN:
                penalites += (VITESSE_MIN - v) * 1e6
            elif v > VITESSE_MAX:
                penalites += (v - VITESSE_MAX) * 1e6
                
        return penalites

    def executer_optimisation(self, callback=None):
        """Exécution principale de l'optimisation (mono-objectif simplifié)"""
        try:
            # Création de la fitness et de l'Individual dans DEAP (si pas déjà fait)
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            
            # Initialisation des opérateurs génétiques
            toolbox.register("individual", tools.initIterate, creator.Individual,
                            lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Exécution de l'algorithme génétique
            population = toolbox.population(n=TAILLE_POPULATION)

            # ─────────────────────────────────────────────────────
            # 1) ÉVALUATION DE LA POPULATION INITIALE (Important)
            # ─────────────────────────────────────────────────────
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Maintenant, tous les individus initiaux ont une fitness

            meilleur_score_global = float('inf')
            generations_sans_amelioration = 0
            
            print("\nDébut de l'optimisation génétique")
            print("-" * 50)
            
            for gen in range(NOMBRE_GENERATIONS):
                # 2) Reproduction (mutation & croisement)
                offspring = algorithms.varAnd(population, toolbox, 
                                            cxpb=TAUX_CROISEMENT, 
                                            mutpb=TAUX_MUTATION)
                
                # 3) Évaluation des individus invalides (dans l'offspring)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # 4) Sélection de la nouvelle population
                population = toolbox.select(offspring + population, k=len(population))
                
                # 5) Statistiques de la génération
                fits = [ind.fitness.values[0] for ind in population]
                score_min = min(fits)
                score_moy = sum(fits) / len(fits)
                
                # Contrôle de l'amélioration
                if score_min < meilleur_score_global:
                    meilleur_score_global = score_min
                    generations_sans_amelioration = 0
                    amelioration = "⭐"
                else:
                    generations_sans_amelioration += 1
                    amelioration = "  "
                
                print(f"Génération {gen+1:3d} | Min: {score_min:10.2f} | Moy: {score_moy:10.2f} {amelioration}")
                self.historique_fitness.append(score_min)
                
                # Callback éventuel (pour affichage de progression)
                if callback:
                    callback(gen)
                
                # Critère d'arrêt anticipé (optionnel)
                if generations_sans_amelioration >= 10:
                    print("\nArrêt anticipé : pas d'amélioration depuis 10 générations.")
                    break

            print("-" * 50)
            print(f"Meilleur score final: {meilleur_score_global:.2f}")
            
            # Sélection de la meilleure solution
            self.meilleure_solution = tools.selBest(population, k=1)[0]

        except Exception as e:
            print(f"\nErreur pendant l'optimisation: {str(e)}")
            raise


    # =========================================================================
    # Méthodes de visualisation
    # =========================================================================
    def _plot_convergence(self):
        """Graphique de convergence de l'algorithme génétique"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.historique_fitness, marker='o')
        plt.title('Convergence de l\'algorithme génétique')
        plt.xlabel('Génération')
        plt.ylabel('Meilleure fitness (score)')
        plt.grid(True)
        plt.savefig('convergence.png')
        plt.close()

    def _plot_distribution_pressions(self):
        """Distribution des pressions dans le réseau, avec la meilleure solution"""
        if not self.meilleure_solution:
            return
        
        try:
            # Application des diamètres de la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            # Simulation
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()
            
            # Récupération des pressions (moyenne)
            pressions = resultats.node['pressure'].mean()
            
            plt.figure(figsize=(12, 6))
            # Histogramme
            plt.subplot(121)
            plt.hist(pressions, bins=20, color='skyblue', edgecolor='black')
            plt.axvline(PRESSION_MIN, color='r', linestyle='--', label=f'Min ({PRESSION_MIN} bar)')
            plt.axvline(PRESSION_MAX, color='r', linestyle='--', label=f'Max ({PRESSION_MAX} bar)')
            plt.title('Distribution des pressions')
            plt.xlabel('Pression (bar)')
            plt.ylabel('Nombre de nœuds')
            plt.legend()
            
            # Boxplot
            plt.subplot(122)
            plt.boxplot(pressions)
            plt.title('Boxplot des pressions')
            plt.ylabel('Pression (bar)')
            
            plt.tight_layout()
            plt.savefig('distribution_pressions.png')
            plt.close()
            
            # Réinitialisation
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

        except Exception as e:
            print(f"Erreur lors de la création du graphique de distribution des pressions: {str(e)}")

    def _plot_carte_chaleur_pertes(self):
        """Exemple de carte de chaleur des pertes de charge (illustratif)"""
        if not self.meilleure_solution:
            return
        
        try:
            # Appliquer la meilleure solution
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # Exemple : calculer des pertes de charge par conduite
            pertes = {}
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                debit = resultats.link['flowrate'][pipe_name].mean()
                # Simple hypothèse ou formule Darcy-Weisbach
                if debit > 0:
                    # On se limite ici à un calcul arbitraire illustratif
                    longueur = pipe.length
                    diam = pipe.diameter
                    pertes[pipe_name] = (longueur / (diam + 1e-9)) * debit
                else:
                    pertes[pipe_name] = 0
            
            # Préparation des coordonnées pour tracer
            x, y, c = [], [], []
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                start_node = self.reseau.get_node(pipe.start_node_name)
                end_node = self.reseau.get_node(pipe.end_node_name)
                
                xs = [start_node.coordinates[0], end_node.coordinates[0]]
                ys = [start_node.coordinates[1], end_node.coordinates[1]]
                val = pertes[pipe_name]
                
                # Pour dessiner une ligne colorée, on duplique x,y
                x.extend(xs)
                y.extend(ys)
                c.extend([val, val])

            # Trace
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(x, y, c=c, cmap='YlOrRd')
            plt.colorbar(sc, label='Pertes de charge (arbitraire)')
            plt.title("Carte de chaleur des pertes de charge")
            plt.xlabel("Coordonnée X")
            plt.ylabel("Coordonnée Y")
            plt.grid(True)
            plt.savefig('carte_chaleur_pertes.png')
            plt.close()

            # Réinitialisation
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

        except Exception as e:
            print(f"Erreur lors de la carte de chaleur des pertes : {e}")

    def _plot_evolution_temporelle(self):
        """Stub : À implémenter selon vos besoins (pressions/vitesses en fonction du temps, etc.)"""
        plt.figure()
        plt.title("Évolution temporelle (exemple à compléter)")
        plt.xlabel("Temps")
        plt.ylabel("Paramètre (pression, débit...)")
        plt.savefig("evolution_temporelle.png")
        plt.close()

    def _plot_pareto(self):
        """Stub : À implémenter si vous faites du multi-objectif (front de Pareto)."""
        plt.figure()
        plt.title("Front de Pareto (exemple à compléter)")
        plt.xlabel("Objectif 1")
        plt.ylabel("Objectif 2")
        plt.savefig("front_pareto.png")
        plt.close()

    def generer_comparaison(self):
        """
        Génère une comparaison avant/après optimisation en utilisant
        la meilleure solution.
        """
        if not self.meilleure_solution:
            print("Aucune meilleure solution pour comparaison.")
            return

        try:
            # 1. Simulation initiale
            print("Exécution de la simulation initiale...")
            # On remet les diamètres initiaux
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
            sim_init = wntr.sim.EpanetSimulator(self.reseau)
            resultats_initiaux = sim_init.run_sim()

            pressions_initiales = resultats_initiaux.node['pressure'].mean().values
            pertes_init = resultats_initiaux.link['flowrate'].sum().sum()
            energie_init = self.calculer_energie(resultats_initiaux)

            # 2. Simulation après optimisation
            print("Exécution de la simulation optimisée...")
            for i, diametre in enumerate(self.meilleure_solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            sim_opt = wntr.sim.EpanetSimulator(self.reseau)
            resultats_opt = sim_opt.run_sim()

            pressions_optim = resultats_opt.node['pressure'].mean().values
            pertes_opt = resultats_opt.link['flowrate'].sum().sum()
            energie_opt = self.calculer_energie(resultats_opt)

            # Préparation des dicts
            resultats_initiaux_dict = {
                'pressions': pressions_initiales,
                'pertes_totales': pertes_init,
                'energie_totale': energie_init
            }
            resultats_optimises_dict = {
                'pressions': pressions_optim,
                'pertes_totales': pertes_opt,
                'energie_totale': energie_opt
            }

            # Création des graphiques comparatifs
            self._plot_comparaison(resultats_initiaux_dict, resultats_optimises_dict)

            # Affichage possible ou rapport supplémentaire
            print("Comparaison avant/après exportée : comparaison_resultats.png")

            # Réinitialisation des diamètres (optionnel)
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

        except Exception as e:
            print(f"Erreur lors de la génération de la comparaison : {e}")

    def _plot_comparaison(self, resultats_initiaux, resultats_optimises):
        """
        Création de graphiques comparatifs entre résultats initiaux et optimisés
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Distribution des pressions
            ax1.hist(resultats_initiaux['pressions'], bins=30, alpha=0.5, label='Initial', color='blue')
            ax1.hist(resultats_optimises['pressions'], bins=30, alpha=0.5, label='Optimisé', color='green')
            ax1.set_title('Distribution des pressions')
            ax1.set_xlabel('Pression (bar)')
            ax1.set_ylabel('Fréquence')
            ax1.legend()
            ax1.grid(True)
            
            # 2. Pertes
            labels = ['Initial', 'Optimisé']
            pertes_values = [resultats_initiaux['pertes_totales'], resultats_optimises['pertes_totales']]
            ax2.bar(labels, pertes_values, color=['blue', 'green'])
            ax2.set_title('Pertes totales')
            ax2.set_ylabel('Volume (m³/h)')
            ax2.grid(True)
            
            # 3. Énergie
            energie_values = [resultats_initiaux['energie_totale'], resultats_optimises['energie_totale']]
            ax3.bar(labels, energie_values, color=['blue', 'green'])
            ax3.set_title('Consommation énergétique')
            ax3.set_ylabel('Énergie (kWh, estimée)')
            ax3.grid(True)
            
            # 4. Améliorations en pourcentage
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
            ax4.set_title("Pourcentage d'amélioration (les valeurs négatives sont des réductions)")
            ax4.set_ylabel('Amélioration (%)')
            ax4.grid(True)
            
            plt.tight_layout()
            fig.suptitle('Comparaison des résultats avant/après optimisation', fontsize=16, y=1.02)
            plt.savefig('comparaison_resultats.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Petit rapport texte
            rapport = f"""
            --- Rapport de comparaison ---
            
            Pression moyenne initiale : {np.mean(resultats_initiaux['pressions']):.2f} bar
            Pression moyenne optimisée : {np.mean(resultats_optimises['pressions']):.2f} bar
            Amélioration moyenne pression : {ameliorations['Pressions']:.2f} %
            
            Pertes initiales : {resultats_initiaux['pertes_totales']:.2f} m³/h
            Pertes optimisées : {resultats_optimises['pertes_totales']:.2f} m³/h
            Amélioration pertes : {ameliorations['Pertes']:.2f} %
            
            Énergie initiale : {resultats_initiaux['energie_totale']:.2f} kWh
            Énergie optimisée : {resultats_optimises['energie_totale']:.2f} kWh
            Amélioration énergie : {ameliorations['Énergie']:.2f} %
            """
            with open('rapport_comparaison.txt', 'w') as f:
                f.write(rapport)

        except Exception as e:
            print(f"Erreur dans la comparaison avant/après : {e}")

    def calculer_energie(self, resultats):
        """
        Calcule une estimation de l'énergie consommée dans le réseau à partir des résultats de simulation
        (exemple simplifié).
        """
        # On peut tenter : Puissance hydraulique = rho*g*Q*H (H = pertes de charge)
        # Approach simpliste ici :
        debits = resultats.link['flowrate'].values  # m3/s
        headloss = resultats.link['headloss'].values  # m de perte de charge
        rho = 1000      # kg/m3
        g = 9.81        # m/s2

        # Somme sur l'ensemble des conduites et du temps :
        # Puissance (W) = rho*g*Débit*Headloss
        # Énergie sur l'ensemble du pas de temps : P * dt, puis conversion en kWh
        # Hypothèse : 1 pas de temps = 1 seconde, à adapter selon votre simulation
        # Démonstration indicative :
        puissances = rho * g * debits * headloss  # shape (time_steps, nb_pipes)
        total_puissance = puissances.sum()        # W

        # Nombre de pas de temps
        n_steps = debits.shape[0]
        # On suppose 1 step = 1 seconde => total_time en heures
        total_time_h = n_steps / 3600

        # Énergie en Wh => (W) * (h), puis en kWh => /1000
        energie_kwh = (total_puissance * total_time_h) / 1000
        return energie_kwh

    # =========================================================================
    # Méthodes d'export de résultats (rapports CSV / JSON)
    # =========================================================================


    def _export_resume_optimisation(self):
        """Export du tableau récapitulatif des résultats de la meilleure solution."""
        if not self.meilleure_solution:
            return

        # --- Copie profonde du réseau ---
        reseau_optimal = copy.deepcopy(self.reseau)

        # Application de la meilleure solution
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_optimal.pipe_name_list[i]
            reseau_optimal.get_link(nom_conduite).diameter = diametre/1000

        # Simulation hydraulique
        sim = wntr.sim.EpanetSimulator(reseau_optimal)
        resultats = sim.run_sim()

        # Création du DataFrame de résumé
        df_resume = pd.DataFrame({
            'Métrique': [
                'Coût total (FCFA)',
                'Pression moyenne (bar)',
                'Pression minimale (bar)',
                'Pression maximale (bar)',
                'Vitesse moyenne (m/s)',
                'Vitesse minimale (m/s)',
                'Vitesse maximale (m/s)'
            ],
            'Valeur': [
                self.calculer_cout_total(self.meilleure_solution),
                resultats.node['pressure'].mean().mean(),
                resultats.node['pressure'].min().min(),
                resultats.node['pressure'].max().max(),
                resultats.link['velocity'].mean().mean(),
                resultats.link['velocity'].min().min(),
                resultats.link['velocity'].max().max()
            ]
        })
        df_resume.to_csv('resume_optimisation.csv', index=False)


    def _export_indicateurs_performance(self):
        """Export des indicateurs de performance (exemple par zones, à personnaliser)."""
        # À adapter selon la structure réelle du réseau
        # Stub d’exemple
        data = [
            {'Zone': 'Nord', 'Pression moyenne': 8.5, 'Nombre de noeuds': 10},
            {'Zone': 'Sud', 'Pression moyenne': 7.2, 'Nombre de noeuds': 8},
        ]
        df = pd.DataFrame(data)
        df.to_csv('indicateurs_performance.csv', index=False)

    def _export_parametres_critiques(self):
        """Export des éléments où les contraintes ne sont pas respectées."""
        if not self.meilleure_solution:
            return

        # --- Copie profonde du réseau ---
        reseau_opt = copy.deepcopy(self.reseau)

        # Application de la meilleure solution
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_opt.pipe_name_list[i]
            reseau_opt.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(reseau_opt)
        res = sim.run_sim()

        pressions_moy = res.node['pressure'].mean()
        vitesses_moy = res.link['velocity'].mean()

        parametres_critiques = []

        # Vérification des nœuds
        for node_name in pressions_moy.index:
            p = pressions_moy[node_name]
            if p < PRESSION_MIN or p > PRESSION_MAX:
                parametres_critiques.append({
                    'Element': node_name,
                    'Type': 'Nœud',
                    'Paramètre': 'Pression',
                    'Valeur': p
                })

        # Vérification des conduites
        for link_name in vitesses_moy.index:
            v = vitesses_moy[link_name]
            if v < VITESSE_MIN or v > VITESSE_MAX:
                parametres_critiques.append({
                    'Element': link_name,
                    'Type': 'Conduite',
                    'Paramètre': 'Vitesse',
                    'Valeur': v
                })

        df = pd.DataFrame(parametres_critiques)
        df.to_csv('parametres_critiques.csv', index=False)


    def exporter_resultats_json(self):
        """Export des résultats en format JSON avec paramètres hydrauliques de la meilleure solution."""
        if not self.meilleure_solution:
            return
        
        # Appliquer la meilleure solution sur une copie
       
        reseau_opt = copy.deepcopy(self.reseau)# --- Copie profonde du réseau ---
        for i, diametre in enumerate(self.meilleure_solution):
            nom_conduite = reseau_opt.pipe_name_list[i]
            reseau_opt.get_link(nom_conduite).diameter = diametre/1000

        sim = wntr.sim.EpanetSimulator(reseau_opt)
        resultats = sim.run_sim()

        pressions = {n: float(resultats.node['pressure'][n].mean()) for n in reseau_opt.node_name_list}
        vitesses = {l: float(resultats.link['velocity'][l].mean()) for l in reseau_opt.link_name_list}

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
                "cout_total": self.calculer_cout_total(self.meilleure_solution),
                "historique_convergence": [float(f) for f in self.historique_fitness]
            }
        }

        with open('resultats_optimisation.json', 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=4, ensure_ascii=False)

    def generer_rapports(self):
        """Génération globale des rapports CSV (pour simplifier l’appel dans main)."""
        self._export_resume_optimisation()
        self._export_indicateurs_performance()
        self._export_parametres_critiques()
        # Vous pourriez ajouter d'autres méthodes d'export ici si nécessaire.


# ------------------------------------------------------------------------
# Classe d'analyse économique (exemple)
# ------------------------------------------------------------------------
class AnalyseEconomique:
    def __init__(self, reseau):
        """
        Initialisation avec les données du réseau (ou paramètres techniques/financiers)
        """
        self.reseau = reseau
        self.taux_actualisation = 0.08  # 8% taux d'actualisation
        self.duree_projet = 20  # années

    def calculer_couts_investissement(self):
        """
        Calcul (exemple) des coûts d'investissement initiaux
        """
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
        
        total_investissement = sum(
            sum(categorie.values()) for categorie in couts.values()
        )
        return couts, total_investissement

    def calculer_economies_annuelles(self, resultats_optimisation):
        """
        Calcul des économies annuelles réalisées (exemple fictif)
        
        Args:
            resultats_optimisation (dict): contient des valeurs comme
                'reduction_pertes', 'reduction_energie', etc.
        """
        economies = {
            'reduction_pertes': {
                'volume_economise': resultats_optimisation.get('reduction_pertes', 5000), # m3/an
                'prix_eau': 500,  # FCFA/m3
            },
            'energie': {
                'kwh_economises': resultats_optimisation.get('reduction_energie', 2000),
                'prix_kwh': 100   # FCFA/kWh
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
        """Valeur Actuelle Nette (VAN) simplifiée."""
        flux = -investissement
        for annee in range(1, self.duree_projet + 1):
            flux += economies_annuelles / (1 + self.taux_actualisation)**annee
        return flux

    def calculer_tri(self, investissement, economies_annuelles):
        """Taux de Rentabilité Interne (TRI) par dichotomie simplifiée."""
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
        """
        Exemple d’analyse de sensibilité : on fait varier investissement et économies de ±20 %
        """
        variations = [-20, -10, 0, 10, 20]
        result = {
            'variation_investissement': [],
            'variation_economies': []
        }
        for var in variations:
            # Variation investissement
            inv_mod = investissement * (1 + var/100.0)
            van_inv = self.calculer_van(inv_mod, economies)
            tri_inv = self.calculer_tri(inv_mod, economies)
            result['variation_investissement'].append({
                'variation_%': var,
                'VAN': van_inv,
                'TRI': tri_inv
            })

            # Variation économies
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
        """Exporte le rapport dans un fichier Excel."""
        writer = pd.ExcelWriter('rapport_economique.xlsx', engine='xlsxwriter')
        
        # Onglet Investissements
        detail_inv = rapport['investissement']['detail']
        df_inv = pd.DataFrame(detail_inv).T  # transpose pour avoir colonnes
        df_inv['Sous-total'] = df_inv.sum(axis=1)
        df_inv.to_excel(writer, sheet_name='Investissements')
        
        # Onglet Économies
        detail_eco = rapport['economies_annuelles']['detail']
        df_eco = []
        for k, v in detail_eco.items():
            df_eco.append({'Type': k, 'Economie_FCFA': v['economie']})
        df_eco = pd.DataFrame(df_eco)
        df_eco.to_excel(writer, sheet_name='Economies', index=False)
        
        # Onglet Indicateurs
        ind = [{
            'VAN': rapport['indicateurs_rentabilite']['VAN'],
            'TRI': rapport['indicateurs_rentabilite']['TRI'],
            'Temps_retour (années)': rapport['indicateurs_rentabilite']['temps_retour']
        }]
        df_ind = pd.DataFrame(ind)
        df_ind.to_excel(writer, sheet_name='Indicateurs', index=False)
        
        # Onglet Sensibilité
        sensi = rapport['analyse_sensibilite']
        df_sens_inv = pd.DataFrame(sensi['variation_investissement'])
        df_sens_eco = pd.DataFrame(sensi['variation_economies'])
        df_sens_inv.to_excel(writer, sheet_name='Sensib_Invest', index=False)
        df_sens_eco.to_excel(writer, sheet_name='Sensib_Eco', index=False)
        
        # Au lieu de writer.save(), on utilise :
        writer.close()


    def generer_rapport_economique(self, resultats_optimisation):
        """
        Génération du rapport économique global + export Excel
        """
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

        # Export en Excel
        self.exporter_rapport_excel(rapport)
        return rapport


# ------------------------------------------------------------------------
# Fonction principale
# ------------------------------------------------------------------------
def main():
    try:
        print("\n=== OPTIMISATION DU RÉSEAU HYDRAULIQUE D'EBOLOWA ===\n")
        print("Initialisation...")
        print("Chargement du fichier réseau : ebolowa_reseau.inp")

        # Instanciation de la classe OptimisationReseau
        optimiseur = OptimisationReseau('ebolowa_reseau.inp')
        
        print("\n1. LANCEMENT DE L'OPTIMISATION")
        print(f"   - Population initiale : {TAILLE_POPULATION} solutions")
        print(f"   - Nombre de générations : {NOMBRE_GENERATIONS}")
        print(f"   - Taux de croisement : {TAUX_CROISEMENT}")
        print(f"   - Taux de mutation : {TAUX_MUTATION}")
        
        print("\nDébut de l'optimisation génétique...")
        print("Progression : ", end='', flush=True)
        
        def callback_progression(gen):
            if gen % 2 == 0:  # Afficher un symbole tous les 2 générations
                print("▓", end='', flush=True)
        
        optimiseur.executer_optimisation(callback=callback_progression)
        print("\nOptimisation terminée !")
        
        print("\n2. GÉNÉRATION DES VISUALISATIONS")
        print("   - Création du graphique de convergence...")
        optimiseur._plot_convergence()
        
        print("   - Création de la distribution des pressions...")
        optimiseur._plot_distribution_pressions()
        
        print("   - Création de la carte de chaleur des pertes...")
        optimiseur._plot_carte_chaleur_pertes()
        
        print("   - Comparaison avant/après optimisation...")
        optimiseur.generer_comparaison()
        
        print("   - Création de l'évolution temporelle (stub)...")
        optimiseur._plot_evolution_temporelle()
        
        print("   - Création du front de Pareto (stub)...")
        optimiseur._plot_pareto()
        
        print("Toutes les visualisations ont été générées avec succès.")
        
        print("\n3. GÉNÉRATION DES RAPPORTS CSV")
        optimiseur.generer_rapports()
        print("   - Resume : resume_optimisation.csv")
        print("   - Indicateurs : indicateurs_performance.csv")
        print("   - Paramètres critiques : parametres_critiques.csv")
        
        print("\n   - Analyse économique (exemple fictif)...")
        analyse_eco = AnalyseEconomique(optimiseur.reseau)
        # On construit un mini-dictionnaire pour illustrer
        resultats_optimisation = {
            'reduction_pertes': 5000,
            'reduction_energie': 2000,
            'reduction_maintenance': 10
        }
        rapport_eco = analyse_eco.generer_rapport_economique(resultats_optimisation)
        print("   => Rapport : rapport_economique.xlsx")
        
        print("\n4. EXPORT DES RÉSULTATS JSON")
        optimiseur.exporter_resultats_json()
        print("   => Fichier : resultats_optimisation.json")
        
        print("\n=== OPTIMISATION TERMINÉE AVEC SUCCÈS ===")
        print("\nRésumé des fichiers générés :")
        print("├── Visualisations")
        print("│   ├── convergence.png")
        print("│   ├── distribution_pressions.png")
        print("│   ├── carte_chaleur_pertes.png")
        print("│   ├── comparaison_resultats.png")
        print("│   ├── evolution_temporelle.png  (stub)")
        print("│   └── front_pareto.png          (stub)")
        print("├── Rapports CSV")
        print("│   ├── resume_optimisation.csv")
        print("│   ├── indicateurs_performance.csv")
        print("│   └── parametres_critiques.csv")
        print("├── Rapport économique")
        print("│   └── rapport_economique.xlsx")
        print("└── Résultats JSON")
        print("    └── resultats_optimisation.json")
        
        # Affichage de la meilleure solution
        print("\nMeilleure solution trouvée :")
        print(optimiseur.meilleure_solution)

    except Exception as e:
        print(f"\n❌ ERREUR : {str(e)}")
        print("L'optimisation n'a pas pu être terminée.")
        raise

# Point d'entrée
if __name__ == "__main__":
    main()
