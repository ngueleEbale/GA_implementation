# -*- coding: utf-8 -*-
import copy
import os
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import json
import random
from datetime import datetime
import matplotlib.collections as mc
import matplotlib.colors as mcolors

# ------------------------------------------------------------------------
# PARAMÈTRES GLOBAUX
# ------------------------------------------------------------------------

# Exemple : vous avez mis 204 et 612, pensant à (20 bar et 60 bar) * 10.2 = 204m et 612m
# Vérifiez bien que la simulation EPANET renvoie la pression en mCE.
PRESSION_MIN = 204   # ~20 bar, si 1 bar = ~10.2 m
PRESSION_MAX = 612   # ~60 bar

DIAMETRE_MIN = 40    # mm
DIAMETRE_MAX = 400   # mm
VITESSE_MIN = 0.5    # m/s
VITESSE_MAX = 1.5    # m/s

TAILLE_POPULATION = 100
NOMBRE_GENERATIONS = 5
TAUX_CROISEMENT = 0.8
TAUX_MUTATION = 0.2

# Exemple de coûts FCFA/m
COUT_DIAMETRE = {
    40 :  5000,
    63 :  7000,
    75 :  8500,
    90 : 10000,
    110: 12000,
    160: 15000,
    200: 18000,
    250: 22000,
    315: 35000,
    400: 45000
}

# ------------------------------------------------------------------------
# Création des dossiers "visualisation" et "rapports" s'ils n'existent pas
# ------------------------------------------------------------------------
os.makedirs("visualisation", exist_ok=True)
os.makedirs("rapports", exist_ok=True)

# ------------------------------------------------------------------------
# CLASSE PRINCIPALE D'OPTIMISATION
# ------------------------------------------------------------------------
class OptimisationReseau:
    def __init__(self, fichier_inp):
        """Initialisation de l'optimiseur de réseau"""
        try:
            self.meilleure_solution = None       # Meilleure solution (mono‐objectif)
            self.historique_fitness = []         # Historique du score min par génération
            self.solutions_pareto = []           # Stockage du front de Pareto (multi‐objectif)

            # Chargement du réseau
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

    # --------------------------------------------------------------------
    # GESTION DE LA POPULATION INITIALE
    # --------------------------------------------------------------------
    def initialiser_population(self):
        """Création de la population initiale"""
        population = []
        nb_conduites = len(self.reseau.pipe_name_list)

        for _ in range(TAILLE_POPULATION):
            # Choix aléatoire des diamètres possibles (clés de COUT_DIAMETRE)
            individu = [np.random.choice(list(COUT_DIAMETRE.keys()))
                        for _ in range(nb_conduites)]
            population.append(individu)

        return population

    # --------------------------------------------------------------------
    # ÉVALUATION MONO‐OBJECTIF
    # --------------------------------------------------------------------
    def evaluer_solution(self, solution):
        """
        Évaluation d'une solution (mono‐objectif).
        Score = coût total + pénalités (pressions/vitesses hors bornes).
        """
        try:
            # 1) Appliquer les diamètres
            for i, diametre in enumerate(solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre / 1000.0

            # 2) Simulation hydraulique
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # 3) Calcul du coût
            cout_total = self.calculer_cout_total(solution)
            # 4) Calcul des pénalités (pressions, vitesses)
            penalites = self.calculer_penalites(resultats)

            # 5) Réinitialisation
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            # Score = coût + pénalités
            score_final = cout_total + penalites
            return (score_final,)

        except Exception as e:
            print(f"Erreur critique lors de l'évaluation: {str(e)}")
            return (float('inf'),)

    # --------------------------------------------------------------------
    # ÉVALUATION MULTI‐OBJECTIF
    # --------------------------------------------------------------------
    def evaluer_solution_multi(self, solution):
        """
        Évaluation d'une solution dans un contexte multi-objectif.
        Objectifs à minimiser: (coût_total, énergie).
        """
        try:
            # 1) Appliquer les diamètres
            for i, diametre in enumerate(solution):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre / 1000.0

            # 2) Simulation
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # 3) Calcul des objectifs
            cout_total = self.calculer_cout_total(solution)
            energie    = self.calculer_energie(resultats)

            # 4) Pénalités
            penalites = self.calculer_penalites(resultats)
            # On gonfle (coût + énergie) par ces pénalités pour sanctionner
            cout_total += penalites
            energie    += penalites

            # 5) Réinitialisation
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            return (cout_total, energie)

        except Exception as e:
            print(f"Erreur dans l'évaluation multi-objectif : {e}")
            return (float('inf'), float('inf'))

    # --------------------------------------------------------------------
    # CALCUL DU COÛT TOTAL
    # --------------------------------------------------------------------
    def calculer_cout_total(self, solution):
        """Somme des coûts (FCFA) pour les diamètres choisis."""
        cout = 0
        for i, diametre_mm in enumerate(solution):
            nom_conduite = self.reseau.pipe_name_list[i]
            longueur_m = self.reseau.get_link(nom_conduite).length
            cout_diam = COUT_DIAMETRE[diametre_mm]  # FCFA/m
            cout += cout_diam * longueur_m
        return cout

    # --------------------------------------------------------------------
    # CALCUL DES PÉNALITÉS
    # --------------------------------------------------------------------
    def calculer_penalites(self, resultats):
        """
        Ajoute des pénalités si la pression/vitesse sort des bornes [PRESSION_MIN, PRESSION_MAX],
        [VITESSE_MIN, VITESSE_MAX].
        Rappel: si EPANET renvoie la pression en mCE, assurez-vous que PRESSION_MIN et MAX sont en mCE!
        """
        penalites = 0

        # 1) Pressions
        pressions = resultats.node['pressure'].values  # (time_steps, nb_nodes)
        pressions_moy = pressions.mean(axis=0)         # Moyenne par nœud
        for p in pressions_moy:
            if p < PRESSION_MIN:
                penalites += (PRESSION_MIN - p) * 1e6
            elif p > PRESSION_MAX:
                penalites += (p - PRESSION_MAX) * 1e6

        # 2) Vitesses
        vitesses = resultats.link['velocity'].values
        vitesses_moy = vitesses.mean(axis=0)
        for v in vitesses_moy:
            if v < VITESSE_MIN:
                penalites += (VITESSE_MIN - v) * 1e6
            elif v > VITESSE_MAX:
                penalites += (v - VITESSE_MAX) * 1e6

        return penalites

    # --------------------------------------------------------------------
    # OPTIMISATION MONO‐OBJECTIF
    # --------------------------------------------------------------------
    def executer_optimisation(self, callback=None):
        """Optimisation mono‐objectif (score = coût + pénalités)."""
        try:
            # Création de la fitness
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()

            # Operators
            toolbox.register("individual", tools.initIterate, creator.Individual,
                             lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Population initiale
            population = toolbox.population(n=TAILLE_POPULATION)

            # Évaluation initiale
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            meilleur_score_global = float('inf')
            generations_sans_amelioration = 0

            print("\n=== Début de l'optimisation mono-objectif ===\n")

            for gen in range(NOMBRE_GENERATIONS):
                # Variation
                offspring = algorithms.varAnd(
                    population, toolbox,
                    cxpb=TAUX_CROISEMENT,
                    mutpb=TAUX_MUTATION
                )
                # Évaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Sélection
                population = toolbox.select(offspring + population, k=len(population))

                # Statistiques
                fits = [ind.fitness.values[0] for ind in population]
                score_min = min(fits)
                score_moy = sum(fits)/len(fits)

                if score_min < meilleur_score_global:
                    meilleur_score_global = score_min
                    generations_sans_amelioration = 0
                    amelioration = "⭐"
                else:
                    generations_sans_amelioration += 1
                    amelioration = "  "

                print(f"Génération {gen+1} | Min: {score_min:.2f} | Moy: {score_moy:.2f} {amelioration}")
                self.historique_fitness.append(score_min)

                if callback:
                    callback(gen)

                # Arrêt anticipé si 10 générations sans amélioration
                if generations_sans_amelioration >= 10:
                    print("Arrêt anticipé (10 générations sans amélioration).")
                    break

            print(f"\nMeilleur score final : {meilleur_score_global:.2f}\n")

            # Meilleure solution
            self.meilleure_solution = tools.selBest(population, k=1)[0]

        except Exception as e:
            print(f"Erreur pendant l'optimisation : {e}")
            raise

    # --------------------------------------------------------------------
    # OPTIMISATION MULTI‐OBJECTIF
    # --------------------------------------------------------------------
    def reproduction_genetique(self, population, toolbox, cxpb=0.8, mutpb=0.2):
        """Fonction manuelle de reproduction (clonage, croisement, mutation)."""
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

    def executer_optimisation_multi(self, callback=None):
        """Optimisation multi-objectif (ex. coût + énergie) avec NSGA-II."""
        try:
            if not hasattr(creator, "FitnessMulti"):
                creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
                creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()

            toolbox.register("individual", tools.initIterate, creator.IndividualMulti,
                             lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution_multi)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)

            population = toolbox.population(n=TAILLE_POPULATION)

            # Évaluation initiale
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Tri initial
            population = toolbox.select(population, len(population))

            print("\n=== Début de l'optimisation multi-objectif (NSGA-II) ===\n")
            for gen in range(NOMBRE_GENERATIONS):
                # Reproduction manuelle
                offspring = self.reproduction_genetique(
                    population, toolbox,
                    cxpb=TAUX_CROISEMENT,
                    mutpb=TAUX_MUTATION
                )

                # Évaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Sélection
                population = toolbox.select(population + offspring, k=len(population))

                # Front non dominé
                fronts = tools.sortNondominated(population, k=len(population), first_front_only=True)
                pareto_front = fronts[0] if len(fronts) > 0 else []
                print(f"Génération {gen+1} : {len(pareto_front)} solutions sur le front de Pareto")

                if callback:
                    callback(gen)

            final_fronts = tools.sortNondominated(population, k=len(population), first_front_only=True)
            self.solutions_pareto = final_fronts[0] if len(final_fronts) > 0 else []

            print(f"\nOptimisation multi-objectif terminée. Taille du front de Pareto : {len(self.solutions_pareto)}\n")

        except Exception as e:
            print(f"Erreur pendant l'optimisation multi-objectif: {str(e)}")
            raise

    # --------------------------------------------------------------------
    # CALCUL DE L'ÉNERGIE (ex. simplifié)
    # --------------------------------------------------------------------
    def calculer_energie(self, resultats):
        """
        Ex. : Puissance ~ rho*g*Q*Headloss (W), cumulée sur le temps => kWh.
        """
        debits    = resultats.link['flowrate'].values   # [time, link]
        headloss  = resultats.link['headloss'].values   # [time, link]
        rho       = 1000
        g         = 9.81

        puissances = rho * g * debits * headloss  # W
        total_puissance = puissances.sum()

        n_steps = debits.shape[0]
        # Si 1 step = 1 seconde => total_time_h = n_steps/3600
        # (vérifiez votre pas de temps dans le .inp)
        total_time_h = n_steps / 3600.0

        energie_kwh = (total_puissance * total_time_h) / 1000.0
        return energie_kwh

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
        plt.savefig('visualisation/convergence.png')  # <--- Dans dossier "visualisation"
        plt.close()
        print("Graphique de convergence sauvegardé : visualisation/convergence.png")

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
            plt.savefig('visualisation/distribution_pressions.png')
            plt.close()

            # Réinit.
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("Distribution des pressions : visualisation/distribution_pressions.png")

        except Exception as e:
            print(f"Erreur dans _plot_distribution_pressions : {e}")

    def _plot_carte_chaleur_pertes(self):
        """
        Carte de chaleur des pertes de charge, + visualisation des nœuds.
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

            # 2) Calcul arbitraire des "pertes"
            pertes = {}
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                debit = resultats.link['flowrate'][pipe_name].mean()
                longu = pipe.length
                diam  = pipe.diameter
                if debit > 0:
                    pertes[pipe_name] = (longu / (diam + 1e-9)) * debit
                else:
                    pertes[pipe_name] = 0

            # 3) Construire la liste de segments
            segments = []
            values   = []
            for pipe_name in self.reseau.pipe_name_list:
                pipe = self.reseau.get_link(pipe_name)
                start_node = self.reseau.get_node(pipe.start_node_name)
                end_node   = self.reseau.get_node(pipe.end_node_name)

                x1, y1 = start_node.coordinates
                x2, y2 = end_node.coordinates

                seg = [(x1, y1), (x2, y2)]
                segments.append(seg)
                values.append(pertes[pipe_name])

            # Préparation du LineCollection
            cmap = plt.cm.YlOrRd
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            lc = mc.LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
            lc.set_array(np.array(values))

            # 4) Tracé
            plt.figure(figsize=(12, 10))
            ax = plt.gca()
            ax.add_collection(lc)

            # 5) Tracer les nœuds
            all_x_nodes = []
            all_y_nodes = []
            for node_name in self.reseau.node_name_list:
                node = self.reseau.get_node(node_name)
                all_x_nodes.append(node.coordinates[0])
                all_y_nodes.append(node.coordinates[1])

            # On peut les dessiner en noir, taille 30
            plt.scatter(all_x_nodes, all_y_nodes, c='black', s=30, marker='o', label='Nœuds')

            # Ajuster l'échelle du plot
            all_x = [c[0] for seg in segments for c in seg] + all_x_nodes
            all_y = [c[1] for seg in segments for c in seg] + all_y_nodes
            ax.set_xlim(min(all_x)*0.95, max(all_x)*1.05)
            ax.set_ylim(min(all_y)*0.95, max(all_y)*1.05)

            plt.colorbar(lc, label='Pertes de charge (arbitraire)')
            plt.title("Carte de chaleur des pertes + Nœuds")
            plt.xlabel("Coordonnée X")
            plt.ylabel("Coordonnée Y")
            plt.legend()
            plt.grid(True)

            plt.savefig('visualisation/carte_chaleur_pertes.png', dpi=300)
            plt.close()

            # 6) Réinitialiser
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("Carte de chaleur des pertes : visualisation/carte_chaleur_pertes.png")

        except Exception as e:
            print(f"Erreur lors de la carte de chaleur des pertes : {e}")


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
            plt.savefig("visualisation/evolution_temporelle.png", dpi=300)
            plt.close()

            # Réinit
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            print("Évolution temporelle : visualisation/evolution_temporelle.png")

        except Exception as e:
            print(f"Erreur dans _plot_evolution_temporelle : {e}")

    def _plot_pareto(self):
        """Trace le front de Pareto (coût vs énergie)."""
        if not self.solutions_pareto:
            print("Aucune solution Pareto disponible.")
            return
        try:
            costs    = [ind.fitness.values[0] for ind in self.solutions_pareto]
            energies = [ind.fitness.values[1] for ind in self.solutions_pareto]

            plt.figure(figsize=(8,6))
            plt.scatter(costs, energies, c='blue', label='Front de Pareto')
            plt.title("Front de Pareto - (Coût vs Énergie)")
            plt.xlabel("Coût total (FCFA)")
            plt.ylabel("Énergie (kWh)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("visualisation/front_pareto.png", dpi=300)
            plt.close()

            print("Front de Pareto : visualisation/front_pareto.png")

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
            plt.savefig('visualisation/comparaison_resultats.png', dpi=300, bbox_inches='tight')
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
            with open('rapports/rapport_comparaison.txt', 'w') as f:
                f.write(rapport)

            print("Comparaison avant/après : visualisation/comparaison_resultats.png\n"
                  "Rapport texte : rapports/rapport_comparaison.txt")

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
                'Coût total (FCFA)',
                'Pression moyenne (m)',
                'Pression minimale (m)',
                'Pression maximale (m)',
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
        df_resume.to_csv('rapports/resume_optimisation.csv', index=False)
        print("Résumé optimisation : rapports/resume_optimisation.csv")

    def _export_indicateurs_performance(self):
        """Export des indicateurs (exemple)."""
        data = [
            {'Zone': 'Nord', 'Pression moyenne': 8.5, 'Nombre de noeuds': 10},
            {'Zone': 'Sud',  'Pression moyenne': 7.2, 'Nombre de noeuds': 8},
        ]
        df = pd.DataFrame(data)
        df.to_csv('rapports/indicateurs_performance.csv', index=False)
        print("Indicateurs performance : rapports/indicateurs_performance.csv")

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
        df.to_csv('rapports/parametres_critiques.csv', index=False)
        print("Paramètres critiques : rapports/parametres_critiques.csv")

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
                "cout_total": self.calculer_cout_total(self.meilleure_solution),
                "historique_convergence": [float(f) for f in self.historique_fitness]
            }
        }

        with open('rapports/resultats_optimisation.json', 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=4, ensure_ascii=False)

        print("Résultats JSON : rapports/resultats_optimisation.json")


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
        writer = pd.ExcelWriter('rapports/rapport_economique.xlsx', engine='xlsxwriter')

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
# POINT D'ENTRÉE PRINCIPAL
# ------------------------------------------------------------------------
def main():
    try:
        print("\n=== OPTIMISATION DU RÉSEAU HYDRAULIQUE D'EBOLOWA ===\n")
        optimiseur = OptimisationReseau('ebolowa_reseau.inp')

        # 1) MONO‐OBJECTIF
        print("\n1) Lancement de l'optimisation MONO-OBJECTIF...")
        optimiseur.executer_optimisation()
        print("Optimisation mono-objectif terminée.")

        # Visualisations
        optimiseur._plot_convergence()
        optimiseur._plot_distribution_pressions()
        optimiseur._plot_carte_chaleur_pertes()
        optimiseur.generer_comparaison()
        optimiseur._plot_evolution_temporelle()

        # Rapports CSV + JSON
        optimiseur.generer_rapports()
        optimiseur.exporter_resultats_json()

        # 2) MULTI‐OBJECTIF
        print("\n2) Lancement de l'optimisation MULTI-OBJECTIF (NSGA-II)...")
        optimiseur.executer_optimisation_multi()
        optimiseur._plot_pareto()

        # 3) Analyse économique (exemple)
        print("\n3) Analyse économique (exemple) ...")
        analyse_eco = AnalyseEconomique(optimiseur.reseau)
        resultats_optimisation = {
            'reduction_pertes': 5000,
            'reduction_energie': 2000,
            'reduction_maintenance': 10
        }
        analyse_eco.generer_rapport_economique(resultats_optimisation)
        print("Rapport économique généré : rapports/rapport_economique.xlsx")

        print("\n=== TOUT EST TERMINÉ AVEC SUCCÈS ===")

        # NOTE: Si vous constatez que vos pressions calculées (en m)
        # sont tjs hors [204,612], c'est peut-être un problème d'unités
        # (vous vouliez en bar ?). Vérifiez le .inp et adaptez.
        # ex: Pression( bar ) = Pression( m ) / 10.2 environ.

    except Exception as e:
        print(f"ERREUR : {str(e)}")
        raise

if __name__ == "__main__":
    main()
