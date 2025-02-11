import wntr
import numpy as np
from deap import base, creator, tools, algorithms
import pandas as pd
import logging
from datetime import datetime
import json

class OptimisationReseau:
    def __init__(self, fichier_inp, parametres):
        """
        Initialisation de l'optimiseur de réseau
        
        Args:
            fichier_inp (str): Chemin du fichier .inp EPANET
            parametres (dict): Paramètres d'optimisation
        """
        self.fichier_inp = fichier_inp
        self.reseau = wntr.network.WaterNetworkModel(fichier_inp)
        self.parametres = parametres
        self.meilleure_solution = None
        self.historique_fitness = []
        self.solutions_pareto = []
        
        # Poids des objectifs
        self.poids = {
            'efficacite': 0.4,
            'pertes': 0.3,
            'energie': 0.3
        }
        
        # Sauvegarde des diamètres initiaux
        self.diametres_initiaux = self._get_diametres_initiaux()
        
        logging.info(f"Réseau chargé: {len(self.reseau.node_name_list)} nœuds, "
                    f"{len(self.reseau.pipe_name_list)} conduites")

    def _get_diametres_initiaux(self):
        """Récupère les diamètres initiaux du réseau"""
        return [self.reseau.get_link(pipe).diameter 
                for pipe in self.reseau.pipe_name_list]

    def initialiser_population(self):
        """Crée la population initiale"""
        population = []
        nb_conduites = len(self.reseau.pipe_name_list)
        nb_noeuds = len(self.reseau.node_name_list)
        diametres = self.parametres['diametres_disponibles']
        
        # Favoriser les grands diamètres dans la population initiale
        probas_diametres = np.array([0.1, 0.15, 0.2, 0.25, 0.1, 0.1, 0.1])  # Ajusté pour 7 diamètres
        
        for _ in range(self.parametres['population_size']):
            individu = {
                'diametres': [np.random.choice(diametres, p=probas_diametres) for _ in range(nb_conduites)],
                'pressions': [np.random.uniform(self.parametres['pression_min'], self.parametres['pression_max']) for _ in range(nb_noeuds)],
                'vitesses': [np.random.uniform(self.parametres['vitesse_min'], self.parametres['vitesse_max']) for _ in range(nb_conduites)]
            }
            population.append(individu)
        
        return population

    def calculer_score_efficacite(self, resultats_sim):
        """
        Calcule le score d'efficacité hydraulique avec des pénalités graduelles
        """
        pressions = resultats_sim.node['pressure'].mean()
        vitesses = resultats_sim.link['velocity'].mean()
        penalites = 0
        
        # Pénalités pour les pressions
        for p in pressions:
            if p < self.parametres['pression_min']:
                penalites += (self.parametres['pression_min'] - p)**2 * 1e4
            elif p < self.parametres['pression_min_service']:
                penalites += (self.parametres['pression_min_service'] - p)**2 * 1e3
            elif p > self.parametres['pression_max']:
                penalites += (p - self.parametres['pression_max'])**2 * 1e4

        # Pénalités pour les vitesses
        for v in vitesses:
            if v < self.parametres['vitesse_min']:
                penalites += (self.parametres['vitesse_min'] - v)**2 * 1e3
            elif v > self.parametres['vitesse_max']:
                penalites += (v - self.parametres['vitesse_max'])**2 * 1e3

        return penalites

    def valider_solution_hydraulique(self, resultats_sim):
        """
        Valide les résultats hydrauliques et retourne un rapport détaillé
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. Vérifications de base des résultats
            pressions = resultats_sim.node['pressure'].mean()
            vitesses = resultats_sim.link['velocity'].mean()
            
            if np.isnan(pressions).any() or np.isnan(vitesses).any():
                validation['errors'].append("Résultats invalides : valeurs NaN détectées")
                validation['valid'] = False
                return validation
                
            # 2. Vérification des pressions
            for node, p in pressions.items():
                if p < -1:  # Pression physiquement impossible
                    validation['errors'].append(f"Pression négative impossible au nœud {node}: {p:.2f}m")
                    validation['valid'] = False
                elif p < self.parametres['pression_min']:
                    validation['errors'].append(f"Pression critique au nœud {node}: {p:.2f}m")
                    validation['valid'] = False
                elif p < self.parametres['pression_min_service']:
                    validation['warnings'].append(f"Pression faible au nœud {node}: {p:.2f}m")
                elif p > 100:  # Pression trop élevée
                    validation['errors'].append(f"Pression excessive au nœud {node}: {p:.2f}m")
                    validation['valid'] = False
                elif p > self.parametres['pression_max']:
                    validation['warnings'].append(f"Pression élevée au nœud {node}: {p:.2f}m")

            # 3. Vérification des vitesses
            for pipe, v in vitesses.items():
                if abs(v) > 5:  # Vitesse physiquement improbable
                    validation['errors'].append(f"Vitesse impossible dans la conduite {pipe}: {v:.2f}m/s")
                    validation['valid'] = False
                elif abs(v) > self.parametres['vitesse_max']:
                    validation['errors'].append(f"Vitesse excessive dans la conduite {pipe}: {v:.2f}m/s")
                    validation['valid'] = False
                elif abs(v) < self.parametres['vitesse_min']:
                    validation['warnings'].append(f"Vitesse faible dans la conduite {pipe}: {v:.2f}m/s")

            return validation
            
        except Exception as e:
            validation['errors'].append(f"Erreur lors de la validation: {str(e)}")
            validation['valid'] = False
            return validation
    
    def calculer_energie(self, resultats):
        """
        Calcule l'énergie consommée
        """
        # Récupération des débits et pertes de charge
        debits = resultats.link['flowrate'].values  # m3/s
        pertes_charge = resultats.link['headloss'].values  # m
        
        # Calcul de la puissance hydraulique (W) = ρ⋅g⋅Q⋅H
        rho = 1000  # masse volumique de l'eau (kg/m3)
        g = 9.81    # accélération gravitationnelle (m/s2)
        
        puissance = rho * g * debits * pertes_charge
        
        # Calcul de l'énergie en kWh sur la période de simulation
        nb_heures = len(resultats.link['flowrate'].index)
        energie = np.sum(puissance) * nb_heures / (3600 * 1000)
        
        return energie

    def evaluer_solution(self, solution):
        """
        Évalue une solution donnée
        """
        try:
            # Application des diamètres
            for i, diametre in enumerate(solution['diametres']):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000

            # Simulation hydraulique 
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats = sim.run_sim()

            # Validation hydraulique
            validation = self.valider_solution_hydraulique(resultats)
            if not validation['valid']:
                logging.warning("Solution invalide hydrauliquement:")
                for error in validation['errors']:
                    logging.warning(f"- {error}")
                return (float('inf'),)

            # Calcul des scores
            score_efficacite = self.calculer_score_efficacite(resultats)
            pertes = resultats.link['headloss'].sum().sum()
            energie = self.calculer_energie(resultats)
            cout = self.calculer_cout(solution['diametres'])
            
            # Score total pondéré
            score_total = (
                self.poids['efficacite'] * score_efficacite +
                self.poids['pertes'] * pertes +
                self.poids['energie'] * energie +
                cout
            )

            # Restauration des diamètres initiaux
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]

            return (score_total,)

        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation: {str(e)}")
            return (float('inf'),)
    def reparer_solution(self, solution):
        """
        Répare une solution en assurant la continuité des diamètres et
        les contraintes hydrauliques
        """
        solution_reparee = solution.copy()
        
        # 1. Correction de la continuité des diamètres
        for i in range(len(solution_reparee['diametres'])):
            conduite = self.reseau.pipe_name_list[i]
            pipe = self.reseau.get_link(conduite)
            
            # Obtenir les diamètres des conduites connectées
            diametres_connectes = []
            for nom_conduite in self.reseau.pipe_name_list:
                autre_pipe = self.reseau.get_link(nom_conduite)
                if autre_pipe.start_node_name == pipe.start_node_name or \
                   autre_pipe.start_node_name == pipe.end_node_name or \
                   autre_pipe.end_node_name == pipe.start_node_name or \
                   autre_pipe.end_node_name == pipe.end_node_name:
                    idx = self.reseau.pipe_name_list.index(nom_conduite)
                    if idx < len(solution_reparee['diametres']):
                        diametres_connectes.append(solution_reparee['diametres'][idx])
            
            # Éviter les transitions trop brusques
            if diametres_connectes:
                diametre_moyen = np.mean(diametres_connectes)
                diff_max = max(self.parametres['diametres_disponibles']) * 0.5
                
                if abs(solution_reparee['diametres'][i] - diametre_moyen) > diff_max:
                    # Choisir le diamètre disponible le plus proche de la moyenne
                    solution_reparee['diametres'][i] = min(self.parametres['diametres_disponibles'], 
                                         key=lambda x: abs(x - diametre_moyen))
        
        # 2. Vérification hydraulique
        for i, diametre in enumerate(solution_reparee['diametres']):
            self.reseau.get_link(self.reseau.pipe_name_list[i]).diameter = diametre/1000
        
        sim = wntr.sim.EpanetSimulator(self.reseau)
        resultats = sim.run_sim()
        
        # 3. Correction des problèmes de pression
        pressions = resultats.node['pressure'].mean()
        for i, noeud in enumerate(self.reseau.node_name_list):
            if pressions[noeud] < self.parametres['pression_min']:
                solution_reparee['pressions'][i] = self.parametres['pression_min']
            elif pressions[noeud] > self.parametres['pression_max']:
                solution_reparee['pressions'][i] = self.parametres['pression_max']
        
        # 4. Correction des problèmes de vitesse
        vitesses = resultats.link['velocity'].mean()
        for i, conduite in enumerate(self.reseau.pipe_name_list):
            if vitesses[conduite] < self.parametres['vitesse_min']:
                solution_reparee['vitesses'][i] = self.parametres['vitesse_min']
            elif vitesses[conduite] > self.parametres['vitesse_max']:
                solution_reparee['vitesses'][i] = self.parametres['vitesse_max']
        
        # Restauration des diamètres initiaux
        for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
            self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
        
        return solution_reparee

        
    def calculer_cout(self, solution):
        """
        Calcule le coût total d'une solution
        """
        cout_total = 0
        
        # Coûts unitaires des conduites (FCFA/m)
        couts_unitaires = {
            90: 9000,
            110: 11000,
            160: 16000,
            200: 20000,
            250: 25000,
            315: 31500,
            400: 60000,
        }
        
        for i, diametre in enumerate(solution):
            nom_conduite = self.reseau.pipe_name_list[i]
            longueur = self.reseau.get_link(nom_conduite).length
            cout_total += couts_unitaires[diametre] * longueur
            
        return cout_total

    def executer_optimisation(self, callback_progression=None):
        """
        Exécute l'algorithme génétique d'optimisation
        """
        try:
            # Configuration de l'algorithme génétique
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            
            # Définition d'une fonction de mutation personnalisée
            def mutationDiscrete(individual, indpb):
                """Mutation qui choisit un diamètre dans la liste des diamètres disponibles"""
                for i in range(len(individual)):
                    if np.random.random() < indpb:
                        individual[i] = np.random.choice(self.parametres['diametres_disponibles'])
                return individual,
                
            # Enregistrement des opérateurs génétiques
            toolbox.register("individual", tools.initIterate, creator.Individual,
                            lambda: self.initialiser_population()[0])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.evaluer_solution)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", mutationDiscrete, indpb=0.15)  # Augmenté à 0.15
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Population initiale
            population = toolbox.population(n=self.parametres['population_size'])
            
            # Variables de suivi
            meilleur_score = float('inf')
            generations_sans_amelioration = 0
            
            # Métriques de convergence
            metriques_convergence = {
                'min_fitness': [],
                'avg_fitness': [],
                'max_fitness': [],
                'std_fitness': [],
                'best_solution': None,
                'generation': [],
                'best_validation': None
            }
            
            # Boucle principale d'optimisation
            for gen in range(self.parametres['nb_generations']):
                try:
                    # Création des offspring par croisement
                    offspring = algorithms.varAnd(
                        population, toolbox,
                        cxpb=self.parametres['taux_croisement'],
                        mutpb=self.parametres['taux_mutation']
                    )
                    
                    # Réparation des solutions après croisement/mutation
                    offspring_repares = []
                    for ind in offspring:
                        ind_repare = self.reparer_solution(ind)
                        offspring_repares.append(creator.Individual(ind_repare))
                    
                    # Évaluation des solutions réparées  
                    fits = toolbox.map(toolbox.evaluate, offspring_repares)
                    for fit, ind in zip(fits, offspring_repares):
                        ind.fitness.values = fit
                    
                    # Sélection pour la prochaine génération
                    population = toolbox.select(offspring_repares + population, 
                                            k=len(population))
                    
                    # Calcul des statistiques
                    fits = [ind.fitness.values[0] for ind in population 
                        if ind.fitness.valid and ind.fitness.values[0] != float('inf')]
                    
                    if fits:
                        min_fit = min(fits)
                        avg_fit = sum(fits) / len(fits)
                        max_fit = max(fits)
                        std_fit = np.std(fits)
                        
                        # Enregistrement des métriques
                        metriques_convergence['min_fitness'].append(min_fit)
                        metriques_convergence['avg_fitness'].append(avg_fit)
                        metriques_convergence['max_fitness'].append(max_fit)
                        metriques_convergence['std_fitness'].append(std_fit)
                        metriques_convergence['generation'].append(gen)
                        
                        # Mise à jour du meilleur score
                        if min_fit < meilleur_score:
                            meilleur_score = min_fit
                            generations_sans_amelioration = 0
                            meilleure_solution = tools.selBest(population, k=1)[0]
                            
                            # Validation de la meilleure solution
                            for i, diametre in enumerate(meilleure_solution):
                                nom_conduite = self.reseau.pipe_name_list[i]
                                self.reseau.get_link(nom_conduite).diameter = diametre/1000
                            
                            sim = wntr.sim.EpanetSimulator(self.reseau)
                            resultats = sim.run_sim()
                            validation = self.valider_solution_hydraulique(resultats)
                            
                            metriques_convergence['best_solution'] = list(meilleure_solution)
                            metriques_convergence['best_validation'] = validation
                            
                            # Restauration des diamètres initiaux
                            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
                        else:
                            generations_sans_amelioration += 1
                        
                        # Sauvegarde de l'historique
                        self.historique_fitness.append(min_fit)
                        
                        # Callback de progression
                        if callback_progression:
                            callback_progression((gen + 1) * 100 / self.parametres['nb_generations'])
                        
                        # Log de progression
                        logging.info(f"Génération {gen+1}: min={min_fit:.2f}, moy={avg_fit:.2f}, "
                                f"max={max_fit:.2f}, std={std_fit:.2f}")
                
                except Exception as e:
                    logging.error(f"Erreur dans la génération {gen}: {str(e)}")
                    continue
                
                # Critère d'arrêt
                if generations_sans_amelioration >= 15:  # Augmenté à 15 générations
                    logging.info("Arrêt anticipé : convergence atteinte")
                    break
            
            # Si aucune solution valide n'a été trouvée
            if meilleur_score == float('inf'):
                logging.error("Aucune solution valide trouvée")
                return {
                    'meilleure_solution': None,
                    'fitness_finale': float('inf'),
                    'historique_fitness': self.historique_fitness,
                    'nb_generations': gen + 1,
                    'metriques_convergence': metriques_convergence,
                    'validation_hydraulique': {'valid': False, 'errors': ['Aucune solution valide trouvée']},
                    'warnings_hydrauliques': [],
                    'erreurs_hydrauliques': ['Aucune solution valide trouvée']
                }
            
            # Application de la meilleure solution finale
            for i, diametre in enumerate(metriques_convergence['best_solution']):
                nom_conduite = self.reseau.pipe_name_list[i]
                self.reseau.get_link(nom_conduite).diameter = diametre/1000
            
            # Simulation finale
            sim = wntr.sim.EpanetSimulator(self.reseau)
            resultats_finaux = sim.run_sim()
            
            # Validation hydraulique finale
            validation_finale = self.valider_solution_hydraulique(resultats_finaux)
            
            # Restauration des diamètres initiaux
            for i, nom_conduite in enumerate(self.reseau.pipe_name_list):
                self.reseau.get_link(nom_conduite).diameter = self.diametres_initiaux[i]
            
            return {
                'meilleure_solution': metriques_convergence['best_solution'],
                'fitness_finale': meilleur_score,
                'historique_fitness': self.historique_fitness,
                'nb_generations': gen + 1,
                'metriques_convergence': metriques_convergence,
                'validation_hydraulique': validation_finale,
                'warnings_hydrauliques': validation_finale['warnings'],
                'erreurs_hydrauliques': validation_finale['errors']
            }
                    
        except Exception as e:
            logging.error(f"Erreur lors de l'optimisation: {str(e)}")
            raise