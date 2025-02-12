# Optimisation de Réseau Hydraulique d'Ebolowa

## Description
Ce projet implémente un système d'optimisation pour le réseau hydraulique de la ville d'Ebolowa. Il utilise des algorithmes génétiques pour optimiser le dimensionnement des conduites tout en respectant les contraintes hydrauliques et en minimisant les coûts.

## 🎯 Fonctionnalités Principales

- Optimisation mono-objectif (minimisation des coûts)
- Optimisation multi-objectif (compromis coût/énergie) avec NSGA-II
- Analyse économique complète (VAN, TRI)
- Visualisations détaillées des résultats
- Génération de rapports techniques et financiers

## 📋 Prérequis

```bash
python >= 3.8
```

### Dépendances Principales
```bash
wntr             # Simulation hydraulique
numpy            # Calculs numériques
pandas           # Manipulation de données
matplotlib       # Visualisation
seaborn          # Visualisation avancée
deap             # Algorithmes génétiques
```

## 🚀 Installation

1. Cloner le répertoire :
```bash
git clone https://github.com/ngueleEbale/GA_implementation
cd optimisation-reseau-ebolowa
```

2. Créer un environnement virtuel :
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Exécution Simple
```python
python nonoOptimizer.py
```

```

## 📊 Structure des Résultats

```
resultats/
├── visualisation/
│   ├── convergence.png
│   ├── distribution_pressions.png
│   ├── carte_chaleur_pertes.png
│   └── front_pareto.png
├── rapports/
│   ├── rapport_technique.csv
│   ├── rapport_economique.xlsx
│   └── resultats_optimisation.json
```

## 📝 Format des Fichiers d'Entrée

### Fichier Réseau (.inp)
Format EPANET standard avec :
- Définition des nœuds
- Caractéristiques des conduites
- Paramètres de simulation

### Fichier Coûts (.csv)
```csv
diametre,cout_unitaire
40,5000
63,7000
...
```

## 🔍 Algorithmes Implémentés

### Optimisation Mono-objectif
- Algorithme génétique standard
- Sélection par tournoi
- Croisement deux points
- Mutation gaussienne

### Optimisation Multi-objectif
- NSGA-II
- Front de Pareto
- Maintien de la diversité

## 📈 Analyses Disponibles

1. **Analyses Hydrauliques**
   - Distribution des pressions
   - Vitesses d'écoulement
   - Pertes de charge

2. **Analyses Économiques**
   - Coûts d'investissement
   - Économies d'exploitation
   - Rentabilité financière

## 🛠 Configuration Avancée

### Paramètres d'Optimisation
```python
TAILLE_POPULATION = 100
NOMBRE_GENERATIONS = 50
TAUX_CROISEMENT = 0.8
TAUX_MUTATION = 0.2
```



