# 🚰 Optimisation de Réseau Hydraulique - Algorithmes Génétiques

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![EPANET](https://img.shields.io/badge/EPANET-Compatible-orange.svg)](https://epanet.org)

> Système d'optimisation avancé pour réseaux hydrauliques utilisant des algorithmes génétiques mono et multi-objectif

## 🎯 Fonctionnalités Principales

### ✨ **Optimisation Intelligente**
- **Mono-objectif** : Minimisation des pertes de charge hydrauliques
- **Multi-objectif (NSGA-II)** : Optimisation simultanée des pertes, vitesses et pressions
- **Génération automatique** de fichiers INP optimisés pour EPANET

### 📊 **Analyses Complètes**
- **Visualisations 3D** du front de Pareto
- **Cartes de chaleur** des performances hydrauliques
- **Rapports détaillés** en CSV avec statistiques complètes
- **Comparaisons avant/après** optimisation

### 🎨 **Interface Utilisateur**
- **Interface graphique** moderne avec Tkinter
- **Interface en ligne de commande** pour l'automatisation
- **Barres de progression** et contrôles d'arrêt en temps réel

### 🔧 **Robustesse Technique**
- **Gestion automatique** des encodages de fichiers INP
- **Système de logging** complet pour le débogage
- **Tests unitaires** intégrés
- **Configuration centralisée** et modulaire

## 📋 Installation Rapide

### Prérequis
- Python 3.8 ou supérieur
- Git

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/ngueleEbale/GA_implementation.git
cd GA_implementation

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Interface Graphique (Recommandée)
```bash
python run_gui.py
```

### Interface en Ligne de Commande
```bash
python main.py
```

### Tests
```bash
python run_tests.py
```

## 📁 Structure du Projet

```
GA_implementation/
├── 📁 src/                          # Code source principal
│   ├── 📁 core/                     # Logique d'optimisation
│   │   ├── optimisation.py         # Algorithmes génétiques
│   │   └── config.py               # Configuration globale
│   ├── 📁 gui/                      # Interface graphique
│   │   └── gui.py                  # Interface Tkinter
│   ├── 📁 cli/                      # Interface ligne de commande
│   │   └── optimisation_cli.py     # CLI
│   ├── 📁 tests/                    # Tests unitaires
│   │   └── tests_complets.py       # Suite de tests
│   └── 📁 utils/                    # Utilitaires
│       └── paths.py                # Gestion des chemins
├── 📁 data/                         # Données et résultats
│   ├── 📁 examples/                 # Exemples de réseaux
│   │   └── ebolowa_reseau.inp      # Réseau d'Ebolowa
│   └── 📁 results/                  # Résultats générés
│       ├── 📁 visualisation/        # Graphiques PNG
│       ├── 📁 rapports/            # Rapports CSV
│       └── 📁 reseaux_optimises/   # Fichiers INP optimisés
├── main.py                          # Point d'entrée principal
├── run_gui.py                       # Lancement interface graphique
├── run_tests.py                     # Lancement des tests
└── requirements.txt                 # Dépendances Python
```

## 🔬 Algorithmes Implémentés

### Optimisation Mono-objectif
- **Objectif** : Minimisation des pertes de charge totales
- **Algorithme** : Algorithme génétique classique
- **Sélection** : Tournoi
- **Croisement** : Deux points
- **Mutation** : Discrète sur diamètres standards

### Optimisation Multi-objectif (NSGA-II)
- **Objectifs simultanés** :
  1. 🔻 Minimiser les pertes de charge
  2. ⚡ Optimiser les vitesses d'écoulement
  3. 📊 Uniformiser les pressions
- **Front de Pareto** : Solutions non-dominées
- **Diversité** : Maintenue par distance de crowding

## 📊 Types de Résultats Générés

### 📈 Visualisations
- `front_pareto.png` - Front de Pareto 3D
- `convergence_amelioree.png` - Courbes de convergence
- `carte_chaleur_amelioree.png` - Cartes thermiques des performances
- `analyse_solutions_pareto.png` - Analyse détaillée des solutions
- `distribution_objectifs.png` - Histogrammes des objectifs

### 📋 Rapports
- `resume_optimisation.csv` - Résumé global mono-objectif
- `resume_multi_objectif.csv` - Résumé multi-objectif
- `solutions_pareto.csv` - Détail de toutes les solutions Pareto
- `indicateurs_performance.csv` - Métriques de qualité
- `comparaison_diametres.csv` - Comparaison avant/après

### 🔧 Fichiers INP Optimisés
- `reseau_optimise_mono_YYYYMMDD_HHMMSS.inp` - Solution mono-objectif
- `reseau_optimise_multi_*.inp` - Solutions multi-objectif spécialisées
- `statistiques_optimisation.csv` - Statistiques des modifications

## ⚙️ Configuration

Tous les paramètres sont configurables dans `src/core/config.py` :

```python
# Paramètres d'optimisation
TAILLE_POPULATION = 100
NOMBRE_GENERATIONS = 50
TAUX_CROISEMENT = 0.8
TAUX_MUTATION_INDIVIDU = 0.2

# Contraintes hydrauliques
PRESSION_MIN = 20.0    # mCE
PRESSION_MAX = 60.0    # mCE
VITESSE_MIN = 0.5      # m/s
VITESSE_MAX = 1.5      # m/s

# Diamètres disponibles (mm)
DIAMETRES_DISPONIBLES = [40, 63, 75, 90, 110, 160, 200, 250, 315, 400]
```

## 🧪 Tests et Qualité

Le projet inclut une suite complète de tests unitaires :

```bash
python run_tests.py
```

**Tests couverts :**
- ✅ Chargement et validation des réseaux
- ✅ Algorithmes d'optimisation
- ✅ Génération de rapports et visualisations
- ✅ Gestion des encodages de fichiers
- ✅ Configuration et paramètres

## 📚 Cas d'Usage

### 🏙️ Réseau Municipal
Optimisation du réseau de distribution d'eau de la ville d'Ebolowa :
- **34 nœuds** de consommation
- **39 conduites** à dimensionner
- **Contraintes réglementaires** de pression et vitesse

### 🔬 Recherche Académique
- Comparaison d'algorithmes d'optimisation
- Analyse de sensibilité des paramètres
- Études de cas multi-objectif

### 🏭 Applications Industrielles
- Réseaux de refroidissement
- Systèmes de distribution industrielle
- Optimisation énergétique

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment participer :

1. **Fork** le projet
2. Créer une **branche** pour votre fonctionnalité
3. **Committer** vos changements
4. **Pousser** vers la branche
5. Ouvrir une **Pull Request**

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Développeur Principal** - Optimisation et algorithmes génétiques
- **Contributeurs** - Voir [Contributors](https://github.com/ngueleEbale/GA_implementation/contributors)

## 🙏 Remerciements

- **WNTR** pour la simulation hydraulique
- **DEAP** pour les algorithmes évolutionnaires
- **EPANET** pour les standards de modélisation hydraulique
- La communauté **open-source** pour les outils et bibliothèques

---

<div align="center">

**⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile ! ⭐**

[🐛 Signaler un Bug](https://github.com/ngueleEbale/GA_implementation/issues) • [💡 Suggérer une Fonctionnalité](https://github.com/ngueleEbale/GA_implementation/issues) • [📖 Documentation](https://github.com/ngueleEbale/GA_implementation/wiki)

</div>