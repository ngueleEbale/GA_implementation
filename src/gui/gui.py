# -*- coding: utf-8 -*-
"""
Interface Graphique pour l'Optimisation de Réseaux Hydrauliques
==============================================================

Ce module implémente une interface graphique moderne basée sur Tkinter pour
le système d'optimisation de réseaux hydrauliques. Il fournit une interface
intuitive permettant de configurer, lancer et suivre les optimisations mono
et multi-objectif en temps réel.

L'interface intègre:
- Configuration des paramètres d'optimisation
- Sélection et validation des fichiers INP
- Lancement et contrôle des optimisations
- Suivi en temps réel avec barres de progression
- Affichage des logs et résultats
- Gestion des arrêts d'urgence

Classes principales:
-------------------
- Application: Interface principale Tkinter
- TextRedirector: Redirection des sorties console vers l'interface

Author: Équipe d'Optimisation Hydraulique
Version: 3.0
Date: 2025
License: MIT
"""

# Imports système et configuration des chemins
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports interface graphique
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter import font as tkfont

# Imports logique métier
from core import optimisation, config

# Imports utilitaires
import json
import threading
from datetime import datetime

# ========================================================================
# CLASSE UTILITAIRE - REDIRECTION DES SORTIES CONSOLE
# ========================================================================

class TextRedirector:
    """
    Classe utilitaire pour rediriger stdout/stderr vers un widget Tkinter.
    
    Cette classe permet d'afficher en temps réel dans l'interface graphique
    tous les messages normalement destinés à la console, offrant une
    expérience utilisateur intégrée et professionnelle.
    
    Attributes:
    -----------
    text_widget : tk.Text
        Widget Tkinter où rediriger les sorties textuelles
    
    Methods:
    --------
    write(message): Écrit un message dans le widget
    flush(): Méthode requise pour l'interface file-like
    
    Examples:
    ---------
    >>> text_area = tk.Text(root)
    >>> redirector = TextRedirector(text_area)
    >>> sys.stdout = redirector  # Redirection des prints
    """
    
    def __init__(self, text_widget):
        """
        Initialise le redirecteur avec un widget de destination.
        
        Parameters:
        -----------
        text_widget : tk.Text or tk.scrolledtext.ScrolledText
            Widget Tkinter où afficher les messages redirigés
        """
        self.text_widget = text_widget

    def write(self, message):
        """
        Écrit un message dans le widget et fait défiler automatiquement.
        
        Cette méthode simule l'interface d'un fichier pour permettre
        la redirection de sys.stdout et sys.stderr.
        
        Parameters:
        -----------
        message : str
            Message à afficher dans le widget
        """
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Auto-scroll vers la fin
        self.text_widget.update_idletasks()  # Mise à jour immédiate

    def flush(self):
        """
        Méthode vide requise pour l'interface file-like.
        
        Tkinter gère automatiquement l'affichage, donc aucune action
        spécifique n'est nécessaire pour le flush.
        """
        pass

# ========================================================================
# CLASSE PRINCIPALE - INTERFACE GRAPHIQUE D'OPTIMISATION
# ========================================================================

class Application(tk.Tk):
    """
    Interface Graphique Principale pour l'Optimisation de Réseaux Hydrauliques
    =========================================================================
    
    Cette classe implémente une interface Tkinter moderne et complète pour
    l'optimisation de réseaux hydrauliques. Elle offre une expérience utilisateur
    intuitive avec des onglets organisés, des contrôles en temps réel et un
    suivi détaillé des optimisations.
    
    Fonctionnalités principales:
    ---------------------------
    - Interface à onglets pour organisation claire
    - Configuration interactive des paramètres
    - Sélection et validation des fichiers INP
    - Optimisation mono et multi-objectif
    - Suivi temps réel avec barres de progression
    - Contrôles d'arrêt d'urgence
    - Affichage intégré des logs et résultats
    - Sauvegarde/chargement des configurations
    
    Architecture de l'interface:
    ---------------------------
    - Onglet 1: Sélection et configuration du réseau
    - Onglet 2: Paramètres d'optimisation avancés
    - Onglet 3: Lancement et suivi des optimisations
    - Onglet 4: Affichage des logs et résultats
    
    Gestion de la concurrence:
    -------------------------
    - Optimisations exécutées dans des threads séparés
    - Interface reste réactive pendant les calculs
    - Mécanismes d'arrêt sécurisés
    - Mise à jour temps réel des barres de progression
    
    Attributes:
    -----------
    optimisation_en_cours : bool
        Indicateur d'état de l'optimisation
    thread_optimisation : threading.Thread
        Thread d'exécution de l'optimisation
    arret_demande : bool
        Flag pour demander l'arrêt de l'optimisation
    opt : optimisation.OptimisationReseau
        Instance du moteur d'optimisation
    
    Examples:
    ---------
    >>> app = Application()
    >>> app.mainloop()  # Lancement de l'interface
    """
    def __init__(self):
        super().__init__()
        self.title("Optimisation d'un réseau hydraulique - Interface avancée")
        
        # Configuration de la fenêtre
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Variables pour le suivi de l'optimisation
        self.optimisation_en_cours = False
        self.thread_optimisation = None
        self.arret_demande = False
        
        # Configuration des styles
        self._configurer_styles()
        
        # Chargement de l'image de fond
        self._charger_background()
        
        # Création des variables
        self._creer_variables()
        
        # Construction de l'interface
        self._creer_interface()
        
        # Redirection de la console
        self._redirect_console()
        
        # Chargement de la configuration par défaut
        self._charger_configuration_defaut()

    def _configurer_styles(self):
        """Configure les styles de l'interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configuration des couleurs
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'background': '#F5F5F5',
            'surface': '#FFFFFF',
            'text': '#212121',
            'text_secondary': '#757575'
        }

    def _charger_background(self):
        """Charge l'image de fond si disponible"""
        try:
            self.background_image = tk.PhotoImage(file="data/examples/background.png")
            bg_label = tk.Label(self, image=self.background_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Impossible de charger l'image de fond : {e}")
            self.background_image = None

    def _creer_variables(self):
        """Crée les variables Tkinter pour les champs"""
        # Paramètres du fichier
        self.fichier_inp = tk.StringVar(value="")
        
        # Paramètres hydrauliques
        self.pression_min = tk.DoubleVar(value=optimisation.PRESSION_MIN)
        self.pression_max = tk.DoubleVar(value=optimisation.PRESSION_MAX)
        self.diam_min = tk.IntVar(value=optimisation.DIAMETRE_MIN)
        self.diam_max = tk.IntVar(value=optimisation.DIAMETRE_MAX)
        self.vitesse_min = tk.DoubleVar(value=optimisation.VITESSE_MIN)
        self.vitesse_max = tk.DoubleVar(value=optimisation.VITESSE_MAX)
        
        # Paramètres d'optimisation
        self.taille_pop = tk.IntVar(value=optimisation.TAILLE_POPULATION)
        self.nb_gen = tk.IntVar(value=optimisation.NOMBRE_GENERATIONS)
        self.taux_crois = tk.DoubleVar(value=optimisation.TAUX_CROISEMENT)
        self.taux_mutation = tk.DoubleVar(value=optimisation.TAUX_MUTATION_INDIVIDU)
        self.taux_mutation_gene = tk.DoubleVar(value=optimisation.TAUX_MUTATION_GENE)
        
        # Paramètres avancés
        self.sauvegarde_periodique = tk.BooleanVar(value=True)
        self.frequence_sauvegarde = tk.IntVar(value=10)
        self.generations_sans_amelioration = tk.IntVar(value=10)
        
        # Variables de statut
        self.statut_texte = tk.StringVar(value="Prêt")
        self.progression_valeur = tk.DoubleVar(value=0.0)

    def _creer_interface(self):
        """Construit l'interface principale avec onglets"""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre principal
        titre = ttk.Label(main_frame, text="Optimisation de Réseau Hydraulique", 
                         font=('Arial', 16, 'bold'))
        titre.pack(pady=(0, 20))
        
        # Notebook (onglets)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Création des onglets
        self._creer_onglet_parametres()
        self._creer_onglet_optimisation()
        self._creer_onglet_avance()
        self._creer_onglet_console()
        
        # Barre de statut et progression
        self._creer_barre_statut(main_frame)

    def _creer_onglet_parametres(self):
        """Crée l'onglet des paramètres de base"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Paramètres")
        
        # Frame pour le fichier INP
        file_frame = ttk.LabelFrame(frame, text="Fichier de réseau", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Fichier .inp :").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(file_frame, textvariable=self.fichier_inp, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Charger...", 
                  command=self._charger_fichier_inp).grid(row=0, column=2, padx=5)
        
        # Frame pour les paramètres hydrauliques
        hydraulique_frame = ttk.LabelFrame(frame, text="Paramètres hydrauliques", padding=10)
        hydraulique_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Pression
        ttk.Label(hydraulique_frame, text="Pression min (m) :").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.pression_min, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(hydraulique_frame, text="Pression max (m) :").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.pression_max, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # Diamètre
        ttk.Label(hydraulique_frame, text="Diamètre min (mm) :").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.diam_min, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(hydraulique_frame, text="Diamètre max (mm) :").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.diam_max, width=15).grid(row=1, column=3, padx=5, pady=2)
        
        # Vitesse
        ttk.Label(hydraulique_frame, text="Vitesse min (m/s) :").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.vitesse_min, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(hydraulique_frame, text="Vitesse max (m/s) :").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(hydraulique_frame, textvariable=self.vitesse_max, width=15).grid(row=2, column=3, padx=5, pady=2)

    def _creer_onglet_optimisation(self):
        """Crée l'onglet des paramètres d'optimisation"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Optimisation")
        
        # Frame pour les paramètres génétiques
        genetique_frame = ttk.LabelFrame(frame, text="Paramètres génétiques", padding=10)
        genetique_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(genetique_frame, text="Taille population :").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(genetique_frame, textvariable=self.taille_pop, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(genetique_frame, text="Nombre générations :").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(genetique_frame, textvariable=self.nb_gen, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(genetique_frame, text="Taux croisement :").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(genetique_frame, textvariable=self.taux_crois, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(genetique_frame, text="Taux mutation (individu) :").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(genetique_frame, textvariable=self.taux_mutation, width=15).grid(row=1, column=3, padx=5, pady=2)
        
        ttk.Label(genetique_frame, text="Taux mutation (gène) :").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(genetique_frame, textvariable=self.taux_mutation_gene, width=15).grid(row=2, column=3, padx=5, pady=2)
        
        # Frame pour les boutons d'optimisation
        boutons_frame = ttk.LabelFrame(frame, text="Lancement de l'optimisation", padding=10)
        boutons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Boutons avec icônes et couleurs
        ttk.Button(boutons_frame, text="🚀 Optimisation Mono-objectif", 
                  command=self._lancer_optimisation).pack(fill=tk.X, pady=2)
        
        ttk.Button(boutons_frame, text="🎯 Optimisation Multi-objectif (NSGA-II)", 
                  command=self._lancer_optimisation_multi).pack(fill=tk.X, pady=2)
        
        ttk.Button(boutons_frame, text="📊 Afficher Visualisations", 
                  command=self._afficher_visualisation).pack(fill=tk.X, pady=2)
        
        # Bouton d'arrêt
        self.bouton_arret = ttk.Button(boutons_frame, text="⏹️ Arrêter Optimisation", 
                                      command=self._arreter_optimisation, state='disabled')
        self.bouton_arret.pack(fill=tk.X, pady=2)

    def _creer_onglet_avance(self):
        """Crée l'onglet des paramètres avancés"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Avancé")
        
        # Frame pour les paramètres de sauvegarde
        sauvegarde_frame = ttk.LabelFrame(frame, text="Sauvegarde et reprise", padding=10)
        sauvegarde_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(sauvegarde_frame, text="Sauvegarde périodique", 
                       variable=self.sauvegarde_periodique).grid(row=0, column=0, sticky="w", padx=5)
        
        ttk.Label(sauvegarde_frame, text="Fréquence (générations) :").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(sauvegarde_frame, textvariable=self.frequence_sauvegarde, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(sauvegarde_frame, text="Générations sans amélioration max :").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(sauvegarde_frame, textvariable=self.generations_sans_amelioration, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Frame pour la gestion des configurations
        config_frame = ttk.LabelFrame(frame, text="Gestion des configurations", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(config_frame, text="💾 Sauvegarder configuration", 
                  command=self._sauvegarder_configuration).pack(fill=tk.X, pady=2)
        
        ttk.Button(config_frame, text="📂 Charger configuration", 
                  command=self._charger_configuration).pack(fill=tk.X, pady=2)
        
        ttk.Button(config_frame, text="🔄 Restaurer défauts", 
                  command=self._charger_configuration_defaut).pack(fill=tk.X, pady=2)

    def _creer_onglet_console(self):
        """Crée l'onglet de la console"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Console")
        
        # Zone de console avec scroll
        self.console = scrolledtext.ScrolledText(frame, wrap='word', height=20, 
                                               font=('Consolas', 10))
        self.console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Boutons pour la console
        boutons_console = ttk.Frame(frame)
        boutons_console.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(boutons_console, text="🗑️ Effacer console", 
                  command=self._effacer_console).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(boutons_console, text="💾 Sauvegarder log", 
                  command=self._sauvegarder_log).pack(side=tk.LEFT)

    def _creer_barre_statut(self, parent):
        """Crée la barre de statut avec progression"""
        statut_frame = ttk.Frame(parent)
        statut_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Barre de progression
        self.progress_bar = ttk.Progressbar(statut_frame, variable=self.progression_valeur, 
                                          maximum=100, length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Label de statut
        ttk.Label(statut_frame, textvariable=self.statut_texte).pack(side=tk.LEFT)

    def _redirect_console(self):
        """Redirige stdout et stderr vers le ScrolledText self.console."""
        sys.stdout = TextRedirector(self.console)
        sys.stderr = TextRedirector(self.console)

    def _charger_fichier_inp(self):
        """Dialogue pour sélectionner un fichier .inp."""
        filepath = filedialog.askopenfilename(
            title="Sélectionner le fichier .inp",
            filetypes=[("EPANET INP", "*.inp"), ("Tous fichiers", "*.*")]
        )
        if filepath:
            self.fichier_inp.set(filepath)
            self.statut_texte.set(f"Fichier chargé : {os.path.basename(filepath)}")

    def _valider_parametres(self):
        """Valide les paramètres avant lancement de l'optimisation"""
        erreurs = []
        
        if not self.fichier_inp.get():
            erreurs.append("Veuillez sélectionner un fichier .inp")
        
        if self.pression_min.get() >= self.pression_max.get():
            erreurs.append("La pression min doit être inférieure à la pression max")
        
        if self.diam_min.get() >= self.diam_max.get():
            erreurs.append("Le diamètre min doit être inférieur au diamètre max")
        
        if self.vitesse_min.get() >= self.vitesse_max.get():
            erreurs.append("La vitesse min doit être inférieure à la vitesse max")
        
        if self.taille_pop.get() <= 0:
            erreurs.append("La taille de population doit être positive")
        
        if self.nb_gen.get() <= 0:
            erreurs.append("Le nombre de générations doit être positif")
        
        if not (0 <= self.taux_crois.get() <= 1):
            erreurs.append("Le taux de croisement doit être entre 0 et 1")
        
        if not (0 <= self.taux_mutation.get() <= 1):
            erreurs.append("Le taux de mutation doit être entre 0 et 1")
        
        if erreurs:
            messagebox.showerror("Erreurs de validation", "\n".join(erreurs))
            return False
        
        return True

    def _mettre_a_jour_parametres(self):
        """Met à jour les paramètres globaux dans optimisation.py"""
        optimisation.PRESSION_MIN = self.pression_min.get()
        optimisation.PRESSION_MAX = self.pression_max.get()
        optimisation.DIAMETRE_MIN = self.diam_min.get()
        optimisation.DIAMETRE_MAX = self.diam_max.get()
        optimisation.VITESSE_MIN = self.vitesse_min.get()
        optimisation.VITESSE_MAX = self.vitesse_max.get()
        optimisation.TAILLE_POPULATION = self.taille_pop.get()
        optimisation.NOMBRE_GENERATIONS = self.nb_gen.get()
        optimisation.TAUX_CROISEMENT = self.taux_crois.get()
        optimisation.TAUX_MUTATION_INDIVIDU = self.taux_mutation.get()
        optimisation.TAUX_MUTATION_GENE = self.taux_mutation_gene.get()

    def _lancer_optimisation(self):
        """Lance l'optimisation mono-objectif dans un thread séparé"""
        if self.optimisation_en_cours:
            messagebox.showwarning("Optimisation en cours", "Une optimisation est déjà en cours")
            return
        
        if not self._valider_parametres():
            return
        
        self.optimisation_en_cours = True
        self.arret_demande = False
        self.statut_texte.set("Optimisation mono-objectif en cours...")
        self.progression_valeur.set(0)
        
        # Activation du bouton d'arrêt
        self.bouton_arret.config(state='normal')
        
        # Lancement dans un thread séparé
        self.thread_optimisation = threading.Thread(target=self._executer_optimisation_mono)
        self.thread_optimisation.daemon = True
        self.thread_optimisation.start()

    def _executer_optimisation_mono(self):
        """Exécute l'optimisation mono-objectif"""
        try:
            self._mettre_a_jour_parametres()
            
            print("=== Lancement de l'optimisation MONO-OBJECTIF ===")
            opt = optimisation.OptimisationReseau(self.fichier_inp.get())

            def callback_generations(gen):
                # Mise à jour de la progression
                progression = (gen / self.nb_gen.get()) * 100
                self.progression_valeur.set(progression)
                self.statut_texte.set(f"Génération {gen}/{self.nb_gen.get()}")
                self.update_idletasks()

            def check_arret():
                return self.arret_demande

            opt.executer_optimisation(callback=callback_generations, arret_demande=check_arret)

            # Génération des visualisations et rapports
            print("📊 Génération des visualisations...")
            opt._plot_visualisations_ameliorees()
            
            print("📋 Génération des rapports...")
            opt.generer_rapports()
            
            print("📈 Génération de la comparaison...")
            opt.generer_comparaison()

            print("=== Optimisation MONO terminée avec succès ===\n")
            self.progression_valeur.set(100)
            self.statut_texte.set("Optimisation mono-objectif terminée")
            
            # Message de succès dans le thread principal
            self.after(0, lambda: messagebox.showinfo("Succès", "Optimisation MONO terminée ! Consultez la console."))
            
        except Exception as e:
            error_msg = str(e)
            print(f"Erreur lors de l'optimisation : {error_msg}")
            self.statut_texte.set("Erreur lors de l'optimisation")
            self.after(0, lambda: messagebox.showerror("Erreur", f"Échec de l'optimisation : {error_msg}"))
        finally:
            self.optimisation_en_cours = False
            self.arret_demande = False
            # Désactivation du bouton d'arrêt
            self.bouton_arret.config(state='disabled')

    def _lancer_optimisation_multi(self):
        """Lance l'optimisation multi-objectif dans un thread séparé"""
        if self.optimisation_en_cours:
            messagebox.showwarning("Optimisation en cours", "Une optimisation est déjà en cours")
            return
        
        if not self._valider_parametres():
            return
        
        self.optimisation_en_cours = True
        self.arret_demande = False
        self.statut_texte.set("Optimisation multi-objectif en cours...")
        self.progression_valeur.set(0)
        
        # Activation du bouton d'arrêt
        self.bouton_arret.config(state='normal')
        
        # Lancement dans un thread séparé
        self.thread_optimisation = threading.Thread(target=self._executer_optimisation_multi)
        self.thread_optimisation.daemon = True
        self.thread_optimisation.start()

    def _executer_optimisation_multi(self):
        """Exécute l'optimisation multi-objectif"""
        try:
            self._mettre_a_jour_parametres()
            
            print("=== Lancement de l'optimisation MULTI-OBJECTIF (NSGA-II) ===")
            opt = optimisation.OptimisationReseau(self.fichier_inp.get())

            def callback_generations(gen):
                # Mise à jour de la progression
                progression = (gen / self.nb_gen.get()) * 100
                self.progression_valeur.set(progression)
                self.statut_texte.set(f"Génération {gen}/{self.nb_gen.get()}")
                self.update_idletasks()

            def check_arret():
                return self.arret_demande

            opt.executer_optimisation_multi(callback=callback_generations, arret_demande=check_arret)

            # Génération des visualisations et rapports
            print("📊 Génération des visualisations multi-objectif...")
            opt._plot_visualisations_multi_objectif()
            
            print("📋 Génération des rapports multi-objectif...")
            opt.generer_rapports_multi_objectif()
            
            print("📈 Génération du front Pareto...")
            opt._plot_pareto()

            print("=== Optimisation MULTI terminée avec succès ===\n")
            self.progression_valeur.set(100)
            self.statut_texte.set("Optimisation multi-objectif terminée")
            
            # Message de succès dans le thread principal
            self.after(0, lambda: messagebox.showinfo("Succès", "Optimisation MULTI terminée ! Consultez la console."))
            
        except Exception as e:
            error_msg = str(e)
            print(f"Erreur lors de l'optimisation multi : {error_msg}")
            self.statut_texte.set("Erreur lors de l'optimisation multi")
            self.after(0, lambda: messagebox.showerror("Erreur", f"Échec de l'optimisation multi : {error_msg}"))
        finally:
            self.optimisation_en_cours = False
            self.arret_demande = False
            # Désactivation du bouton d'arrêt
            self.bouton_arret.config(state='disabled')

    def _arreter_optimisation(self):
        """Arrête l'optimisation en cours"""
        if self.optimisation_en_cours:
            self.arret_demande = True
            self.statut_texte.set("Arrêt de l'optimisation en cours...")
            print("=== Arrêt de l'optimisation demandé ===")
            messagebox.showinfo("Arrêt", "Arrêt de l'optimisation demandé. L'arrêt sera effectif à la fin de la génération en cours.")

    def _afficher_visualisation(self):
        """Affiche les visualisations générées"""
        messagebox.showinfo(
            "Visualisation",
            "Les figures (convergence, distributions, etc.) se trouvent dans le dossier 'visualisation'.\n\n"
            "Vous pouvez également ouvrir le fichier 'algo_gen.html' pour une visualisation interactive."
        )

    def _sauvegarder_configuration(self):
        """Sauvegarde la configuration actuelle"""
        config = {
            'fichier_inp': self.fichier_inp.get(),
            'pression_min': self.pression_min.get(),
            'pression_max': self.pression_max.get(),
            'diam_min': self.diam_min.get(),
            'diam_max': self.diam_max.get(),
            'vitesse_min': self.vitesse_min.get(),
            'vitesse_max': self.vitesse_max.get(),
            'taille_pop': self.taille_pop.get(),
            'nb_gen': self.nb_gen.get(),
            'taux_crois': self.taux_crois.get(),
            'taux_mutation': self.taux_mutation.get(),
            'taux_mutation_gene': self.taux_mutation_gene.get(),
            'sauvegarde_periodique': self.sauvegarde_periodique.get(),
            'frequence_sauvegarde': self.frequence_sauvegarde.get(),
            'generations_sans_amelioration': self.generations_sans_amelioration.get(),
            'date_sauvegarde': datetime.now().isoformat()
        }
        
        filepath = filedialog.asksaveasfilename(
            title="Sauvegarder la configuration",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Tous fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Succès", f"Configuration sauvegardée dans {filepath}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder : {e}")

    def _charger_configuration(self):
        """Charge une configuration sauvegardée"""
        filepath = filedialog.askopenfilename(
            title="Charger une configuration",
            filetypes=[("JSON", "*.json"), ("Tous fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Application de la configuration
                self.fichier_inp.set(config.get('fichier_inp', ''))
                self.pression_min.set(config.get('pression_min', optimisation.PRESSION_MIN))
                self.pression_max.set(config.get('pression_max', optimisation.PRESSION_MAX))
                self.diam_min.set(config.get('diam_min', optimisation.DIAMETRE_MIN))
                self.diam_max.set(config.get('diam_max', optimisation.DIAMETRE_MAX))
                self.vitesse_min.set(config.get('vitesse_min', optimisation.VITESSE_MIN))
                self.vitesse_max.set(config.get('vitesse_max', optimisation.VITESSE_MAX))
                self.taille_pop.set(config.get('taille_pop', optimisation.TAILLE_POPULATION))
                self.nb_gen.set(config.get('nb_gen', optimisation.NOMBRE_GENERATIONS))
                self.taux_crois.set(config.get('taux_crois', optimisation.TAUX_CROISEMENT))
                self.taux_mutation.set(config.get('taux_mutation', optimisation.TAUX_MUTATION_INDIVIDU))
                self.taux_mutation_gene.set(config.get('taux_mutation_gene', optimisation.TAUX_MUTATION_GENE))
                self.sauvegarde_periodique.set(config.get('sauvegarde_periodique', True))
                self.frequence_sauvegarde.set(config.get('frequence_sauvegarde', 10))
                self.generations_sans_amelioration.set(config.get('generations_sans_amelioration', 10))
                
                messagebox.showinfo("Succès", f"Configuration chargée depuis {filepath}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger la configuration : {e}")

    def _charger_configuration_defaut(self):
        """Charge la configuration par défaut"""
        self.pression_min.set(optimisation.PRESSION_MIN)
        self.pression_max.set(optimisation.PRESSION_MAX)
        self.diam_min.set(optimisation.DIAMETRE_MIN)
        self.diam_max.set(optimisation.DIAMETRE_MAX)
        self.vitesse_min.set(optimisation.VITESSE_MIN)
        self.vitesse_max.set(optimisation.VITESSE_MAX)
        self.taille_pop.set(optimisation.TAILLE_POPULATION)
        self.nb_gen.set(optimisation.NOMBRE_GENERATIONS)
        self.taux_crois.set(optimisation.TAUX_CROISEMENT)
        self.taux_mutation.set(optimisation.TAUX_MUTATION_INDIVIDU)
        self.taux_mutation_gene.set(optimisation.TAUX_MUTATION_GENE)

    def _effacer_console(self):
        """Efface le contenu de la console"""
        self.console.delete(1.0, tk.END)

    def _sauvegarder_log(self):
        """Sauvegarde le contenu de la console dans un fichier"""
        filepath = filedialog.asksaveasfilename(
            title="Sauvegarder le log",
            defaultextension=".txt",
            filetypes=[("Texte", "*.txt"), ("Tous fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.console.get(1.0, tk.END))
                messagebox.showinfo("Succès", f"Log sauvegardé dans {filepath}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder le log : {e}")

# --------------------------------------------------
# 3) Lancement de l'application
# --------------------------------------------------
if __name__ == "__main__":
    app = Application()
    app.mainloop()
