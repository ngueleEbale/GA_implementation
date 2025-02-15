import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import optimisation  # Votre fichier "optimisation.py"

# --------------------------------------------------
# 1) Petite classe pour rediriger stdout/stderr
# --------------------------------------------------
class TextRedirector:
    """
    Permet de rediriger toutes les écritures sur sys.stdout ou sys.stderr
    vers un widget Text/ScrolledText Tkinter.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Fait défiler la zone de texte

    def flush(self):
        pass  # Méthode requise en Python 3, laisse vide


# --------------------------------------------------
# 2) Classe principale de l'application Tkinter
# --------------------------------------------------
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimisation d'un réseau hydraulique")

        # ---------------------------------------
        # (1) Charger l'image de fond SANS PIL
        # ---------------------------------------
        try:
            self.background_image = tk.PhotoImage(file="background.png")
        except Exception as e:
            # Si l'image ne se charge pas, on met None
            print("Impossible de charger l'image de fond :", e)
            self.background_image = None

        # ---------------------------------------
        # (2) Fixer la taille de la fenêtre à celle de l'image
        # et empêcher son redimensionnement
        # ---------------------------------------
        if self.background_image:
            w = self.background_image.width()
            h = self.background_image.height() +250
            self.geometry(f"{w}x{h}")
            self.resizable(False, True)  # Empêche le redimensionnement

        # ---------------------------------------
        # (3) Placer le label qui couvre toute la fenêtre
        # ---------------------------------------
        if self.background_image:
            bg_label = tk.Label(self, image=self.background_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # ---------------------------------------
        # Création des variables associées aux champs
        # ---------------------------------------
        self.fichier_inp   = tk.StringVar(value="")
        self.pression_min  = tk.DoubleVar(value=optimisation.PRESSION_MIN)
        self.pression_max  = tk.DoubleVar(value=optimisation.PRESSION_MAX)
        self.diam_min      = tk.IntVar(value=optimisation.DIAMETRE_MIN)
        self.diam_max      = tk.IntVar(value=optimisation.DIAMETRE_MAX)
        self.vitesse_min   = tk.DoubleVar(value=optimisation.VITESSE_MIN)
        self.vitesse_max   = tk.DoubleVar(value=optimisation.VITESSE_MAX)
        self.taille_pop    = tk.IntVar(value=optimisation.TAILLE_POPULATION)
        self.nb_gen        = tk.IntVar(value=optimisation.NOMBRE_GENERATIONS)
        self.taux_crois    = tk.DoubleVar(value=optimisation.TAUX_CROISEMENT)

        # Construit l'interface
        self._creer_widgets()

        # Redirige stdout et stderr dans le ScrolledText
        self._redirect_console()

    def _creer_widgets(self):
        """
        Construit les champs et boutons de l'IHM.
        """
        # Au lieu d'utiliser grid (qui s'adapte moins bien quand on fixe la taille),
        # on peut utiliser place() ou pack(). Ici, on reste avec grid() pour l'exemple.
        row = 0

        # 1) Fichier INP
        tk.Label(self, text="Fichier .inp :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.fichier_inp, width=50).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(self, text="Charger...", command=self._charger_fichier_inp)\
          .grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 2) Pression min / max
        tk.Label(self, text="Pression min (m) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.pression_min).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(self, text="Pression max (m) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.pression_max).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # 3) Diamètre min / max
        tk.Label(self, text="Diamètre min (mm) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.diam_min).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(self, text="Diamètre max (mm) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.diam_max).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # 4) Vitesse min / max
        tk.Label(self, text="Vitesse min (m/s) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.vitesse_min).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(self, text="Vitesse max (m/s) :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.vitesse_max).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # 5) Population, Nb gén, Taux crois
        tk.Label(self, text="Taille population :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.taille_pop).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(self, text="Nb Générations :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.nb_gen).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(self, text="Taux Croisement :", bg="#dceeff").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.taux_crois).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # 6) Boutons
        # Mono-objectif
        tk.Button(self, text="Lancer Optimisation (mono)", command=self._lancer_optimisation)\
          .grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # Multi-objectif
        tk.Button(self, text="Lancer Optimisation (multi)", command=self._lancer_optimisation_multi)\
          .grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # Visualisation
        tk.Button(self, text="Afficher Visualisation", command=self._afficher_visualisation)\
          .grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # 7) Zone console (ScrolledText) pour afficher les print
        self.console = scrolledtext.ScrolledText(self, wrap='word', width=70, height=10)
        self.console.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        row += 1

        # (Optionnel) Ajustements si vous voulez un ratio flexible
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_rowconfigure(row, weight=1)

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

    def _lancer_optimisation(self):
        """Optimisation mono-objectif."""
        inp_file = self.fichier_inp.get()
        if not inp_file:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier .inp.")
            return

        # Maj des paramètres globaux dans optimisation.py
        optimisation.PRESSION_MIN      = self.pression_min.get()
        optimisation.PRESSION_MAX      = self.pression_max.get()
        optimisation.DIAMETRE_MIN      = self.diam_min.get()
        optimisation.DIAMETRE_MAX      = self.diam_max.get()
        optimisation.VITESSE_MIN       = self.vitesse_min.get()
        optimisation.VITESSE_MAX       = self.vitesse_max.get()
        optimisation.TAILLE_POPULATION = self.taille_pop.get()
        optimisation.NOMBRE_GENERATIONS= self.nb_gen.get()
        optimisation.TAUX_CROISEMENT   = self.taux_crois.get()

        try:
            print("=== Lancement de l'optimisation MONO-OBJECTIF ===")
            opt = optimisation.OptimisationReseau(inp_file)

            def callback_generations(gen):
                # Actualiser la fenêtre pour montrer la progression
                self.update_idletasks()

            opt.executer_optimisation(callback=callback_generations)

            # Visualisations
            opt._plot_convergence()
            opt._plot_distribution_pressions()
            opt._plot_evolution_temporelle()
            opt._plot_carte_chaleur_pertes()
            opt.generer_comparaison()

            print("=== Optimisation MONO terminée avec succès ===\n")
            messagebox.showinfo("Succès", "Optimisation MONO terminée ! Consultez la console.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de l'optimisation : {e}")

    def _lancer_optimisation_multi(self):
        """Optimisation multi-objectif (NSGA-II)."""
        inp_file = self.fichier_inp.get()
        if not inp_file:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier .inp.")
            return

        # Maj des paramètres
        optimisation.PRESSION_MIN      = self.pression_min.get()
        optimisation.PRESSION_MAX      = self.pression_max.get()
        optimisation.DIAMETRE_MIN      = self.diam_min.get()
        optimisation.DIAMETRE_MAX      = self.diam_max.get()
        optimisation.VITESSE_MIN       = self.vitesse_min.get()
        optimisation.VITESSE_MAX       = self.vitesse_max.get()
        optimisation.TAILLE_POPULATION = self.taille_pop.get()
        optimisation.NOMBRE_GENERATIONS= self.nb_gen.get()
        optimisation.TAUX_CROISEMENT   = self.taux_crois.get()

        try:
            print("=== Lancement de l'optimisation MULTI-OBJECTIF (NSGA-II) ===")
            opt = optimisation.OptimisationReseau(inp_file)

            def callback_generations(gen):
                self.update_idletasks()

            opt.executer_optimisation_multi(callback=callback_generations)

            # Visualisation du front Pareto
            opt._plot_pareto()

            print("=== Optimisation MULTI terminée avec succès ===\n")
            messagebox.showinfo("Succès", "Optimisation MULTI terminée ! Consultez la console.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de l'optimisation multi : {e}")

    def _afficher_visualisation(self):
        """Message d’information sur où trouver les figures générées."""
        messagebox.showinfo(
            "Visualisation",
            "Les figures (convergence, distributions, etc.) se trouvent dans le dossier 'visualisation'."
        )


# --------------------------------------------------
# 3) Lancement de l'application
# --------------------------------------------------
if __name__ == "__main__":
    app = Application()
    app.mainloop()
