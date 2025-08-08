#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour l'optimisation de réseau hydraulique
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Fonction principale"""
    print("🚀 Optimisation de Réseau Hydraulique")
    print("=" * 50)
    print("1. Interface graphique")
    print("2. Interface ligne de commande")
    print("3. Tests")
    print("4. Quitter")
    
    while True:
        try:
            choix = input("\nVotre choix (1-4) : ").strip()
            
            if choix == "1":
                print("\n🎨 Lancement de l'interface graphique...")
                from gui.gui import Application
                app = Application()
                app.mainloop()
                break
                
            elif choix == "2":
                print("\n💻 Lancement de l'interface ligne de commande...")
                print("Utilisez : python -m src.cli.optimisation_cli --help")
                from cli.optimisation_cli import main as cli_main
                cli_main()
                break
                
            elif choix == "3":
                print("\n🧪 Lancement des tests...")
                from tests.tests_complets import main as test_main
                test_main()
                break
                
            elif choix == "4":
                print("\n👋 Au revoir !")
                break
                
            else:
                print("❌ Choix invalide. Veuillez choisir 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main() 