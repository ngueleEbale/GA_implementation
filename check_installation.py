#!/usr/bin/env python3
"""
Script de vérification de l'installation du projet GA_implementation
"""

import sys
import os
import importlib.util

def check_python_version():
    """Vérifier la version de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Version Python compatible")
        return True
    else:
        print("   ❌ Python 3.8+ requis")
        return False

def check_dependencies():
    """Vérifier les dépendances principales"""
    dependencies = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'wntr', 'deap', 'tkinter'
    ]
    
    print("\n📦 Vérification des dépendances :")
    all_ok = True
    
    for dep in dependencies:
        try:
            if dep == 'tkinter':
                import tkinter
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - Non installé")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Vérifier la structure du projet"""
    print("\n📁 Vérification de la structure :")
    
    required_paths = [
        'src/core/optimisation.py',
        'src/core/config.py',
        'src/gui/gui.py',
        'src/cli/optimisation_cli.py',
        'src/tests/tests_complets.py',
        'data/examples/ebolowa_reseau.inp',
        'main.py',
        'run_gui.py',
        'run_tests.py'
    ]
    
    all_ok = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"   ✅ {path}")
        else:
            print(f"   ❌ {path} - Manquant")
            all_ok = False
    
    return all_ok

def check_imports():
    """Vérifier les imports du projet"""
    print("\n🔧 Vérification des imports :")
    
    try:
        sys.path.append('src')
        from core import config
        print("   ✅ core.config")
        
        from core import optimisation
        print("   ✅ core.optimisation")
        
        from gui import gui
        print("   ✅ gui.gui")
        
        return True
    except Exception as e:
        print(f"   ❌ Erreur d'import : {e}")
        return False

def main():
    """Fonction principale"""
    print("🚰 Vérification de l'installation GA_implementation")
    print("=" * 50)
    
    checks = [
        ("Version Python", check_python_version),
        ("Dépendances", check_dependencies),
        ("Structure", check_project_structure),
        ("Imports", check_imports)
    ]
    
    results = []
    for name, check_func in checks:
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 Installation complète et fonctionnelle !")
        print("\n🚀 Vous pouvez maintenant utiliser :")
        print("   • python main.py          - Menu principal")
        print("   • python run_gui.py       - Interface graphique")
        print("   • python run_tests.py     - Tests unitaires")
    else:
        print("❌ Des problèmes ont été détectés.")
        print("\n🔧 Actions recommandées :")
        print("   • pip install -r requirements.txt")
        print("   • Vérifier la structure du projet")
        
if __name__ == "__main__":
    main()