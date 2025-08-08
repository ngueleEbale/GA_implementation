#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour lancer l'interface graphique
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from gui.gui import Application
    app = Application()
    app.mainloop() 