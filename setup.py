"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['MOM_GUI_v09_Main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': ['django', 'mysql'],
    'includes': [],
    'excludes': ['PyInstaller'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
