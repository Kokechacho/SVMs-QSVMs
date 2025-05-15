# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'SVMs and QSVM'
copyright = '2025, Álvaro Sánchez-Paniagua'
author = 'Álvaro Sánchez-Paniagua'
release = '-'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # si usas docstrings estilo Google o NumPy
]

templates_path = ['_templates']
exclude_patterns = []

language = 'esp'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'

latex_elements = {
    'inputenc': '',
    'fontenc': '',
    'preamble': r'''
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{amsmath,amssymb}
\usepackage{newunicodechar}
\newunicodechar{α}{\ensuremath{\alpha}}
''',
}

sys.path.insert(0, os.path.abspath('../../'))  # ajusta la ruta a tu proyecto