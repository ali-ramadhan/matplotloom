import os
import sys
from datetime import datetime

# Add matplotloom package to path.
sys.path.insert(0, os.path.abspath(".."))

import matplotloom

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "matplotloom"
copyright = f"2024-{datetime.now().year}, Ali Ramadhan"
author = "Ali Ramadhan"
version = matplotloom.__version__
release = matplotloom.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_theme_options = {
    "nosidebar": True
}
