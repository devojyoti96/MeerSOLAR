# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys
sys.path.insert(0, os.path.abspath("../../meersolar"))  # Adjust path if needed

project = 'MeerSOLAR'
copyright = '2025, Devojyoti Kansabanik, Deepan Patra'
author = 'Devojyoti Kansabanik, Deepan Patra'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # For Google/NumPy-style docstrings
    "sphinx.ext.viewcode",          # Add [source] links to functions
    "sphinx_autodoc_typehints",     # Show type hints in docs
    "sphinx_copybutton",            # Optional: copy-paste button for code blocks
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']
html_title = "MeerSOLAR"

html_theme_options = {
    "light_logo": "yourlogo-light.png",
    "dark_logo": "yourlogo-dark.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}


napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "casatools",
    "casatasks",
]

