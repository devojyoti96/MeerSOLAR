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
    "sphinxcontrib.programoutput",  
    "sphinx_togglebutton",
    "myst_parser",
    'sphinx.ext.graphviz',
    "sphinxcontrib.mermaid",

]
html_theme_options = {
    "source_repository": "https://github.com/devojyoti96/MeerSOLAR/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = "furo"
html_static_path = ['_static']
html_title = "MeerSOLAR"
html_css_files = ['custom.css']
sphinx_togglebutton_selector = ".toggle-this-element, #my-special-id"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "casatools",
    "casatasks",
]
myst_enable_extensions = [
    "colon_fence",
    "attrs_inline",
    "deflist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

