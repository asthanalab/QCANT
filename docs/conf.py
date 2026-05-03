"""Sphinx configuration for the QCANT documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import QCANT


project = "QCANT"
author = "Asthana Lab"
copyright = "2026, Asthana Lab"
version = QCANT.__version__
release = QCANT.__version__

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "default"

html_theme = "pydata_sphinx_theme"
html_title = "QCANT"
html_static_path = ["_static"]
html_css_files = ["qcant.css"]
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "navbar_align": "left",
    "github_url": "https://github.com/asthanalab/QCANT",
}

htmlhelp_basename = "QCANTdoc"

latex_documents = [
    (master_doc, "QCANT.tex", "QCANT Documentation", author, "manual"),
]

man_pages = [
    (master_doc, "QCANT", "QCANT Documentation", [author], 1),
]

texinfo_documents = [
    (
        master_doc,
        "QCANT",
        "QCANT Documentation",
        author,
        "QCANT",
        "Quantum chemistry algorithms and acceleration utilities for near-term quantum computing research.",
        "Scientific/Engineering",
    ),
]
