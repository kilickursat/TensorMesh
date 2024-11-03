# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 
import datetime 
sys.path.insert(0,"../..")
import tensormesh    
import tensormesh_sphinx_theme


project = 'tensormesh'
author  = 'walkerchi'
copyright = f'{datetime.datetime.now().year}, {author}'
version = tensormesh.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
     'sphinx.ext.autodoc',
     'sphinx.ext.napoleon',
     'sphinx.ext.viewcode',
     'sphinx.ext.mathjax',
     'sphinx.ext.githubpages',
     'sphinx.ext.intersphinx',
     'sphinx.ext.autosummary',
     'nbsphinx'
]


html_theme = 'tensormesh_sphinx_theme'
html_theme_options = {
    'navigation_depth': 4,  # Increase navigation depth
    'collapse_navigation': False,  # Keep the navigation expanded
    'sticky_navigation': True,     # Keep the sidebar fixed during scrolling
    'includehidden': True,         # Include hidden toctree elements
}

# Make sure these settings are present and correct
html_sidebars = {
    '**': [
        'globaltoc.html',  # Table of contents
        'searchbox.html',  # Search box
        'relations.html',  # Previous/next buttons
    ]
}

html_logo  = ('https://raw.githubusercontent.com/walkerchi/tensormesh_sphinx_theme/'
             'master/tensormesh_sphinx_theme/static/img/tensormesh_logo.webp')
html_favicon =('https://raw.githubusercontent.com/walkerchi/tensormesh_sphinx_theme/'
             'master/tensormesh_sphinx_theme/static/img/tensormesh_logo.webp')
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

exclude_patterns = []

napoleon_google_docstring = False

autosummary_generate = True

autodoc_member_order = 'groupwise'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output




def skip(app, what, name, obj, skip, options):
     print(f"what: {what}, name: {name}, obj: {obj}, skip: {skip}, options: {options}\n")
     if hasattr(obj, '__autodoc__'):
          return not name in obj.__autodoc__
     
     return skip

def setup(app):
     app.connect('autodoc-skip-member', skip)
     app.add_css_file('custom.css')

# These settings can also help with documentation structure
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Enable better section numbering
numfig = True
numfig_secnum_depth = 2