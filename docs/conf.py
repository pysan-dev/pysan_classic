# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os, sys, time
from urllib.parse import quote
# get current directory and cut off the '/docs' so sphinx can find the source code
sys.path.append(os.getcwd()[:-5])

# -- Project information -----------------------------------------------------

project = 'pysan'
copyright = '2020, Oliver J. Scholten'
author = 'Oliver J. Scholten'

# The full version, including alpha/beta/rc tags
release = '0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinxcontrib.napoleon',
    'sphinx_automodapi.automodapi',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_gallery.gen_gallery',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.autosectionlabel'
]

plot_include_source = True
plot_pre_code = '''
import sys
sys.path.append('../../')
import pysan as ps
'''
plot_rcparams = {'savefig.bbox': 'tight'}
doctest_test_doctest_blocks = 'nonemptystring'
doctest_global_setup = '''
import sys
sys.path.append('../../')
import pysan as ps
'''
sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

autodoc_default_options = {
    'member-order': 'bysource'
}
#intersphinx_mapping = {'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None)}

#extlinks = {'pandas':('http://pandas.pydata.org', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

add_module_names = False

from better import better_theme_path
html_theme_path = [better_theme_path]
html_theme = 'bizstyle'
html_theme_options = {}
html_css_files = ['pysan_style.css']

html_sidebars = { '**': ['sidebar.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_title='pysan documentation'