# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Squish'
copyright = '2021, ksjdragon'
author = 'ksjdragon'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_static_path= ['_static']
html_theme = 'sphinx_rtd_theme'
html_css_files = [
    "ellipses.css"
]

# -- Options for EPUB output
epub_show_urls = 'footnote'
