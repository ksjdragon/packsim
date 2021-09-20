Installation
============

Basic Installation
------------------

.. important::

   This library requires Python 3.8+! Also, if you need to export diagrams
   as a video, the ``ffmpeg`` library must be installed on your system and in your ``PATH``!

To use **Squish**, first install it using `pip`:

.. code-block:: console

   (.venv) $ pip install squish

The **Squish** library depends on matplotlib, numpy, and scipy, which all will be installed along with it.

Documentation
-------------

This documentation for **Squish** is `hosted online <https://squish.readthedocs.io>`_ at **ReadTheDocs**. However, you may also build the documentation for yourself locally.


Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

You will need to install ``sphinx`` and ``sphinx_rtd_theme`` packages via ``pip``:

.. code-block:: bash

   (.venv) /path/to/squish: pip install sphinx sphinx_rtd_theme

Then, to build, run the go into the documentation source directory and execute ``make``:

.. code-block:: bash

   (.venv) /path/to/squish: cd docs
   (.venv) /path/to/squish/docs: make html

.. autosummary::
   :toctree: generated
