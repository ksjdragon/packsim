Squish: Soft Packing Simulations
================================

**Squish** is a Python library that handles simulations for the flow of 'soft' or 'compressible' objects,
under some energy in a *periodic* domain. The gradient flow of the objects are simulated and is able to be saved and rendered into diagrams. **Squish** also provides simple to use scripts for basic usage, as well as a simple API for more advanced functionality.

Fast parallelized Cython code is used to compute the simulations efficienctly. Also, `NumPy <https://numpy.org>`_ arrays are used, enabling easy integratation with the standard scientific Python ecosystem.

Resources
---------

- `Documentation <https://squish.readthedocs.io>`_: Examples, guides, and **Squish** API.
- `GitHub repository <https://github.com/ksjdragon/squish>`_: View and download the **Squish** source code.
- `Issue tracker <https://github.com/ksjdragon/squish/issues>`_: Report issues or request features for **Squish**.


Installation
------------
The simplest way to install **Squish** is to use ``pip``:

.. code-block:: bash

   (.venv) /path/to/squish: pip install squish

License
-------
This project uses `GNU AGPLv3.0 <https://choosealicense.com/licenses/agpl-3.0/>`_.

Support and Contribution
------------------------
Feel free to visit our repository on `GitHub <https://github.com/ksjdragon/squish>`_ for source code of this library. Any issues or bugs may be reported at  the `issue tracker <https://github.com/ksjdragon/squish/issues>`_. All contributions to **Squish** are welcome!