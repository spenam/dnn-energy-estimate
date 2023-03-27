This project uses a Neural Network to combine information about the energy reconstructions with the triggered hits to obtain a better energy estimate.
======================================================================================================================================================

.. image:: https://git.km3net.de/spenamartinez/dnn_energy_estimate/badges/master/pipeline.svg
    :target: https://git.km3net.de/spenamartinez/dnn_energy_estimate/pipelines

.. image:: https://git.km3net.de/spenamartinez/dnn_energy_estimate/badges/master/coverage.svg
    :target: https://spenamartinez.pages.km3net.de/dnn_energy_estimate/coverage

.. image:: https://git.km3net.de/examples/km3badges/-/raw/master/docs-latest-brightgreen.svg
    :target: https://spenamartinez.pages.km3net.de/dnn_energy_estimate


Installation
~~~~~~~~~~~~

It is recommended to first create an isolated virtualenvironment to not interfere
with other Python projects::

  git clone https://git.km3net.de/spenamartinez/dnn_energy_estimate
  cd dnn_energy_estimate
  python3 -m venv venv
  . venv/bin/activate

Install directly from the Git server via ``pip`` (no cloneing needed)::

  pip install git+https://git.km3net.de/spenamartinez/dnn_energy_estimate

Or clone the repository and run::

  make install

To install all the development dependencies, in case you want to contribute or
run the test suite::

  make install-dev
  make test


---

*Created with ``cookiecutter https://git.km3net.de/templates/python-project``*
