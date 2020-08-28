.. `pysan <https://github.com/pysan-dev/pysan>`_ documentation master file, created by
   sphinx-quickstart on Mon May 18 19:34:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome
=================================

`pysan <https://github.com/pysan-dev/pysan>`_ is a library of methods for `social sequence analysis <https://www.cambridge.org/core/books/social-sequence-analysis/3AC786DA3C99EB8795C7271BB350CB88>`_ using Python.

Why use pysan?
===============
Analyses in the sequence domain can help explain the order in which events occur.
In science, this can be applied to anything from communication data, to consumer purchasing behaviour, to gambling behaviours.
`pysan <https://github.com/pysan-dev/pysan>`_ contains methods for visualising, comparing, and dissecting sequences, helping you develop insights into your data with only a few lines of code.

To find out more about sequence analysis using pysan, continue reading down this page, starting with how to install the latest version from PyPI.

:pandas:

Installation
===============
Install pysan using the following pip command;

.. code-block::

	pip install pysan


User Guide
===============
`pysan <https://github.com/pysan-dev/pysan>`_ is designed to analyse sequences, which can be represented in Python using lists (comma seperated values inside square brackets);

.. ipython:: python

	import pysan as ps

Loading Sequence Data
-----------------------

.. ipython:: python

	sequence = [0,1,1,2,3,1,3]
	print(sequence)


You can get descriptive information on sequence using the `describe` method;

.. ipython:: python

	ps.describe(sequence)

Basic Visualisation
----------------------

.. ipython:: python

	sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,3,3,2,1,1,2]
	@savefig plot_sequence.png
	plot = ps.plot_sequence(sequence)


Understanding Subsequences
----------------------------


Make a Transition Matrix
---------------------------







API Reference
=================

Method Signatures
-------------------

`pysan <https://github.com/pysan-dev/pysan>`_'s core module contains a collection of useful methods for exploring sequence data. see below for a list of 

.. automodsumm:: pysan.core
	:functions-only:

All Methods
---------------


.. automodule:: pysan.core
	:members:
	:undoc-members: