.. PySAN documentation master file, created by
   sphinx-quickstart on Mon May 18 19:34:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome
=================================

PySAN is a library of methods for `social sequence analysis <https://www.cambridge.org/core/books/social-sequence-analysis/3AC786DA3C99EB8795C7271BB350CB88>`_ using Python.
Analyses in the sequence domain concern the order in which events occur - to help explore such data, PySAN provides a collection of descriptive, analytical, and visualisation methods.


Installation
===============

.. code-block::

	pip install pysan


User Guide
===============
PySAN revolves around analysing sequences, which can be represented in Python using lists (comma seperated values inside square brackets);

.. ipython:: python

	import pysan as ps



.. ipython:: python

	sequence = [0,1,1,2,3,1,3]
	print(sequence)



.. ipython:: python

	sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,3,3,2,1,1,2]
	@savefig plot_sequence.png width=4in
	plot = ps.plot_sequence(sequence)


You can get descriptive information on sequence using the `describe` method;

.. ipython:: python

	ps.describe(sequence)





API Reference
=================

Method Signatures
-------------------

PySAN's core module contains a collection of useful methods for exploring sequence data. see below for a list of 

.. automodsumm:: pysan.core
	:functions-only:

All Methods
---------------

.. automodule:: pysan.core
	:members:
	:undoc-members: