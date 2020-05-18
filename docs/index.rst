.. PySAN documentation master file, created by
   sphinx-quickstart on Mon May 18 19:34:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySAN's documentation!
=================================

PySAN is a library of methods for social sequence analysis using Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference



User Guide
===============
PySAN revolves around analysing sequences, which can be represented in Python using lists (comma seperated values inside square brackets);

.. ipython:: python

	sequence = [0,1,1,2,3,1,3]

	print(sequence)



.. ipython:: python

	import pysan as ps

	@savefig basic_sequence.png
	ps.plot_sequence(sequence)


.. ipython:: python
	
	ps.describe(sequence)

