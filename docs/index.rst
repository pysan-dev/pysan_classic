.. `pysan <https://github.com/pysan-dev/pysan>`_ documentation master file, created by
   sphinx-quickstart on Mon May 18 19:34:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. meta::
	:twitter\:card: summary_large_image
	:twitter\:site: @ojscholten
	:twitter\:title: pysan\: Sequence Analysis in Python
	:twitter\:description: The pysan library is a free and open source collection of methods for doing sequence analysis. Get started now 
	:twitter\:image: https://pysan.org/pysan.png

.. container:: notice

	**Please note: pysan is currently in beta - parts of this page may be incomplete**


Welcome
=================================

`pysan <https://github.com/pysan-dev/pysan>`_ is a library of methods for doing `sequence analysis <https://www.cambridge.org/core/books/social-sequence-analysis/3AC786DA3C99EB8795C7271BB350CB88>`_ using Python. It's free, open source, and each method is fully documented with examples. Using `pysan <https://github.com/pysan-dev/pysan>`_ you'll be able to run powerful sequence analysis and visualisation tasks with just a few lines of code, and integrate or extend them into your other Python workflows.

.. image:: index-14_00.png

.. raw:: html
	
	<p class='links'> <a href='https://github.com/pysan-dev/pysan'><img src="https://img.shields.io/badge/-Visit%20Repo-grey?style=for-the-badge&logo=github"></a> <a href='https://pypi.org/project/PySAN/'><img src="https://img.shields.io/badge/-Download%20Package-grey?style=for-the-badge&logo=python&logoColor=white"></a> <a href='https://pypi.org/project/PySAN/'><img src="https://img.shields.io/badge/-Share%20Project-grey?style=for-the-badge&logo=twitter&logoColor=white"></a></p>



Why use pysan?
===============
Lots of events in the natural world happen in a particular order, from making a cup of tea, to getting promotions, and so on.
In science, this applies to everything from communication data, to consumer spending, and to gambling behaviour.
Analysing these events whilst preserving their sequential order requires analysis in the sequence domain.
The pysan library contains methods for visualising, comparing, and dissecting sequences, helping you develop insights into your sequences with only a few lines of code.

**Why Python?** - Python is quickly becoming *the* go-to language for data analysis across a number of research areas.
By writing a sequence analysis library in Python, analysts can take advantage of other tools written in Python, whilst writing simple and readable code.

To find out more about sequence analysis using pysan, continue reading down this page, starting with how to install the latest version from PyPI.


Installation
===============
Install pysan using the following pip command;

.. code-block::

	pip install pysan

You can also use the very latest version by cloning the github repository.

.. code-block::

	git clone https://github.com/pysan-dev/pysan.git

User Guide
===============
`pysan <https://github.com/pysan-dev/pysan>`_ is designed to analyse sequences, which can be represented in Python using lists.
The first step to any analysis using pysan is to import the library.
The convention used throughout this documentation is to import it as **ps**.

.. ipython:: python

	import pysan as ps

Loading Sequence Data
-----------------------
If you're familiar with python, load your sequences in as lists.
All methods in pysan accept a single list, or a list of lists, as input.
Each sequence should be represented as one list, encoded into integers.
If you're using multi-channel data, each position in your list should be a tuple, with each tuple containing the elements in each channel.

.. ipython:: python

	sequence = [0,1,1,2,3,1,3]
	multi_channel_sequence = [(1,2),(2,2),(3,2),(3,3)]


You can get descriptive information on sequence using the :func:`.describe` method;

.. ipython:: python

	ps.describe(sequence)

Basic Visualisation
----------------------
The simplest way to visualise a sequence is to plot each position along the x-axis, and each element along the y-axis.
This can be done using the :func:`.plot_sequence` method, which uses matplotlib's scatter plot to create a simple image.

.. ipython:: python

	sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,3,3,2,1,1,2]
	@savefig plot_sequence.png
	plot = ps.plot_sequence(sequence)

This can be advanced slightly by providing an n-gram to highlight to the :func:`.plot_sequence` method.

.. ipython:: python

	@savefig plot_sequence_highlighted.png
	plot = ps.plot_sequence(sequence, [1,2])


Understanding Subsequences
----------------------------
We've already seen how to visualise subsequences within a single sequence, but pysan is powerful enough to do this across a collection of sequences.
Using the :func:`.get_common_ngrams` method we can get the most common n-grams across a sample of sequences, plot each of them, and highlight the n-gram in just a few lines.

.. ipython:: python

	s1 = [1,2,3,4,3,3,2,2,3,2,3,2,3,1,3]
    s2 = [2,3,3,2,1,2,2,2,3,4,4,1,2,1,3]
    s3 = [1,3,3,2,2,2,2,3,3,3,2,3,3,4,4]
    sequences = [s1,s2,s3]
	@savefig plot_common_ngrams.png
    ps.plot_common_ngrams(sequences, 3)

Make a Transition Matrix
---------------------------







API Reference
=================

Method Signatures
-------------------

Core Module
++++++++++++++

`pysan <https://github.com/pysan-dev/pysan>`_'s core module contains a collection of useful methods for exploring single sequence data. These methods form the backbone of the library and are used in all other modules.


.. automodsumm:: pysan.core
	:functions-only:


MultiSequence Module
++++++++++++++++++++++++
`pysan <https://github.com/pysan-dev/pysan>`_'s multisequence module contains methods for exploring many sequences at the same time. This is useful for understanding how sequences vary within a population, and to look for common patterns or outliers.

.. automodsumm:: pysan.multisequence
	:functions-only:

Method Documentation
-----------------------


.. automodule:: pysan.core
	:members:
	:undoc-members:

.. automodule:: pysan.multisequence
	:members:
	:undoc-members:



Useful Links
------------------

`Sequence Analysis Association <https://sequenceanalysis.org/>`_ (SAA)

`SAA publication library <https://www.zotero.org/groups/2268769/saa_bibliography/library>`_

`SAA software library <https://sequenceanalysis.org/softwares/>`_.

`Social Sequence Analysis by Benjamin Cornwell <https://www.cambridge.org/core/books/social-sequence-analysis/3AC786DA3C99EB8795C7271BB350CB88>`_

`TraMineR sequence analysis package <http://traminer.unige.ch/>`_
