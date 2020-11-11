
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
