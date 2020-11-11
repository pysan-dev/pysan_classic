API Reference
=================
This page contains each method in the pysan library, including examples of how they can be used. Visit the :ref:`User Guide` for more examples.


Method Signatures
-------------------

Core Module
++++++++++++++

`pysan <https://github.com/pysan-dev/pysan>`_'s core module contains a collection of useful methods for exploring single sequence data. These methods form the backbone of the library and are used in all other modules.


.. automodsumm:: pysan.core
	:functions-only:


MultiSequence Module
++++++++++++++++++++++++
`pysan <https://github.com/pysan-dev/pysan>`_'s multisequence module contains methods for exploring many sequences at the same time. This is useful for understanding how sequences vary within a population, and to look for common patterns or outliers. Many of the methods are wrappers of those in the `Core` module above for computing values across collections, see each description for details.

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