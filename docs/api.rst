API Reference
=================
This page contains each method in the pysan library, including examples of how they can be used. Visit the :ref:`User Guide` for more examples.



Core Module
++++++++++++++

`pysan <https://github.com/pysan-dev/pysan>`_'s core module contains a collection of useful methods for exploring single sequence data. These methods form the backbone of the library and are used in all other modules.

.. csv-table:: Summary Statistics

	:meth:`is_recurrent() <pysan.core.is_recurrent>`
	:meth:`get_entropy() <pysan.core.get_entropy>`
	:meth:`get_turbulence() <pysan.core.get_turbulence>`
	:meth:`get_complexity() <pysan.core.get_complexity>`
	:meth:`get_routine() <pysan.core.get_routine>`


.. csv-table:: Element Descriptions

	:meth:`get_alphabet() <pysan.core.get_alphabet>`
	:meth:`get_first_positions() <pysan.core.get_first_positions>`
	:meth:`get_element_counts() <pysan.core.get_element_counts>`
	:meth:`get_element_frequency() <pysan.core.get_element_frequency>`


.. csv-table:: Subsequence/Ngrams

	:meth:`get_ndistinct_subsequences() <pysan.core.get_ndistinct_subsequences>`
	:meth:`get_all_ngrams() <pysan.core.get_all_ngrams>`
	:meth:`get_unique_ngrams() <pysan.core.get_unique_ngrams>`
	:meth:`get_ngram_counts() <pysan.core.get_ngram_counts>`
	:meth:`get_ngram_universe() <pysan.core.get_ngram_universe>`

.. csv-table:: Transitions

	:meth:`get_transitions() <pysan.core.get_transitions>`
	:meth:`get_ntransitions() <pysan.core.get_ntransitions>`
	:meth:`get_transition_matrix() <pysan.core.get_transition_matrix>`

.. csv-table:: Spells

	:meth:`get_spells() <pysan.core.get_spells>`
	:meth:`get_longest_spell() <pysan.core.get_longest_spell>`
	:meth:`get_spell_durations() <pysan.core.get_spell_durations>`

MultiSequence Module
++++++++++++++++++++++++
`pysan <https://github.com/pysan-dev/pysan>`_'s multisequence module contains methods for exploring many sequences at the same time. This is useful for understanding how sequences vary within a population, and to look for common patterns or outliers. Many of the methods are wrappers of those in the `Core` module above for computing values across collections, see each description for details.

.. automodsumm:: pysan.multisequence
	:functions-only:

Method Documentation
++++++++++++++++++++++++


.. automodule:: pysan.core
	:members:
	:undoc-members:
	:member-order: bysource

.. automodule:: pysan.multisequence
	:members:
	:undoc-members: