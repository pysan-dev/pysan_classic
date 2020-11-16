API Reference
=================
This page contains each method in the pysan library, including examples of how they can be used. Visit the :ref:`User Guide` for more examples, or the :ref:`Gallery` for some visualisations.



Core Module
++++++++++++++

`pysan <https://github.com/pysan-dev/pysan>`_'s core module contains a collection of useful methods for exploring single sequences. These methods form the backbone of the library and are used in all other modules.


.. container:: core_top

	.. csv-table:: Summary Statistics

		:meth:`is_recurrent() <pysan.core.is_recurrent>`
		:meth:`get_entropy() <pysan.core.get_entropy>`
		:meth:`get_turbulence() <pysan.core.get_turbulence>`
		:meth:`get_complexity() <pysan.core.get_complexity>`
		:meth:`get_routine() <pysan.core.get_routine>`
		:meth:`get_homogeneity() <pysan.core.get_homogeneity>`


	.. csv-table:: Element Descriptions

		:meth:`get_alphabet() <pysan.core.get_alphabet>`
		:meth:`get_first_positions() <pysan.core.get_first_positions>`
		:meth:`get_element_counts() <pysan.core.get_element_counts>`
		:meth:`get_element_frequency() <pysan.core.get_element_frequency>`


.. container:: core_subsequence

	.. csv-table:: Subsequence/Ngrams

		:meth:`get_subsequences() <pysan.core.get_subsequences>`
		:meth:`get_ndistinct_subsequences() <pysan.core.get_ndistinct_subsequences>`
		:meth:`get_all_ngrams() <pysan.core.get_all_ngrams>`
		:meth:`get_unique_ngrams() <pysan.core.get_unique_ngrams>`
		:meth:`get_ngram_counts() <pysan.core.get_ngram_counts>`
		:meth:`get_ngram_universe() <pysan.core.get_ngram_universe>`

.. container:: core_top

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


.. container:: multi-top

	.. csv-table:: Attributes

		:meth:`get_synchrony() <pysan.multisequence.get_synchrony>`
		:meth:`get_motif() <pysan.multisequence.get_motif>`
		:meth:`get_modal_state() <pysan.multisequence.get_modal_state>`



.. container:: multi-top

	.. csv-table:: Edit Distances

		:meth:`get_optimal_distance() <pysan.multisequence.get_optimal_distance>`
		:meth:`get_hamming_distance() <pysan.multisequence.get_hamming_distance>`
		:meth:`get_levenshtein_distance() <pysan.multisequence.get_levenshtein_distance>`

	.. csv-table:: Non-alignment Techniques

		:meth:`get_dt_coefficient() <pysan.multisequence.none>`
		:meth:`get_combinatorial_distance() <pysan.multisequence.get_combinatorial_distance>`
		:meth:`get_geometric_distance() <pysan.multisequence.none>`

	.. csv-table:: Whole Sequence Comparison

		:meth:`get_dt_coefficient() <pysan.multisequence.none>`
		:meth:`get_combinatorial_distance() <pysan.multisequence.get_combinatorial_distance>`
		:meth:`get_geometric_distance() <pysan.multisequence.none>`


Example
++++++++++++++++++++++++


.. automodule:: pysan.core
	:members:
	:undoc-members:
	:member-order: bysource

.. automodule:: pysan.multisequence
	:members:
	:undoc-members: