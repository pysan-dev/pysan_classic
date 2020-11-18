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


.. container:: core_top

	.. csv-table:: Subsequences

		:meth:`get_subsequences() <pysan.core.get_subsequences>`
		:meth:`get_ndistinct_subsequences() <pysan.core.get_ndistinct_subsequences>`

	.. csv-table:: N-grams

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

.. container:: plotting

	.. csv-table:: Visualisation

		:meth:`plot_sequence() <pysan.core.plot_sequence>` :meth:`plot_sequence_1d() <pysan.core.plot_sequence_1d>` :meth:`plot_element_counts() <pysan.core.plot_element_counts>` :meth:`plot_ngram_counts() <pysan.core.plot_ngram_counts>` :meth:`plot_transition_matrix() <pysan.core.plot_transition_matrix>`

MultiSequence Module
++++++++++++++++++++++++
`pysan <https://github.com/pysan-dev/pysan>`_'s multisequence module contains methods for exploring many sequences at the same time. This is useful for understanding how sequences vary within a population, and to look for common patterns or outliers. Many of the methods are wrappers of those in the `Core` module above for computing values across collections, see each description for details.


.. container:: core-top

	.. csv-table:: Attributes

		:meth:`are_recurrent() <pysan.multisequence.are_recurrent>`
		:meth:`get_summary_statistic() <pysan.multisequence.get_summary_statistics>`
		:meth:`get_synchrony() <pysan.multisequence.get_synchrony>`
		:meth:`get_sequence_frequencies() <pysan.multisequence.get_sequence_frequencies>`

	.. csv-table:: Derivative Sequences

		:meth:`get_motif() <pysan.multisequence.get_motif>`
		:meth:`get_modal_state() <pysan.multisequence.get_modal_state>`



.. container:: core-top

	.. csv-table:: Edit Distances

		:meth:`get_optimal_distance() <pysan.multisequence.get_optimal_distance>`
		:meth:`get_hamming_distance() <pysan.multisequence.get_hamming_distance>`
		:meth:`get_levenshtein_distance() <pysan.multisequence.get_levenshtein_distance>`


	.. csv-table:: Non-alignment Distances

		:meth:`get_dt_coefficient() <pysan.multisequence.none>`
		:meth:`get_combinatorial_distance() <pysan.multisequence.get_combinatorial_distance>`
		:meth:`get_geometric_distance() <pysan.multisequence.none>`

.. container:: core-top

	.. csv-table:: Spell-Adjusted Distances

		:meth:`get_dom_distance() <pysan.multisequence.none>`
		:meth:`get_lom_distance() <pysan.multisequence.none>`
		:meth:`get_twe_distance() <pysan.multisequence.none>`


	.. csv-table:: Whole Sequence Comparison

		:meth:`get_dissimilarity_matrix() <pysan.multisequence.get_dissimilarity_matrix>`
		:meth:`get_heirarchical_clustering() <pysan.multisequence.get_heirarchical_clustering>`
		:meth:`get_ch_index() <pysan.multisequence.get_ch_index>`


.. container:: plotting

	.. csv-table:: Visualisation

		:meth:`plot_common_ngrams() <pysan.multisequence.plot_common_ngrams>` :meth:`plot_sequences() <pysan.multisequence.plot_sequences>` :meth:`plot_state_distribution() <pysan.multisequence.plot_state_distribution>` :meth:`plot_sequence_frequencies() <pysan.multisequence.plot_sequence_frequencies>` :meth:`plot_transition_frequencies() <pysan.multisequence.plot_transition_frequencies>` :meth:`plot_mean_occurance() <pysan.multisequence.plot_mean_occurance>` :meth:`plot_modal_state() <pysan.multisequence.plot_modal_state>` :meth:`plot_dendrogram() <pysan.multisequence.plot_dendrogram>`



Documentation
++++++++++++++++++++++++


.. automodule:: pysan.core
	:members:
	:undoc-members:
	:member-order: bysource

.. automodule:: pysan.multisequence
	:members:
	:undoc-members:
	:member-order: bysource