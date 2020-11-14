import matplotlib.pyplot as plt
import pysan.core as pysan_core
import itertools
import numpy as np

def generate_sequences(count, length, alphabet):
	"""
	Generates a number of sequences of a given length, with elements uniformly distributed using a given alphabet.
	This is useful for speed testing and other developer use-cases.
	
	Example
	--------
	>>> ps.generate_sequences(5, 10, [1,2,3]) #doctest: +SKIP
	[[2, 3, 2, 2, 3, 1, 2, 2, 1, 2],
	 [3, 1, 3, 3, 1, 3, 1, 3, 3, 1],
	 [1, 1, 2, 3, 3, 1, 3, 1, 3, 3],
	 [1, 3, 1, 2, 3, 2, 3, 1, 3, 2],
	 [1, 3, 2, 2, 2, 2, 3, 3, 1, 3]]
	"""
	
	sequences = []
	for x in range(count):
		sequences.append(pysan_core.generate_sequence(length, alphabet))
	
	return sequences

def get_common_ngrams(sequences, ngram_length):
	"""
	Extracts n-grams which appear one or more times in a collection of sequences, returning the number of occurances in a dictionary.

	Example
	---------
	>>> s1 = [1,1,1,1,1,2,2,2,2,3,3,3,4,4,4]
	>>> s2 = [1,1,1,2,2,2,2,2,3,3,3,3,4,4,4]
	>>> s3 = [1,1,2,2,2,2,2,3,3,3,2,3,3,4,4]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_common_ngrams(sequences, 3) #doctest: +NORMALIZE_WHITESPACE
	{'[1, 1, 2]': 3,
	'[1, 2, 2]': 3,
	'[2, 2, 2]': 8,
	'[2, 2, 3]': 3,
	'[2, 3, 3]': 4,
	'[3, 3, 3]': 4,
	'[3, 3, 4]': 3,
	'[3, 4, 4]': 3}
	"""

	found_ngrams = 'none'
	for sequence in sequences:
		ngrams = pysan_core.get_ngram_counts(sequence, ngram_length)
		if found_ngrams == 'none':
			found_ngrams = ngrams
		else:
			keys_to_remove = []
			for key, value in found_ngrams.items():
				if key in ngrams.keys():
					found_ngrams[key] = value + ngrams[key]
				else:
					keys_to_remove.append(key)
			for key in keys_to_remove:
				del found_ngrams[key]
	return found_ngrams

def get_all_spells(sequences):
	"""
	UC Computes spells across a collection of sequences, returning a list of tuples where each tuple holds the element, the length of the spell, and the number of occurances in the collection.
	"""
	
	all_spells = []
	for sequence in sequences:
		spells = ps.get_spells(sequence)
		
	pass

def get_longest_spells(sequences):
	"""
	UC Extracts the longest spell for each sequence in a collection, returning the element, count, and starting position for each of the spells.

	"""

	pass

def are_recurrent(sequences):
	"""
	Returns false if all sequences in a given collection are recurrant, true otherwise.
	
	Example
	---------
	>>> s1 = [1,2,3,4]
	>>> s2 = [3,2,4,5]
	>>> s3 = [2,3,4,1]
	>>> sequences = [s1,s2,s3]
	>>> ps.are_recurrent(sequences)
	False

	"""

	for sequence in sequences:
		if pysan_core.is_recurrent(sequence):
			return True
	
	return False

def get_first_position_reports(sequences):
	"""
	UC Reports the positions of each first occurance of each element across a collection of sequences.
	"""

	pass

def get_all_ntransitions(sequences):
	"""
	UC Returns a list containing the number of transactions in each sequence in a collection.
	"""

	pass


def get_statistics(sequence, function):
	"""
	UC Computes a summary statistic (e.g. entropy, complexity, or turbulence) for each sequence in a collection.

	"""

	pass



def get_routine_scores(sequences):
	"""
	UC Returns a list containing the routine scores for each sequence in a collection using :meth:`get_routine() <pysan.core.get_routine>`.
	"""

	pass

def get_all_unique_ngrams(sequences ,n):
	"""
	UC Creates a list of all unique ngrams in a collection of sequences.
	"""

	pass

def get_every_ngram(sequences, n):
	"""
	UC Creates a list of all ngrams across all sequences in a collection.
	"""

	pass

def get_all_ngram_counts(sequences, n):
	"""
	UC Computes the prevalence of ngrams in a collection of sequences.
	"""

	pass

def get_all_element_counts(sequences):
	"""
	UC Counts the number of occurances of each element across a collection of sequences.
	"""

	pass

def get_all_element_frequencies(sequences):
	"""
	UC Computes the frequencies of each element across a collection of sequences.
	"""

	pass

def get_modal_state(sequences):
	"""
	Computes the modal states for each position in a collection of sequences, returning a sequence of tuples containing the modal element and its number of occurances at that position.

	Example
	--------
	>>> s1 = [1,1,1,2,2,3,3]
	>>> s2 = [1,2,2,2,2,3,3]
	>>> s3 = [1,1,1,1,2,2,3]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_modal_state(sequences)
	[(1, 3), (1, 2), (1, 2), (2, 2), (2, 3), (3, 2), (3, 3)]


	"""

	longest_sequence = max([len(s) for s in sequences])
	
	modal_elements = []
	for position in range(longest_sequence):
		
		elements_at_this_position = []
		for sequence in sequences:
			try:
				elements_at_this_position.append(sequence[position])
			except:
				continue
		
		# this line leaves multi-modal position behaviour undefined
		modal_element = max(set(elements_at_this_position), key=elements_at_this_position.count)
		
		modal_elements.append((modal_element, elements_at_this_position.count(modal_element)))
	
	return modal_elements

def get_sequence_frequencies(sequences):
	"""
	Computes the frequencies of different sequences in a collection, returning a dictionary of their string representations and counts.
	
	Example
	--------
	
	.. plot::
	
		>>> s1 = [1,1,2,2,3]
		>>> s2 = [1,2,2,3,3]
		>>> s3 = [1,1,2,2,2]
		>>> sequences = [s1,s2,s2,s3,s3,s3]
		>>> ps.get_sequence_frequencies(sequences) #doctest: +NORMALIZE_WHITESPACE
		{'[1, 1, 2, 2, 2]': 3, 
		 '[1, 2, 2, 3, 3]': 2, 
		 '[1, 1, 2, 2, 3]': 1}
	"""
	
	# converting to strings makes comparison easy
	sequences_as_strings = [str(s) for s in sequences]
	
	sequence_frequencies = {}
	for sequence in set(sequences_as_strings):
		sequence_frequencies[sequence] = sequences_as_strings.count(sequence)
	
	sequence_frequencies = {k: v for k, v in sorted(sequence_frequencies.items(), key=lambda item: item[1], reverse=True)}
	
	return sequence_frequencies

def get_global_alphabet(sequences):
	"""
	Computes the alphabet across all sequences in a collection.

	Example
	---------
	>>> s1 = [1,1,1,2,2,2]
	>>> s2 = [1,1,2,2,3,3]
	>>> sequences = [s1,s2]
	>>> ps.get_global_alphabet(sequences)
	[1, 2, 3]

	"""
	
	alphabets = [pysan_core.get_alphabet(s) for s in sequences]
	
	global_alphabet = sorted(list(set([item for sublist in alphabets for item in sublist])))
	
	return global_alphabet

def get_transition_frequencies(sequences):
	"""
	Computes the number of transitions for all sequences in a collection.

	Example
	--------
	.. plot::
	
		>>> s1 = [1,1,1,2,2,3,3,3]
		>>> s2 = [1,1,2,2,3,2,4,4]
		>>> s3 = [1,1,1,2,2,3,3,3]
		>>> s4 = [1,1,1,1,2,3,2,3]
		>>> sequences = [s1,s2,s3,s4]
		>>> ps.get_transition_frequencies(sequences) #doctest: +NORMALIZE_WHITESPACE
		{'[2, 3]': 5, 
		 '[1, 2]': 4, 
		 '[3, 2]': 2, 
		 '[2, 4]': 1}


	"""
	
	all_transitions = []
	for sequence in sequences:
		all_transitions += pysan_core.get_transitions(sequence)

		
	all_transitions_as_strings = [str(t) for t in all_transitions]
	transition_frequencies = {}
	
	for transition in set(all_transitions_as_strings):
		transition_frequencies[str(transition)] = all_transitions_as_strings.count(transition)
	
	transition_frequencies = {k: v for k, v in sorted(transition_frequencies.items(), key=lambda item: item[1], reverse=True)}
	
	return transition_frequencies

def get_all_transitions_matrix(sequences):
	"""
	UC Computes a transition matrix across all transitions in every sequence in a collection.
	"""

	pass

def get_motif(sequences):
	"""
	Computes the motif for a given collection of sequences.
	A motif is a representative sequence for all sequences in a collection, with blank values (0) being those which are variable within the collection, and fixed values which are not.
	Motifs are related to the measure of synchrony in that synchrony is equal to the number of non-blank elements in the motif.
	
	Example
	--------
	
	>>> s1 = [1,1,2,2,3]
	>>> s2 = [1,2,2,3,3]
	>>> s3 = [1,1,2,2,2]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_motif(sequences)
	[1, 0, 2, 0, 0]

	"""
	
	shortest_sequence = min([len(s) for s in sequences])
	
	same_elements = []
	for position in range(shortest_sequence):
	
		elements_at_this_position = []
		for sequence in sequences:
			elements_at_this_position.append(sequence[position])
	
		if elements_at_this_position.count(elements_at_this_position[0]) == len(elements_at_this_position):
			same_elements.append(sequences[0][position])
		else:
			same_elements.append(0)
	
	return same_elements

def get_synchrony(sequences):
	"""
	Computes the normalised synchrony between a two or more sequences. 
	Synchrony here refers to positions with identical elements, e.g. two identical sequences have a synchrony of 1, two completely different sequences have a synchrony of 0.
	The value is normalised by dividing by the number of positions compared.
	This computation is defined in Cornwell's 2015 book on social sequence analysis, page 230.
	
	Example
	--------
	>>> s1 = [1,1,2,2,3]
	>>> s2 = [1,2,2,3,3]
	>>> sequences = [s1,s2]
	>>> ps.get_synchrony(sequences)
	0.6
	
	"""
	
	shortest_sequence = min([len(s) for s in sequences])
	
	same_elements = []
	for position in range(shortest_sequence):
	
		elements_at_this_position = []
		for sequence in sequences:
			elements_at_this_position.append(sequence[position])
			
		same_elements.append(elements_at_this_position.count(elements_at_this_position[0]) == len(elements_at_this_position))
		
	return same_elements.count(True) / shortest_sequence

def get_dissimilarity(sequences, function):
	"""
	Computes a dissimilarity matrix using a given function.
	This function is applied pairwise between all sequences in the collection provided.
	This function can be used abstractly for all kinds of pairwise measures.
	
	Example
	---------
	>>> s1 = [1,1,2,2,3]
	>>> s2 = [1,2,2,3,3]
	>>> s3 = [1,1,2,2,2]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_dissimilarity(sequences, ps.get_synchrony) #doctest: +NORMALIZE_WHITESPACE
	array([[1. , 0.6, 0.8],
       	   [0.6, 1. , 0.4],
           [0.8, 0.4, 1. ]])
	
	"""
	
	num_sequences = len(sequences)
	scores = np.zeros((num_sequences,num_sequences),dtype = float)
	
	
	for row in range(num_sequences):
		for column in range(num_sequences):
			score = function([sequences[row], sequences[column]])
			scores[row,column] = float(score)
	
	return scores

# ============= MULTISEQUENCE PLOTTING ===============

def plot_common_ngrams(sequences, ngram_length):
	"""
	Plot the number of occurances (per sequence) of ngrams common to a collection of sequences.

	.. plot::
	
		>>> s1 = [1,2,3,4,3,3,2,2,3,2,3,2,3,1,3]
		>>> s2 = [2,3,3,2,1,2,2,2,3,4,4,1,2,1,3]
		>>> s3 = [1,3,3,2,2,2,2,3,3,3,2,3,3,4,4]
		>>> sequences = [s1,s2,s3]
		>>> ps.plot_common_ngrams(sequences, 3) #doctest: +SKIP

	"""
	found_ngrams = get_common_ngrams(sequences, ngram_length)
	
	ngrams = [eval(key) for key in found_ngrams.keys()]

	most_common_ngram = eval(max(found_ngrams, key=lambda key: found_ngrams[key]))

	for sequence in sequences:
		pysan_core.plot_sequence(sequence, most_common_ngram)

	return plt

def plot_sequences(sequences, gap=True):
	"""
	Creates a scatter style sequence plot for a collection of sequences.

	Example
	----------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,2,4,4,3,2,1,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,1,2,3,2,2,2,3,3,2,4,4]
		>>> s3 = [1,1,1,2,2,3,2,4,4,3,2,1,2,3,3,3,4,4,4,3,3]
		>>> s4 = [1,1,1,1,2,3,2,3,3,3,3,1,2,2,3,3,3,4,4,4,4]
		>>> sequences = [s1,s2,s3,s4]
		>>> ps.plot_sequences(sequences) #doctest: +SKIP
		>>> ps.plot_sequences(sequences, gap=False) #doctest: +SKIP

	"""
	max_sequence_length = max([len(s) for s in sequences])
	plt.figure(figsize=[max_sequence_length*0.3,0.3 * len(sequences)])

	for y,sequence in enumerate(sequences):
		np_sequence = np.array(sequence)
		alphabet_len = len(pysan_core.get_alphabet(sequence))

		plt.gca().set_prop_cycle(None)
		unique_values = pysan_core.get_alphabet(sequence)
		for i, value in enumerate(unique_values):
			
			if gap:
				points = np.where(np_sequence == value, y + 1, np.nan)
				plt.scatter(x=range(len(np_sequence)), y=points, marker='s', label=value, s=100)
			else:
				points = np.where(np_sequence == value, 1, np.nan)
				plt.bar(range(len(points)), points, bottom=[y for x in range(len(points))], width=1, align='edge', label=i)

	if gap:
		plt.ylim(0.4, len(sequences) + 0.6)
		plt.xlim(-0.6, max_sequence_length - 0.4)
	else:
		plt.ylim(0,len(sequences))
		plt.xlim(0,max_sequence_length)
		
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.1), loc='upper left')
	plt.tick_params(
		axis='y',
		which='both',
		left=False,
		labelleft=False)

	return plt

def plot_state_distribution(sequences):
	"""
	Creates a state distribution plot based on a collection of sequences.
	
	Example
	--------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,3,4,4,3,2,2,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,2,2,3,2,2,2,3,3,3,4,4]
		>>> s3 = [1,1,1,2,2,3,3,3,4,3,2,2,2,3,3,3,4,4,4,3,3]
		>>> s4 = [1,1,1,1,2,3,2,3,3,3,3,2,2,2,3,3,3,4,4,4,4]
		>>> sequences = [s1,s2,s3,s4]
		>>> ps.plot_state_distribution(sequences) #doctest: +SKIP
	
	"""
	

	longest_sequence = max([len(s) for s in sequences])

	alphabets = [list(pysan_core.get_alphabet(s)) for s in sequences]

	global_alphabet = list(set(list(itertools.chain.from_iterable(alphabets))))

	sorted_global_alphabet = sorted(global_alphabet)

	plt.figure()

	previous_bar_tops = [0 for x in range(longest_sequence)]
	for element in sorted_global_alphabet:

		element_position_counts = []
		for position in range(longest_sequence):

			elements_this_position = 0
			for sequence in sequences:

				try: # this try is for sequences of non-identical lengths
					if sequence[position] == element:
						elements_this_position += 1 / len(sequences)
				except:
					continue

			element_position_counts.append(elements_this_position)

		plt.bar(range(longest_sequence), element_position_counts, bottom=previous_bar_tops, label=element, width=1, align='edge')
		previous_bar_tops = [a + b for a, b in zip(previous_bar_tops, element_position_counts)]

	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.xlim(0, longest_sequence)

	plt.ylabel('Frequency (n=' + str(len(sequence)) + ')')
	plt.xlabel('Position')

	return plt

def plot_sequence_frequencies(sequences):
	"""
	Plots sequences using :meth:`plot_sequences`, ordering sequences with the most common at the bottom, and the rarest at the top. This is most useful when comparing short sequences.
	
	Example
	---------
	.. plot::
	
		>>> s1 = [1,1,1,2,2,3,3,2,2,3,2,2,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,2,2,3,2,2,2,3,3,3,4,4]
		>>> s3 = [1,1,1,2,2,3,3,3,4,3,2,2,2,3,3,3,4,4,4,3,3]
		>>> sequences = [s1,s2,s2,s3,s3,s3]
		>>> ps.plot_sequence_frequencies(sequences) #doctest: +SKIP

	"""
	frequencies = get_sequence_frequencies(sequences)
	
	raw_sequences_ordered = []
	for sequence, count in frequencies.items():
		for x in range(count):
			raw_sequences_ordered.append(eval(sequence))
			
	plt = plot_sequences(raw_sequences_ordered, gap=False)
	
	plt.tick_params(
		axis='y',
		which='both',
		left=True,
		labelleft=True)
	
	plt.yticks([0, len(sequences) * 0.25, len(sequences) * 0.5, len(sequences) * 0.75, len(sequences)], [0,25,50,75,100])
	plt.ylabel('Frequency (%)')
	plt.xlabel('Position')
	
	return plt

def plot_transition_frequencies(sequences):
	"""
	Creates a transition frequency plot for each transition in a collection of sequences.

	Example
	--------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,3,4,4,3,2,2,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,2,2,3,2,2,2,3,3,3,4,4]
		>>> s3 = [1,1,1,2,2,3,3,3,4,3,2,2,2,3,3,3,4,4,4,3,3]
		>>> s4 = [1,1,1,1,2,3,2,3,3,3,3,2,2,2,3,3,3,4,4,4,4]
		>>> sequences = [s1,s2,s3,s4]
		>>> ps.plot_transition_frequencies(sequences) #doctest: +SKIP
	"""    

	transition_frequencies = get_transition_frequencies(sequences)

	transitions = [key.replace(', ', '>') for key, v in transition_frequencies.items()]
	counts = [value for k, value in transition_frequencies.items()]

	plt.bar(transitions, counts)
	plt.xlim(-0.6, len(transitions) - 0.4)
	plt.ylabel('Number of Transitions')
	plt.xlabel('State Transitions')

	return plt

def plot_mean_occurance(sequences):
	"""
	Plots the mean number of occurances of each element across a collection of sequences.
	
	Example
	--------
	.. plot::
	
		>>> s1 = [1,1,1,1,1,2,2,2,2,3,3,3,4,4,4]
		>>> s2 = [1,1,1,2,2,2,2,2,3,3,3,3,4,4,4]
		>>> s3 = [1,1,2,2,2,2,2,3,3,3,2,3,3,4,4]
		>>> sequences = [s1,s2,s3]
		>>> ps.plot_mean_occurance(sequences) #doctest: +SKIP
	
	"""
	
	longest_sequence = max([len(s) for s in sequences])

	alphabets = [list(pysan_core.get_alphabet(s)) for s in sequences]

	global_alphabet = list(set(list(itertools.chain.from_iterable(alphabets))))

	sorted_global_alphabet = sorted(global_alphabet)
	
	
	for element in sorted_global_alphabet:
		occurances = 0
		for sequence in sequences:
			occurances += sequence.count(element)
		
		plt.bar(element, occurances / len(sequences))
	
	plt.xticks(range(1, len((sorted_global_alphabet)) + 1), sorted_global_alphabet)
	plt.xlabel('Element')
	plt.ylabel('Mean Occurance per Sequence')
	
	return plt

def plot_modal_state(sequences):
	"""
	Plots the modal state for each position in a collection of sequences.


	Example
	--------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,3]
		>>> s2 = [1,2,2,2,2,3,3]
		>>> s3 = [1,1,1,1,2,2,3]
		>>> sequences = [s1,s2,s3]
		>>> ps.plot_modal_state(sequences) #doctest: +SKIP

	"""
	modal_elements = get_modal_state(sequences)
	
	longest_sequence = max([len(s) for s in sequences])
	
	plt.figure()
	   
	global_alphabet = get_global_alphabet(sequences)
	
	
	for element in global_alphabet:
		modal_element_counts = []
		for position in range(longest_sequence):
			if modal_elements[position][0] == element:
				modal_element_counts.append(modal_elements[position][1] / len(sequences))
			else:
				modal_element_counts.append(0)
		plt.bar(range(longest_sequence), modal_element_counts, label=element, width=1, align='edge')
		
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.xlim(0, longest_sequence)
	plt.ylim(0, 1)
	plt.ylabel('State Frequency, n=' + str(len(sequences)))
	plt.xlabel('Position')
	
	return plt

def plot_entropy(sequences):
	"""
	Plots the entropy at each position across a collection of sequences.

	Example
	----------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,2,4,4,3,2,1,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,1,2,3,2,2,2,3,3,2,4,4]
		>>> s3 = [2,2,1,1,2,3,2,4,4,3,2,1,2,3,3,3,4,4,4,3,4]
		>>> s4 = [1,1,1,1,2,3,2,3,3,3,3,1,2,2,3,3,3,4,4,4,3]
		>>> sequences = [s1,s2,s3,s4]
		>>> ps.plot_entropy(sequences) #doctest: +SKIP

	"""

	longest_sequence = max([len(s) for s in sequences])

	entropies = []
	for position in range(longest_sequence):

		this_position_crosssection = [sequence[position] for sequence in sequences]

		entropy = pysan_core.get_entropy(this_position_crosssection)

		entropies.append(entropy)

	plt.ylim(0,1)
	plt.plot(range(len(entropies)), entropies)
	plt.xlabel('Position, p')
	plt.ylabel('Normalised Entropy, e')

	return plt