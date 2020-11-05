import matplotlib.pyplot as plt
import pysan.core as pysan_core
import itertools

def get_common_ngrams(sequences, ngram_length):
	"""
	Extracts n-grams which appear one or more times in a collection of sequences, returning the number of occurances in a dictionary.

	Example
	---------
	>>> s1 = [1,1,1,1,1,2,2,2,2,3,3,3,4,4,4]
	>>> s2 = [1,1,1,2,2,2,2,2,3,3,3,3,4,4,4]
	>>> s3 = [1,1,2,2,2,2,2,3,3,3,2,3,3,4,4]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_common_ngrams(sequences, 3)
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

def get_modal_state(sequences):
	"""
	Computes the modal states for each position in a collection of sequences, returning a sequence.

	Example
	--------
	>>> s1 = [1,1,1,2,2,3,3]
	>>> s2 = [1,2,2,2,2,3,3]
	>>> s3 = [1,1,1,1,2,2,3]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_modal_state(sequences)
	[1,1,1,2,2,3,3]

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

def get_state_frequency(sequences):
	"""
	Computes the sequence frequency for each position in a collection of sequences.
	"""

	pass

def get_global_alphabet(sequences):
	"""
	Computes the alphabet across all sequences in a collection.

	Example
	---------
	>>> s1 = [1,1,1,2,2,2]
	>>> s2 = [1,1,2,2,3,3]
	>>> sequences = [s1,s2]
	>>> ps.get_global_alphabet(sequences)
	[1,2,3]

	"""
	
	alphabets = [pysan_core.get_alphabet(s) for s in sequences]
	
	global_alphabet = sorted(list(set([item for sublist in alphabets for item in sublist])))
	
	return global_alphabet

def get_transition_frequencies(sequences):
	"""
	Computes the frequency of transitions between states across a collection of sequences.

	"""

	pass


# ============= MULTISEQUENCE PLOTTING ===============


def plot_common_ngrams(sequences, ngram_length):
	"""
	Plot the number of occurances (per sequence) of ngrams common to a collection of sequences.

	.. plot::
	
		>>> s1 = [1,2,3,4,3,3,2,2,3,2,3,2,3,1,3]
		>>> s2 = [2,3,3,2,1,2,2,2,3,4,4,1,2,1,3]
		>>> s3 = [1,3,3,2,2,2,2,3,3,3,2,3,3,4,4]
		>>> sequences = [s1,s2,s3]
		>>> ps.plot_common_ngrams(sequences, 3)

	"""
	found_ngrams = get_common_ngrams(sequences, ngram_length)
	
	ngrams = [eval(key) for key in found_ngrams.keys()]

	most_common_ngram = eval(max(found_ngrams, key=lambda key: found_ngrams[key]))

	for sequence in sequences:
		pysan_core.plot_sequence(sequence, most_common_ngram)

	return plt
		
def plot_state_distribution(sequences):
	"""
	Creates a state distribution plot based on a collection of sequences.
	
	Example
	--------
	.. plot::

		>>> s1 = [1,1,1,1,1,2,2,2,2,3,3,3]
		>>> s2 = [1,1,1,2,2,2,2,2,3,3,3,3]
		>>> s3 = [1,1,2,2,2,2,3,3,3,3,2,3]
		>>> sequences = [s1,s2,s3]
		>>> ps.plot_state_distribution(sequences)
	
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
						elements_this_position += 1
				except:
					continue

			element_position_counts.append(elements_this_position)

		plt.bar(range(longest_sequence), element_position_counts, bottom=previous_bar_tops, label=element, width=1)
		previous_bar_tops = [a + b for a, b in zip(previous_bar_tops, element_position_counts)]

	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.xlim(-0.5, longest_sequence - 0.5)
	plt.tick_params(
		axis='y',
		which='both',
		left=False,
		labelleft=False)

	plt.xlabel('Position')

	return plt

def plot_sequence_frequency(sequences):
	"""
	UC Creates a sequence frequency plot for a collection of sequences.

	"""

	pass

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
		>>> ps.plot_mean_occurance(sequences)
	
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
		>>> ps.plot_modal_state(sequences)

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
		plt.bar(range(longest_sequence), modal_element_counts, label=element, width=1)
		
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.xlim(-0.5, longest_sequence - 0.5)
	plt.ylabel('State Frequency, n=' + str(len(sequences)))
	plt.xlabel('Position')
	
	return plt


def plot_entropy(sequence):
	"""
	
	UC Plots the entropy of a collection of sequences at each position over time.
	


	"""
	pass