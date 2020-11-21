import matplotlib.pyplot as plt
import pysan.core as pysan_core
import itertools, math
import numpy as np
import pandas as pd
from sklearn import cluster
import scipy



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



# ===== ELEMENTS =====

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

def get_first_position_reports(sequences):
	"""
	UC Reports the positions of each first occurance of each element across a collection of sequences.
	"""

	pass



# ===== NGRAM METHODS =====

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



# ===== TRANSITIONS =====

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

def get_all_ntransitions(sequences):
	"""
	UC Returns a list containing the number of transactions in each sequence in a collection.
	"""

	pass



# ===== SPELLS =====

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



# ===== COLLECTION ATTRIBUTES =====

def are_recurrent(sequences):
	"""
	Returns true if any of the sequences in a given collection are recurrant, false otherwise.
	
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

def get_summary_statistic(sequence, function):
	"""
	UC Computes a summary statistic (e.g. entropy, complexity, or turbulence) for each sequence in a collection, returning the results as a list.

	"""

	pass

def get_routine_scores(sequences, duration):
	"""
	UC Returns a list containing the routine scores for each sequence in a collection using :meth:`get_routine() <pysan.core.get_routine>`.
	"""

	pass

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

def get_sequence_frequencies(sequences):
	"""
	Computes the frequencies of different sequences in a collection, returning a dictionary of their string representations and counts.
	
	Example
	--------
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



# ===== DERIVATIVE SEQUENCES =====

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



# ===== EDIT DISTANCES =====

def get_optimal_distance(s1,s2, match = 0, mismatch = -1, gap = -1):
	"""
	Computes the optimal matching distance between two sequences using the `Needleman-Wunsch algorithm <https://www.sciencedirect.com/science/article/abs/pii/0022283670900574?via%3Dihub>`_ based on Devon Ryan's implementation found `here <https://www.biostars.org/p/231391/>`_.
	
	
	Example
	--------
	>>> s1 = [1,1,1,1,2,2,2,2]
	>>> s2 = [1,2,2,3,3,4,5,5]
	>>> ps.get_optimal_distance(s1,s2)
	7.0
	
	"""
	
	penalty = {'MATCH': match, 'MISMATCH': mismatch, 'GAP': gap} #A dictionary for all the penalty valuse.
	n = len(s1) + 1 #The dimension of the matrix columns.
	m = len(s2) + 1 #The dimension of the matrix rows.
	al_mat = np.zeros((m,n),dtype = float) #Initializes the alighment matrix with zeros.
	p_mat = np.zeros((m,n),dtype = str) #Initializes the pointer matrix with zeros.
	#Scans all the first rows element in the matrix and fill it with "gap penalty"
	for i in range(m):
		al_mat[i][0] = penalty['GAP'] * i
		p_mat[i][0] = 'V'
	#Scans all the first columns element in the matrix and fill it with "gap penalty"
	for j in range (n):
		al_mat[0][j] = penalty['GAP'] * j
		p_mat [0][j] = 'H'
	
	
	#-------------------------------------------------------
	#This function returns to values for cae of match or mismatch
	def Diagonal(n1,n2,pt):
		if(n1 == n2):
			return pt['MATCH']
		else:
			return pt['MISMATCH']
	
	#------------------------------------------------------------   
	#This function gets the optional elements of the aligment matrix and returns the elements for the pointers matrix.
	def Pointers(di,ho,ve):
		pointer = max(di,ho,ve) #based on python default maximum(return the first element).

		if(di == pointer):
			return 'D'
		elif(ho == pointer):
			return 'H'
		else:
			 return 'V'
	
	#Fill the matrix with the correct values.
	p_mat [0][0] = 0 #Return the first element of the pointer matrix back to 0.
	for i in range(1,m):
		for j in range(1,n):
			di = al_mat[i-1][j-1] + Diagonal(s1[j-1],s2[i-1],penalty) #The value for match/mismatch -  diagonal.
			ho = al_mat[i][j-1] + penalty['GAP'] #The value for gap - horizontal.(from the left cell)
			ve = al_mat[i-1][j] + penalty['GAP'] #The value for gap - vertical.(from the upper cell)
			al_mat[i][j] = max(di,ho,ve) #Fill the matrix with the maximal value.(based on the python default maximum)
			p_mat[i][j] = Pointers(di,ho,ve)
	
	#print(np.matrix(al_mat))
	#print(np.matrix(p_mat))
	
	# optimal alignment score = bottom right value in al_mat
	score = al_mat[m-1][n-1]
	#print(score)
	if score == 0: # fixes -0 bug for completeness
		return 0

	return -score

def get_levenshtein_distance(s1,s2):
	"""
	Computes the `Levenshtein II distance <https://journals.sagepub.com/doi/abs/10.1177/0049124110362526>`_ between two sequences, which is the optimal distance using only insertions and deletions.
	This is identical to the :meth:`get_optimal_distance` method with a mismatch cost of ~infinity (-9999999) and a gap cost of -1.
	See the :meth:`get_optimal_distance` method with its default parameters for the Levenshtein I distance.
	
	Example
	--------
	>>> s1 = [1,1,1,1,2,2,2,2]
	>>> s2 = [1,2,2,3,3,4,5,5]
	>>> ps.get_levenshtein_distance(s1,s2)
	10.0
	
	"""
	return get_optimal_distance(s1,s2, match=0, mismatch=-9999999, gap=-1)

def get_hamming_distance(s1,s2):
	"""
	Computes the Hamming distance  between two sequences, which is the optimal distance using only substitutions (no indels).
	This is identical to the :meth:`get_optimal_distance` method with a mismatch cost of -1 and a gap cost of ~infinity (-999999).
	Note that this can only be used on sequences of the same length given the infinite cost of gaps.
	
	Example
	--------
	>>> s1 = [1,1,1,1,2,2,2,2]
	>>> s2 = [1,2,2,3,3,4,5,5]
	>>> ps.get_hamming_distance(s1,s2)
	7.0
	
	"""
	if len(s1) != len(s2):
		raise Exception('sequences provided are not equal length - cannot compute Hamming distance')
	
	return get_optimal_distance(s1,s2, match=0, mismatch=-1, gap=-999999)

def get_combinatorial_distance(s1,s2):
	"""
	Computes the combinatorial distance between two sequences.
	This is defined as 1 minus the number of common subsequences divided by the square root of the product of the number of subsequences in each sequence.
	See page 149 in Social Sequence Analysis by Benjamin Cornwell for details.
	
	Example
	--------
	
	>>> s1 = [1,2,3]
	>>> s2 = [2,3,4]
	>>> ps.get_combinatorial_distance(s1,s2)
	0.5714285714285714
	
	"""
	
	s1_subs = pysan_core.get_subsequences(s1)
	s2_subs = pysan_core.get_subsequences(s2)
	
	# parse to strings so that they can be easily compared
	# - haven't tried it without, may be faster...
	s1_subs_strings = [str(s) for s in s1_subs]
	s2_subs_strings = [str(s) for s in s2_subs]
	
	common_subs = list(set(s1_subs_strings) & set(s2_subs_strings))
	
	bottom_fraction = math.sqrt(len(s1_subs) * len(s2_subs))
	full_fraction = len(common_subs) / bottom_fraction
	
	return 1 - full_fraction



# ===== WHOLE SEQUENCE COMPARISON =====

def get_dissimilarity_matrix(sequences, function):
	"""
	Computes a dissimilarity matrix using a given function.
	This function can be a measure of dissimilarity, distance, or any other measurement between two sequences.
	The column names and index on the matrix are the indexes of each sequences in the collection.
	
	Example
	----------
	>>> s1 = [1,1,1,2,2,3,3,3]
	>>> s2 = [1,1,3,2,2,3,1,3]
	>>> s3 = [1,1,2,2,3,2,3,2]
	>>> sequences = [s1,s2,s3]
	>>> ps.get_dissimilarity_matrix(sequences, ps.get_optimal_distance) #doctest: +NORMALIZE_WHITESPACE
	 	0 	1 	2
	0 	0.0 	2.0 	3.0
	1 	2.0 	0.0 	3.0
	2 	3.0 	3.0 	0.0
	"""
	
	matrix = np.zeros((len(sequences), len(sequences)))
	for x, row in enumerate(sequences):
		for y, column, in enumerate(sequences):
			matrix[x][y] = function(row, column)
			
	df = pd.DataFrame(matrix, columns=range(len(sequences)), index=range(len(sequences)))
	
	return df

def get_heirarchical_clustering(sequences, function):
	"""
	Fits an `sklearn agglomerative clustering model <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ using the 'average' linkage criteria.
	The source code for this method is only two lines, so please copy to your own script to modify specific clustering parameters!
	
	Example
	---------
	
	
	AgglomerativeClustering(affinity='precomputed', distance_threshold=0,
                        linkage='average', n_clusters=None)
	"""

	matrix = get_dissimilarity_matrix(sequences, function)

	model = cluster.AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=0, n_clusters=None).fit(matrix)

	return model

def get_ch_index(model):
	"""
	UC Computes the Calinski-Harabasz index
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
	plt.ylim(0,1)

	plt.ylabel('Frequency (n=' + str(len(sequences)) + ')')
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


def plot_dendrogram(model, **kwargs):
	"""
	Plots a heirarchical clustering model - example taken from the `sklearn library <https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py>`_
	
	Example
	----------
	.. plot::

		>>> s1 = [1,1,1,2,2,3,2,4,4,3,2,1,2,3,3,3,2,2,1,1,1]
		>>> s2 = [1,1,2,2,3,2,4,4,3,2,1,2,3,2,2,2,3,3,2,4,4]
		>>> s3 = [2,2,1,1,2,3,2,4,4,3,2,1,2,3,3,3,4,4,4,3,4]
		>>> s4 = [1,1,1,1,2,3,2,3,3,3,3,1,2,2,3,3,3,4,4,4,3]
		>>> sequences = [s1,s2,s3,s4]
		>>> model = ps.get_heirarchical_clustering(sequences, ps.get_optimal_distance)
		>>> ps.plot_dendrogram(model) #doctest: +SKIP

	"""

	# Create linkage matrix and then plot the dendrogram
	# create the counts of samples under each node
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

	plot = scipy.cluster.hierarchy.dendrogram(linkage_matrix, **kwargs)

	return plot