import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random, copy
import matplotlib.cm as cm
import itertools
import scipy.stats
import math
import statistics
import pysan.multisequence as pysan_ms
from itertools import combinations

random.seed('12345')

def generate_sequence(length, alphabet):
	"""
	Generates a random sequence of a given length, given an alphabet of elements.
	This is useful for benchmarking function performance, and creating examples in the docs.
	
	Example
	--------
	>>> ps.generate_sequence(12, [1,2,3])
	[2, 3, 3, 3, 2, 2, 2, 1, 3, 3, 2, 2]
	"""
	return [random.choice(alphabet) for x in range(length)]

def full_analysis(sequence):
	"""
	UC Computes a collection of information on a given sequence plus a collection of plots.
	
	"""
	
	details = describe(sequence)
	sequence_plot = plot_sequence(sequence)
	tm = plot_transition_matrix(sequence)
	
	element_counts = get_element_counts(sequence)
	element_prevalence = plot_element_counts(sequence)
	bigrams = plot_ngram_counts(sequence, 2)
	trigrams = plot_ngram_counts(sequence, 3)
	
	print(details)
	print(element_counts, element_prevalence)
	sequence_plot.show()
	tm.show()
	bigrams.show()
	trigrams.show()
	
	return None

def describe(sequence):
	"""
	Computes descriptive properties of a given sequence, returning a dictionary containing the keys: 
	{'length', 'alphabet', 'sequence_universe', 'unique_bigrams', 'bigram_universe', 'entropy'}.

	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3,4,2]
	>>> ps.describe(sequence) #doctest: +NORMALIZE_WHITESPACE
	{'length': 9, 
	 'alphabet': {1, 2, 3, 4}, 
	 'is_recurrent': True, 
	 'entropy': 0.8763576394898526, 
	 'complexity': 0.6885628567541515, 
	 'turbulence': 7.868228975239414, 
	 'element_counts': {1: 3, 2: 4, 3: 1, 4: 1}, 
	 'first_positions': {1: 0, 2: 2, 3: 6, 4: 7}, 
	 'ntransitions': 6, 
	 'sequence_universe': 262144, 
	 'distinct_subsequences': 175, 
	 'unique_bigrams': 7, 
	 'bigram_universe': 16, 
	 'longest_spell': {'element': 1, 'count': 2, 'start': 0}}

	"""
	details = {
	'length': len(sequence),
	'alphabet': get_alphabet(sequence),
	'is_recurrent': is_recurrent(sequence),
	'entropy' : get_entropy(sequence),
	'complexity': get_complexity(sequence),
	'turbulence': get_turbulence(sequence),
	'element_counts': get_element_counts(sequence),
	'first_positions': get_first_positions(sequence),
	'ntransitions' : get_ntransitions(sequence),
	'sequence_universe': get_ngram_universe(sequence, len(sequence)),
	'distinct_subsequences': get_ndistinct_subsequences(sequence),
	'unique_bigrams': len(get_unique_ngrams(sequence, 2)),
	'bigram_universe' : get_ngram_universe(sequence, 2),
	'longest_spell': get_longest_spell(sequence)
	# spell durations here
	}
	return details

# ====================================================================================
# SUMMARY STATISTICS
# ====================================================================================

def is_recurrent(sequence):
	"""
	Returns true if the given sequence is recurrent (elements can exist more than once), otherwise returns false.

	Example
	---------
	>>> sequence = [1,2,3,4,5]
	>>> ps.is_recurrent(sequence)
	False
	>>> sequence = [1,1,2,2,3]
	>>> ps.is_recurrent(sequence)
	True


	"""
	
	element_counts = get_element_counts(sequence)
	
	truths = [count > 1 for element, count in element_counts.items()]
	
	if True in truths:
	
		return True

	return False

def get_entropy(sequence):
	"""
	Computes the normalised `Shannon entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of a given sequence, using the `scipy.stats.entropy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html>`_ implementation.
	Note that this measure is insensitive to transition frequency or event order, so should be used in conjunction with other measures.
	
	Example
	--------
	>>> low_entropy_sequence = [1,1,1,1,1,1,1,2]
	>>> ps.get_entropy(low_entropy_sequence)
	0.543...
	>>> high_entropy_sequence = [1,2,2,3,4,3]
	>>> ps.get_entropy(high_entropy_sequence)
	0.959...
	
	"""
	
	alphabet = get_alphabet(sequence)

	entropy = 0
	for state in alphabet:
		proportion_occurances = sequence.count(state) / len(sequence)
		entropy += proportion_occurances * math.log(proportion_occurances)
		
	maximal_occurances = 1 / len(alphabet)
	alphabet_entropy = sum([maximal_occurances * math.log(maximal_occurances) for x in alphabet])
		
	if alphabet_entropy == 0:
		return 0
	
	return -entropy / -alphabet_entropy

def get_turbulence(sequence):
	"""
	
	Computes turbulence for a given sequence, based on `Elzinga & Liefbroer's 2007 definition <https://www.researchgate.net/publication/225402919_De-standardization_of_Family-Life_Trajectories_of_Young_Adults_A_Cross-National_Comparison_Using_Sequence_Analysis>`_ which is also implemented in the `TraMineR <http://traminer.unige.ch/doc/seqST.html>`_ sequence analysis library.
	
	Example
	--------
	>>> sequence = [1,1,2,2,3]
	>>> ps.get_turbulence(sequence)
	5.228...

	"""

	phi = get_ndistinct_subsequences(sequence)

	#print('phi', phi)

	state_durations = [value for key, value in get_spells(sequence)]

	#print('durations', state_durations)
	#print('mean duration', statistics.mean(state_durations))

	variance_of_state_durations = statistics.variance(state_durations)

	#print('variance', variance_of_state_durations)

	tbar = statistics.mean(state_durations)

	maximum_state_duration_variance = (len(sequence) - 1) * (1 - tbar) ** 2

	#print('smax', maximum_state_duration_variance)

	top_right = maximum_state_duration_variance + 1
	bot_right = variance_of_state_durations + 1

	turbulence = math.log2(phi * (top_right / bot_right))

	#print('turbulence', turbulence)

	return turbulence

def get_complexity(sequence):
	"""
	
	Computes the complexity of a given sequence, based on TraMineR's `seqici <http://traminer.unige.ch/doc/seqici.html>`_ method.
	


	"""
	
	alphabet = get_alphabet(sequence)

	pre_log = 1 / len(alphabet)
	hmax = -math.log(pre_log)
	#print('hmax', hmax)
	if hmax == 0:
		return 0 # all identical elements, no complexity

	hs = get_entropy(sequence)
	#print('hs', hs)

	qs = get_ntransitions(sequence)
	#print('qs', qs)

	qmax = len(sequence) - 1
	#print('qmax', qmax)

	norm_transitions = qs / qmax
	norm_entropy = hs / hmax

	#print('nt', norm_transitions)
	#print('ne', norm_entropy)

	complexity = math.sqrt(norm_transitions * norm_entropy)

	#print('complexity', complexity)
	return complexity
	
def get_routine(sequence, duration):
	"""
	Computes a normalised measure of routine within a sequence for a given duration within that sequence.
	E.g. with a sequence where each element is one day, calling get_routine() with a duration of 7 would look at weekly routines.
	Note that this routine measure is identical to the multisequence measure of synchrony, but applied within-sequence in duration length chunks.
	
	Example
	---------
	>>> sequence = [1,1,2,2,3,1,1,2,3,2,1,1,3,2,2]
	>>> ps.get_routine(sequence, 5)
	0.4
	
	"""
	
	if len(sequence) % duration != 0:
		raise Exception('sequence not divisible by interval, check data input')
	
	num_cycles = int(len(sequence) / duration)
	cycles = [sequence[n * duration:n * duration + duration] for n in range(num_cycles)]
	
	return pysan_ms.get_synchrony(cycles)

# ====================================================================================
# ELEMENTS
# ====================================================================================

def get_alphabet(sequence):
	"""
	Computes the alphabet of a given sequence (set of its unique elements).

	Parameters
	----------
	sequence : int
		A sequence of elements, encoded as integers e.g. [1,3,2,1].

	Example
	----------

	>>> sequence = [1,1,2,1,2,2,3,4,2]
	>>> ps.get_alphabet(sequence)
	{1, 2, 3, 4}

	"""
	return set(sequence)
	
def get_first_positions(sequence):
	"""
	Reports the first occurance of each element in the sequence in a dictionary, with each element as keys, and their first position as values.
	
	Example
	---------
	>>> sequence = [1,1,2,3,4]
	>>> ps.get_first_positions(sequence)
	{1: 0, 2: 2, 3: 3, 4: 4}


	"""
	unique_elements = list(set(sequence))

	first_positions = {}
	for element in unique_elements:
		first_positions[element] = sequence.index(element)
		
	return first_positions

def get_element_counts(sequence):
	"""
	Counts the number of occurances for each element in a sequence, returning a dictionary containing the elements as keys and their counts as values.
	
	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3,4,2]
	>>> ps.get_element_counts(sequence)
	{1: 3, 2: 4, 3: 1, 4: 1}
	
	"""
	alphabet = get_alphabet(sequence)
	
	counts = {}
	for element in alphabet:
		counts[element] = sequence.count(element)
		
	return counts

def get_element_frequency(sequence):
	"""
	Computes the relative frequency (aka prevalence or unconditional probability) of each element in a sequence, returning a dictionary where each key is an element and each value is that elements relative frequency.
	
	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3,4,2,1]
	>>> ps.get_element_frequency(sequence)
	{1: 0.4, 2: 0.4, 3: 0.1, 4: 0.1}

	"""
	
	alphabet = get_alphabet(sequence)
	
	prevalences = {}
	for element in alphabet:
		prevalences[element] = sequence.count(element) / len(sequence)
		
	return prevalences


# ====================================================================================
# SUBSEQUENCES
# ====================================================================================

# NGRAMS

def get_subsequences(sequence):
	"""
	Computes the actual possible subsequences in a given sequence, returning them as a list of lists.
	Note that this does not include a single empty list as a subsequence.
	This method is based on a similar implementation available `here <https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-33.php>`_.
	
	Example
	--------
	>>> sequence = [1,2,3]
	>>> ps.get_subsequences(sequence)
	[[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
	
	"""
	
	subsequences = []
	for i in range(0, len(sequence)+1):
		temp = [list(x) for x in combinations(sequence, i)]
		if len(temp)>0:
			subsequences.extend(temp)
	return subsequences[1:]

def get_ndistinct_subsequences(sequence):
	"""
	Computes the number of distinct subsequences for a given sequence, based on original implementation by 
	Mohit Kumar available `here <https://www.geeksforgeeks.org/count-distinct-subsequences/>`_.
	
	Example
	--------
	>>> sequence = [1,2,1,3]
	>>> ps.get_ndistinct_subsequences(sequence)
	14
	
	"""
	# this implementation works on strings, so parse non-strings to strings
	if sequence is not str:
		sequence = [str(e) for e in sequence]

	# create an array to store index of last
	last = [-1 for i in range(256 + 1)] # hard-coded value needs explaining -ojs
	 
	# length of input string
	sequence_length = len(sequence)
	 
	# dp[i] is going to store count of discount subsequence of length of i
	dp = [-2 for i in range(sequence_length + 1)]
	  
	# empty substring has only one subseqence
	dp[0] = 1
	 
	# Traverse through all lengths from 1 to n 
	for i in range(1, sequence_length + 1):
		 
		# number of subseqence with substring str[0...i-1]
		dp[i] = 2 * dp[i - 1]
 
		# if current character has appeared before, then remove all subseqences ending with previous occurrence.
		if last[ord(sequence[i - 1])] != -1:
			dp[i] = dp[i] - dp[last[ord(sequence[i - 1])]]
			
		last[ord(sequence[i - 1])] = i - 1    
	
	return dp[sequence_length]

def get_unique_ngrams(sequence, n):
	"""
	Creates a list of all unique ngrams found in a given sequence.
	
	Example
	---------
	>>> sequence = [2,1,1,4,2,2,3,4,2,1,1]
	>>> ps.get_unique_ngrams(sequence, 3) #doctest: +NORMALIZE_WHITESPACE
	[[2, 1, 1],
	 [1, 1, 4],
	 [1, 4, 2],
	 [4, 2, 2],
	 [2, 2, 3],
	 [2, 3, 4],
	 [3, 4, 2],
	 [4, 2, 1]]
	"""

	
	unique_ngrams = []
	for x in range(len(sequence) -  n + 1):
		this_ngram = sequence[x:x + n]
		
		if str(this_ngram) not in unique_ngrams:
			unique_ngrams.append(str(this_ngram))
			
	return [eval(x) for x in unique_ngrams]

def get_all_ngrams(sequence, n):
	"""
	Creates a list of all ngrams found in a given sequence.

	
	Example
	---------
	>>> sequence = [2,1,1,4,2,2,3,4,2,1,1]
	>>> ps.get_unique_ngrams(sequence, 3) #doctest: +NORMALIZE_WHITESPACE
	[[2, 1, 1],
	 [1, 1, 4],
	 [1, 4, 2],
	 [4, 2, 2],
	 [2, 2, 3],
	 [2, 3, 4],
	 [3, 4, 2],
	 [4, 2, 1]]

	"""
	
	all_ngrams = []
	for x in range(len(sequence) -  n + 1):
		this_ngram = sequence[x:x + n]
		all_ngrams.append(this_ngram)
			
	return all_ngrams

def get_ngram_universe(sequence, n):
	"""
	Computes the universe of possible ngrams given a sequence. Where n is equal to the length of the sequence, the resulting number represents the sequence universe.

	Example
	--------
	>>> sequence = [2,1,1,4,2,2,3,4,2,1,1]
	>>> ps.get_ngram_universe(sequence, 3)
	64

	"""
	# if recurrance is possible, the universe is given by k^t (SSA pg 68)
	k = len(set(sequence))
	if k > 10 and n > 10:
		return 'really big'
	return k**n

def get_ngram_counts(sequence, n):
	"""
	Computes the prevalence of ngrams in a sequence, returning a dictionary where each key is an ngram, and each value is the number of times that ngram appears in the sequence.
	
	Parameters
	-------------
	sequence : list(int)
		A sequence of elements, encoded as integers e.g. [1,3,2,1].
	n: int
		The number of elements in the ngrams to extract.
	
	Example
	---------
	>>> sequence = [2,1,1,4,2,2,3,4,2,1,1]
	>>> ps.get_ngram_counts(sequence, 3) #doctest: +NORMALIZE_WHITESPACE
	{'[2, 1, 1]': 2,
	 '[1, 1, 4]': 1,
	 '[1, 4, 2]': 1,
	 '[4, 2, 2]': 1,
	 '[2, 2, 3]': 1,
	 '[2, 3, 4]': 1,
	 '[3, 4, 2]': 1,
	 '[4, 2, 1]': 1}

	"""
	
	ngrams = get_unique_ngrams(sequence, n)
	
	ngram_counts = {str(i):0 for i in ngrams}    
	
	for x in range(len(sequence) -  n + 1):
		this_ngram = sequence[x:x + n]
		ngram_counts[str(this_ngram)] += 1
	
	return ngram_counts


# TRANSITIONS

def get_transitions(sequence):
	"""
	Extracts a list of transitions from a sequence, returning a list of lists containing each transition.

	Example
	--------
	>>> sequence = [1,2,2,1,2,3,2,3,1]
	>>> ps.get_transitions(sequence)
	[[1, 2], [2, 1], [1, 2], [2, 3], [3, 2], [2, 3], [3, 1]]

	"""

	transitions = []
	for position in range(len(sequence) - 1):
		if sequence[position] != sequence[position + 1]:
			transitions.append([sequence[position], sequence[position + 1]])

	return transitions

def get_ntransitions(sequence):
	"""
	Computes the number of transitions in a sequence.
	
	Example
	--------
	>>> sequence = [1,1,1,2,2,3,3,3,4,4]
	>>> ps.get_ntransitions(sequence)
	3
	
	"""
	
	return len(get_transitions(sequence))

def get_transition_matrix(sequence, alphabet=None, verbose=False):
	"""
	Computes a transition matrix for each bigram in a sequence.
	The resulting matrix can be interpreted by reading along the side first, then across the top, indicating from the element in down the side to the element along the top.
	For example, to find the number of transitions from element 2 to element 3, find element 2 down the side, then follow that row across until it reaches element 3 across the top.

	Examples
	----------
	>>> sequence = [1,2,2,1,2,3,2,3,1]
	>>> ps.get_transition_matrix(sequence) #doctest: +NORMALIZE_WHITESPACE
		->1 	->2 	->3
	1-> 	0.0 	2.0 	0.0
	2-> 	1.0 	1.0 	2.0
	3-> 	1.0 	1.0 	0.0
	"""
	if alphabet == None:
		alphabet = get_alphabet(sequence)
	all_ngrams = get_all_ngrams(sequence, 2)

	transition_matrix = np.zeros((len(alphabet), len(alphabet)))
	descriptive_matrix = [['-' for x in range(len(alphabet))] for y in range(len(alphabet))]

	for x, element_row in enumerate(alphabet):
		for y, element_column in enumerate(alphabet):
			current_ngram = [element_row, element_column]
			descriptive_matrix[x][y] = 'n' + str(current_ngram)
			#print('from', current_ngram[0], 'to', current_ngram[1], ':', all_ngrams.count(current_ngram))
			transition_matrix[x, y] = all_ngrams.count(current_ngram)

	# add column & index labelling in TraMineR style
	pre_alphabet = [str(a) + '->' for a in alphabet]
	post_alphabet = ['->' + str(a) for a in alphabet]

	if verbose:
		de_df = pd.DataFrame(descriptive_matrix, columns=post_alphabet, index=pre_alphabet)
		print(de_df)
	tm_df = pd.DataFrame(transition_matrix, columns=post_alphabet, index=pre_alphabet)
	return tm_df

# SPELLS 

def get_spells(sequence):
	"""
	Returns a list of tuples where each tuple holds the element and the length of the spell (also known as run or episode) for each spell in the sequence.
	
	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3]
	>>> ps.get_spells(sequence)
	[(1, 2), (2, 1), (1, 1), (2, 2), (3, 1)]
	"""
	
	# get each spell and its length
	spells = [(k, sum(1 for x in v)) for k,v in itertools.groupby(sequence)]
	# this is functionally equivalent to the following;
	# spells = [(k, len(list(v))) for k,v in itertools.groupby(sequence)]
	
	return spells

def get_longest_spell(sequence):
	"""
	Returns a dict containing the element, count, and starting position of the longest spell in the sequence. The keys of this dict are 'element, 'count', and 'start'.
	
	Example
	--------
	>>> sequence = [1,1,1,4,2,2,3,4,2]
	>>> ps.get_longest_spell(sequence)
	{'element': 1, 'count': 3, 'start': 0}

	"""
	
	spells = get_spells(sequence)

	longest_spell = max(count for element, count in spells)

	for i, (element, count) in enumerate(spells):
		if count == longest_spell:
			# sum the counts of all previous spells to get its starting position
			position_in_sequence = sum(count for _,count in spells[:i])
			
			return {'element':element, 'count':count,'start':position_in_sequence}

def get_spell_durations(sequence):
	"""
	Computes the durations of each spell in the sequence, returning a list.

	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3]
	>>> ps.get_spell_durations(sequence)
	[2, 1, 1, 2, 1]
	"""

	spells = get_spells(sequence)
	durations = [spell[1] for spell in spells]

	return durations

# ====================================================================================
# PLOTTING FUNCTIONS
# ====================================================================================

def plot_sequence(sequence, highlighted_ngrams=[]):
	"""
	Creates a standard sequence plot where each element corresponds to a position on the y-axis.
	The optional highlighted_ngrams parameter can be one or more n-grams to be outlined in a red box.


	Example
	----------
	.. plot::

		>>> sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_sequence(sequence) #doctest: +SKIP
		
	.. plot::

		>>> sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_sequence(sequence, [1,2]) #doctest: +SKIP

	.. plot::

		>>> sequence = [1,2,3,2,3,4,4,3,2,3,1,3,1,2,3,1,3,4,2,3,2,2]
		>>> ps.plot_sequence(sequence, [[1,2,3], [3,4]]) #doctest: +SKIP
	
	"""
	np_sequence = np.array(sequence)
	alphabet_len = len(get_alphabet(sequence))

	plt.figure(figsize=[len(sequence)*0.3,alphabet_len * 0.3])

	unique_values = list(set(sequence))
	for i, value in enumerate(unique_values):

		points = np.where(np_sequence == value, i, np.nan)

		plt.scatter(x=range(len(np_sequence)), y=points, marker='s', label=value, s=35)

	plt.yticks(range(len(unique_values)), unique_values)
	plt.ylim(-1, len(unique_values))
	
	# highlight any of the n-grams given
	
	if highlighted_ngrams != []:
		
		def highlight_ngram(ngram):
			n = len(ngram)
			match_positions = []
			for x in range(len(sequence) -  n + 1):
				this_ngram = sequence[x:x + n]
				if str(this_ngram) == str(ngram):
					match_positions.append(x)

			for position in match_positions:
				bot = min(ngram) - 1.5
				top = max(ngram) - 0.5
				left = position - 0.5
				right = left + n
				
				line_width = 1
				plt.plot([left,right], [bot,bot], color='red', linewidth=line_width)
				plt.plot([left,right], [top,top], color='red', linewidth=line_width)
				plt.plot([left,left], [bot,top], color='red', linewidth=line_width)
				plt.plot([right,right], [bot,top], color='red', linewidth=line_width)
				
		# check if only one n-gram has been supplied
		if type(highlighted_ngrams[0]) is int:
			
			highlight_ngram(highlighted_ngrams)
		
		else: # multiple n-gram's found
			
			for ngram in highlighted_ngrams:
				highlight_ngram(ngram)

	return plt

def plot_sequence_1d(sequence, flat=False):
	"""
	Plots a sequence in one dimension - useful for stacking multiple sequences above one another.
	
	Example
	---------    
	.. plot::

		>>> sequence = [1,1,1,2,2,2,3,1,3,2,2,2,4,4,1,1,1,1,2,1,1]
		>>> ps.plot_sequence_1d(sequence) #doctest: +SKIP
	
	"""
	
	np_sequence = np.array(sequence)
	alphabet_len = len(get_alphabet(sequence))

	plt.figure(figsize=[len(sequence)*0.4, 0.5])

	unique_values = list(set(sequence))
	for i, value in enumerate(unique_values):

		points = np.where(np_sequence == value, 1, np.nan)
		
		plt.bar(range(len(points)), points, width=1, align='edge', label=i)
	
	plt.ylim(-0.3, 1.3)
	plt.tick_params(
		axis='y',
		which='both',
		left=False,
		labelleft=False)
	plt.xlabel('Position, p')
	plt.legend(bbox_to_anchor=(1, 1.2), loc='upper left')
	
	return plt

def plot_element_counts(sequence):
	"""
	Plots the number of occurances of each unique element in a given sequence.

	Example
	---------
	.. plot::

		>>> sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_element_counts(sequence) #doctest: +SKIP

	"""
	
	prev = get_element_counts(sequence)
	prev = {k: prev[k] for k in sorted(prev, key=prev.get)}

	xdata = [str(key) for key,value in prev.items()]
	ydata = [value for key,value in prev.items()]
	
	plt.figure()
	plt.barh(xdata, ydata, label='element count')
	plt.gca().yaxis.grid(False)
	plt.legend()
	return plt

def plot_ngram_counts(sequence, n):
	"""
	Plots the number of occurances of ngrams in a given sequence.

	Example
	---------
	.. plot::

		>>> sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_ngram_counts(sequence, 3) #doctest: +SKIP

	"""
	
	ngram_counts = get_ngram_counts(sequence, n)
	ngram_counts = {k: ngram_counts[k] for k in sorted(ngram_counts, key=ngram_counts.get)}
	
	xdata = [key[1:len(key)-1].replace(', ', ', ') for key,value in ngram_counts.items()]
	ydata = [value for key,value in ngram_counts.items()]
	
	plt.figure()
	plt.barh(xdata, ydata, label=str(n) +'-gram')
	plt.gca().yaxis.grid(False)
	plt.legend()
	return plt

def plot_transition_matrix(sequence, cmap='summer'):
	"""
	Computes and plots a transition matrix, returning a colored matrix with elements at position n up the y axis, and elements at position n+1 along the x axis.

	Example
	---------
	.. plot::

		>>> sequence = [1,1,2,1,4,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_transition_matrix(sequence) #doctest: +SKIP

	"""

	tm = get_transition_matrix(sequence)
	plot = color_matrix(tm, cmap=cmap)
	return plot


def color_matrix(matrix, cmap='summer'):
	"""
	Creates a shaded matrix based on the values in that matrix. This is most useful when given a transition matrix as it intuitively plots the prevalence of transitions between states. The y axis represents the elements at position n, and the x axis represents the elements at position n+1.

	Parameters
	-----------
	matrix: DataFrame
		A 2D matrix of values in the form of a `pandas dataframe <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_. Column names are used as axis ticks.
	cmap: string
		The name of a `matplotlib color map <https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html>`_.
	

	"""


	results_size = len(matrix.columns)
	values = np.empty((results_size, results_size), dtype=object)
	for r, row in enumerate(matrix.values):
		for e, element in enumerate(row):
			if element == "-":
				values[r, e] = 100
				continue
			if element == "":
				values[r, e] = np.nan
				continue
			if "*" in str(element):
				value = element.replace("*", "")
				values[r, e] = float(value)
			else:
				values[r, e] = element

	current_cmap = copy.copy(cm.get_cmap(cmap))
	current_cmap.set_bad(color="white")

	plt.figure()

	# this one-lines sets the x axis to appear at the top of this plot only
	with plt.rc_context({'xtick.bottom':False, 'xtick.labelbottom':False, 'xtick.top':True, 'xtick.labeltop':True}):
		ax = plt.gca()
		ax.xaxis.set_label_position('top')
		plt.imshow(np.array(values).astype(np.float), cmap=current_cmap)

		plt.yticks(range(len(matrix.index)), list(matrix.index))
		plt.xticks(range(len(matrix.columns)), list(matrix.columns))
		cbar = plt.colorbar()
		#cbar.set_ticks([-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100])
		#cbar.set_ticklabels([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
		plt.grid(False)
		return plt

# print to console to confirm everything is loaded properly
print('pysan ready')