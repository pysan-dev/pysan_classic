import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def generate_sequence(length, alphabet):
    """
    Generates a random sequence of a given length, given an alphabet of elements.
    This is useful for benchmarking function performance, and creating examples in the docs.
    
    Example
    --------
    >>> ps.generate_sequence(12, [1,2,3])
    """
    return [random.choice(alphabet) for x in range(length)]


def plot_sequence(sequence):
	"""
	Creates a standard sequence plot where each element corresponds to a position on the y-axis.
	

	Example
	----------
	.. plot::

		>>> sequence = [1,1,2,1,2,2,3,1,1,2,2,1,2,2,3,1,1,2]
		>>> ps.plot_sequence(sequence)

	"""
	np_sequence = np.array(sequence)
	alphabet_len = len(get_alphabet(sequence))
	
	plt.figure(figsize=[len(sequence)*0.2,alphabet_len * 0.25])
		
	unique_values = list(set(sequence))
	for i, value in enumerate(unique_values):
		
		points = np.where(np_sequence == value, i, np.nan)
		
		plt.scatter(x=range(len(np_sequence)), y=points, marker='s', label=value)
	
	plt.yticks(range(len(unique_values)), unique_values)
	plt.ylim(-1, len(unique_values))
	
	return plt

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
	

def get_unique_ngrams(sequence, n):
	"""
	Creates a list of all unique ngrams found in a given sequence.
	"""
	
	unique_ngrams = []
	for x in range(len(sequence) -  n):
		this_ngram = sequence[x:x + n]
		
		if str(this_ngram) not in unique_ngrams:
			unique_ngrams.append(str(this_ngram))
			
	return [eval(x) for x in unique_ngrams]

def get_all_ngrams(sequence, n):
	"""
	Creates a list of all ngrams found in a given sequence.
	"""
	
	all_ngrams = []
	for x in range(len(sequence) -  n):
		this_ngram = sequence[x:x + n]
		all_ngrams.append(this_ngram)
			
	return all_ngrams

def get_ngram_universe(sequence, n):
	"""
	Computes the universe of possible ngrams given a sequence. Where n is equal to the length of the sequence, the resulting number represents the sequence universe.
	"""
	# if recurrance is possible, the universe is given by k^t (SSA pg 68)
	k = len(set(sequence))
	if k > 10 and n > 10:
		return 'really big'
	return k**n



def describe(sequence):
	"""
	Prints useful descriptive details of a given sequence to the console.
	"""
	print('length:', len(sequence))
	print('alphabet:', get_alphabet(sequence))
	print('sequence universe:', f'{get_ngram_universe(sequence, len(sequence)):,}')
	print('unique bigrams:', get_unique_ngrams(sequence, 2))
	print('bigram universe:', get_ngram_universe(sequence, 2))

	details = {
	'length': len(sequence),
	'alphabet': get_alphabet(sequence),
	'sequence_universe': get_ngram_universe(sequence, len(sequence)),
	'unique_bigrams': get_unique_ngrams(sequence, 2),
	'bigram_universe' : get_ngram_universe(sequence, 2)
	}
	return details

def get_element_prevalence(sequence):
    
    elements = ps.get_alphabet(sequence)
    
    prevalences = {}
    for element in elements:
        prevalences[element] = sequence.count(element)
        
    return prevalences

def get_element_frequency(sequence):
    """
    Computes the relative frequency of each element in a sequence, returning a dictionary where each key is an element and each value is that elements relative frequency.
    """
    
    elements = ps.get_alphabet(sequence)
    
    prevalences = {}
    for element in elements:
        prevalences[element] = sequence.count(element) / len(sequence)
        
    return prevalences

def get_transition_matrix(sequence):
	"""
	Computes a transition matrix for each bigram in a sequence.
	The resulting matrix can be interpreted by reading along the top row first, then down the side, indicating from the element in the top row to the element along the side.
	For example, to find the number of transitions from element 2 to element 3, find element 2 across the top, then follow that column down until it reaches element 3 on the side.

	Examples
	----------
	>>> sequence = ['cook','exercise','sleep','sleep','cook','exercise','sleep']
	>>> ps.get_transition_matrix(sequence)
		cook  exercise  sleep
	cook       0.0       0.0    1.0
	exercise   2.0       0.0    0.0
	sleep      0.0       1.0    1.0

	"""
	alphabet = get_alphabet(sequence)
	all_ngrams = get_all_ngrams(sequence, 2)
	
	transition_matrix = np.zeros((len(alphabet), len(alphabet)))
	descriptive_matrix = np.zeros((len(alphabet), len(alphabet)))
	
	for x, element_row in enumerate(alphabet):
		for y, element_column in enumerate(alphabet):
			current_ngram = [element_row, element_column]
			descriptive_matrix[x,y] = str(current_ngram)
			#print('from', current_ngram[0], 'to', current_ngram[1], ':', all_ngrams.count(current_ngram))
			transition_matrix[x, y] = all_ngrams.count(current_ngram)
			
	tm_df = pd.DataFrame(transition_matrix, columns=alphabet, index=alphabet)
	return tm_df

