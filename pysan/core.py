import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, copy
import matplotlib.cm as cm

def generate_sequence(length, alphabet):
	"""
	Generates a random sequence of a given length, given an alphabet of elements.
	This is useful for benchmarking function performance, and creating examples in the docs.
	
	Example
	--------
	>>> ps.generate_sequence(12, [1,2,3])
	"""
	return [random.choice(alphabet) for x in range(length)]


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

def get_ngram_prevalence(sequence, n):
	"""
	Computes the prevalence of ngrams in a sequence, returning a dictionary where each key is an ngram, and each value is the number of times that ngram appears in the sequence.
	
	Parameters
	-------------
	sequence : list(int)
		A sequence of elements, encoded as integers e.g. [1,3,2,1].
	n: int
		The number of elements in the ngrams to extract.
	
	"""
	
	ngrams = get_unique_ngrams(sequence, n)
	
	ngram_prevalence = {str(i):0 for i in ngrams}    
	
	for x in range(len(sequence) -  n):
		this_ngram = sequence[x:x + n]
		ngram_prevalence[str(this_ngram)] += 1
	
	return ngram_prevalence


def describe(sequence):
	"""
	Computes descriptive properties of a given sequence, returning a dictionary containing the keys: 
	{'length','alphabet','sequence_universe','unique_bigrams','bigram_universe'}.

	Example
	---------
	>>> sequence = [1,1,2,1,2,2,3,4,2]
	>>> ps.describe(sequence)

	"""
	details = {
	'length': len(sequence),
	'alphabet': get_alphabet(sequence),
	'sequence_universe': get_ngram_universe(sequence, len(sequence)),
	'unique_bigrams': len(get_unique_ngrams(sequence, 2)),
	'bigram_universe' : get_ngram_universe(sequence, 2)
	}
	return details

def get_element_prevalence(sequence):
	
	elements = get_alphabet(sequence)
	
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

def get_transition_matrix(sequence, alphabet=None, verbose=False):
	"""
	Computes a transition matrix for each bigram in a sequence.
	The resulting matrix can be interpreted by reading along the side first, then across the top, indicating from the element in down the side to the element along the top.
	For example, to find the number of transitions from element 2 to element 3, find element 2 down the side, then follow that row across until it reaches element 3 across the top.

	Examples
	----------
	>>> sequence = ['cook','exercise','sleep','sleep','cook','exercise','sleep']
	>>> ps.get_transition_matrix(sequence)
		cook  exercise  sleep
	cook       0.0       0.0    1.0
	exercise   2.0       0.0    0.0
	sleep      0.0       1.0    1.0

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

	if verbose:
		de_df = pd.DataFrame(descriptive_matrix, columns=alphabet, index=alphabet)
		print(de_df)
	tm_df = pd.DataFrame(transition_matrix, columns=alphabet, index=alphabet)
	return tm_df








# ====================================================================================
# PLOTTING FUNCTIONS
# ====================================================================================


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


def plot_element_prevalence(sequence):
	"""
	Plots the number of occurances of each unique element in a given sequence.

	"""
	
	prev = get_element_prevalence(sequence)
	prev = {k: prev[k] for k in sorted(prev, key=prev.get)}

	xdata = [str(key) for key,value in prev.items()]
	ydata = [value for key,value in prev.items()]
	
	plt.figure()
	plt.barh(xdata, ydata, label='element count')
	plt.gca().yaxis.grid(False)
	plt.legend()
	return plt

	
def plot_ngram_prevalence(sequence, n):
	"""
	Plots the number of occurances of ngrams in a given sequence.

	"""
	
	ngram_prevalence = get_ngram_prevalence(sequence, n)
	ngram_prevalence = {k: ngram_prevalence[k] for k in sorted(ngram_prevalence, key=ngram_prevalence.get)}
	
	xdata = [key[1:len(key)-1].replace(', ', ', ') for key,value in ngram_prevalence.items()]
	ydata = [value for key,value in ngram_prevalence.items()]
	
	plt.figure()
	plt.barh(xdata, ydata, label=str(n) +'-gram')
	plt.gca().yaxis.grid(False)
	plt.legend()
	return plt


def color_matrix(matrix, cmap='summer'):
	"""
	Creates a shaded matrix based on the values in that matrix. This is most useful when given a transition matrix as it intuitively plots the prevalence of transitions between states. The y axis represents the elements at position n, and the x axis represents the elements at position n+1.

	Parameters
	-----------
	matrix: DataFrame
		A 2D matrix of values in the form of a pandas dataframe. Column names are used as axis ticks.
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
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	ax = plt.gca()
	ax.xaxis.set_label_position('top')
	plt.imshow(np.array(values).astype(np.float), cmap=current_cmap)
	plt.yticks(range(len(matrix.columns)), list(matrix.columns))
	plt.xticks(range(len(matrix.columns)), list(matrix.columns))
	cbar = plt.colorbar()
	#cbar.set_ticks([-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100])
	#cbar.set_ticklabels([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
	plt.ylabel("n")
	plt.xlabel("n+1")
	plt.grid(False)
	return plt



# print to console to confirm everything is loaded properly
print('pysan ready')