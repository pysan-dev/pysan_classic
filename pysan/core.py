import numpy as np
import matplotlib.pyplot as plt

def plot_sequence(sequence):
    """
    Creates a standard sequence plot where each element corresponds to a position on the y-axis.
    """
    np_sequence = np.array(sequence)
    
    plt.figure(figsize=[len(sequence)*0.2,3])
        
    unique_values = list(set(sequence))
    for i, value in enumerate(unique_values):
        
        points = np.where(np_sequence == value, i, np.nan)
        
        plt.scatter(x=range(len(np_sequence)), y=points, marker='s', label=value)
    
    plt.yticks(range(len(unique_values)), unique_values)
    
    return plt

def get_alphabet(sequence):
    """
    Computes the alphabet of a given sequence (set of its unique elements).
    """
    return set(sequence)


def get_ngrams(sequence, n):
    """
    Creates a list of all unique ngrams found in a given sequence. Currently only works for bigrams.
    """
    all_ngram_indexes = [[x, x + 1] for x in range(len(sequence) - 1)]
    
    all_ngrams = []
    for ngram_indexes in all_ngram_indexes:
        this_ngram = str([sequence[x] for x in ngram_indexes])
        all_ngrams.append(this_ngram)
    
    unique_ngrams = [eval(ngram) for ngram in set(all_ngrams)]
    
    return unique_ngrams


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
    print('sequence universe:', get_ngram_universe(sequence, len(sequence)))
    print('bigrams:', get_ngrams(sequence, 2))
    print('bigram universe:', get_ngram_universe(sequence, 2))
