# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:38:21 2020

@author: Meet Gandhi
"""

import numpy as np
import nltk
from nltk import bigrams
import itertools
import pandas as pd

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))

    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = co_occurrence_matrix[pos_current][pos_previous] + count
        co_occurrence_matrix[pos_previous][pos_current] = co_occurrence_matrix[pos_previous][pos_current] + count

    # return the matrix and the index
    return co_occurrence_matrix, vocab_index

def get_embedding_lower_bound(corpus):

    def generate_co_occurrence_matrix(corpus):
        vocab = set(corpus)
        vocab = list(vocab)
        vocab_index = {word: i for i, word in enumerate(vocab)}

        # Create bigrams from all words in corpus
        bi_grams = list(bigrams(corpus))

        # Frequency distribution of bigrams ((word1, word2), num_occurrences)
        bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

        # Initialise co-occurrence matrix
        # co_occurrence_matrix[current][previous]
        co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

        # Loop through the bigrams taking the current and previous word,
        # and the number of occurrences of the bigram.
        for bigram in bigram_freq:
            current = bigram[0][1]
            previous = bigram[0][0]
            count = bigram[1]
            pos_current = vocab_index[current]
            pos_previous = vocab_index[previous]
            co_occurrence_matrix[pos_current][pos_previous] = co_occurrence_matrix[pos_current][pos_previous] + count
            co_occurrence_matrix[pos_previous][pos_current] = co_occurrence_matrix[pos_previous][pos_current] + count

        # return the matrix and the index
        return co_occurrence_matrix, vocab_index

    co_occurrence_matrix, vocab_index = generate_co_occurrence_matrix(corpus)

    index_vocab = dict(zip(vocab_index.values(),vocab_index.keys()))
    cosine_data_matrix = cosine_similarity(co_occurrence_matrix,co_occurrence_matrix)
    cosine_data_matrix = np.around(cosine_data_matrix,5)
    cosine_data_matrix_utri = np.triu(cosine_data_matrix, k=1)

    unique_similarity_values = np.unique(cosine_data_matrix_utri, return_counts = True)
    vocab = list(vocab_index.keys())

    look_up_dict = {6:4,10:5,16:6,28:13,30:14,36:15,42:16,51:17,61:18,
                    76:19,96:20,126:21,176:22,276: 41,288:44,344:43}
    lambda_k = []
    for vlaue in unique_similarity_values[0][1:]:
        G = nx.Graph()
        G. add_nodes_from(vocab)

        index = np.where(cosine_data_matrix_utri==vlaue)

        edge_index = list(zip(index[0],index[1]))
        edge_list = [(index_vocab[pair[0]], index_vocab[pair[1]]) for pair in edge_index]
        G.add_edges_from(edge_list)
        cliques = list(nx.algorithms.clique.find_cliques(G))
        clique_lens = [len(x) for x in cliques]
        equidistant_points = max(clique_lens)

        if equidistant_points in look_up_dict.keys():
            lambda_k.append(look_up_dict[equidistant_points])
            if equidistant_points == 28:
                print('Also experiment with a few values between [7,12]')
            if equidistant_points == 276:
                print('Also experiment with 23 and a few values between [24,41]')
        if equidistant_points > 344:
            lambda_k.append(128)
        if equidistant_points<4:
            lambda_k.append(3)
        else:
            i=0
            while list(look_up_dict.keys())[i]<equidistant_points:
                i = i+1
            one = list(look_up_dict.keys())[i]
            two = list(look_up_dict.keys())[i-1]
            lambda_k.append(int(0.5*(look_up_dict[one] + look_up_dict[two])))

    lower_bound_dim = max(lambda_k)

    return lower_bound_dim

def get_embedding_dims_pca(corpus):

    def generate_co_occurrence_matrix(corpus):
        vocab = set(corpus)
        vocab = list(vocab)
        vocab_index = {word: i for i, word in enumerate(vocab)}

        # Create bigrams from all words in corpus
        bi_grams = list(bigrams(corpus))

        # Frequency distribution of bigrams ((word1, word2), num_occurrences)
        bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

        # Initialise co-occurrence matrix
        # co_occurrence_matrix[current][previous]
        co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

        # Loop through the bigrams taking the current and previous word,
        # and the number of occurrences of the bigram.
        for bigram in bigram_freq:
            current = bigram[0][1]
            previous = bigram[0][0]
            count = bigram[1]
            pos_current = vocab_index[current]
            pos_previous = vocab_index[previous]
            co_occurrence_matrix[pos_current][pos_previous] = co_occurrence_matrix[pos_current][pos_previous] + count
            co_occurrence_matrix[pos_previous][pos_current] = co_occurrence_matrix[pos_previous][pos_current] + count

        # return the matrix and the index
        return co_occurrence_matrix, vocab_index

    co_occurrence_matrix, vocab_index = generate_co_occurrence_matrix(corpus)

    pca = PCA()
    pca.fit(co_occurrence_matrix)
    for i in range(len(pca.explained_variance_ratio_)):
        if pca.explained_variance_ratio_[:i].sum()>= 0.95:
            explained_var = pca.explained_variance_ratio_[:i].sum()
            min_dims = i
            break
    return min_dims, explained_var



text_data = [['Where', 'Python', 'is', 'used'],
             ['What', 'is', 'Python','used', 'in'],
             ['Why', 'Python', 'is', 'best'],
             ['What', 'companies', 'use', 'Python']]

# Create one list using many lists
data = list(itertools.chain.from_iterable(text_data))
matrix, vocab_index = generate_co_occurrence_matrix(data)

data_matrix = pd.DataFrame(matrix, index=vocab_index, columns=vocab_index)

min_dims = get_embedding_lower_bound(matrix,vocab_index)

# =============================================================================
#
# =============================================================================

df_train = pd.read_pickle('df_train.pkl')

data = list(itertools.chain.from_iterable(df_train['marked_journey']))

matrix, vocab_index = generate_co_occurrence_matrix(data)

min_dims = get_embedding_lower_bound(data)

# =============================================================================
#
# =============================================================================

data = 'Novartis would like to understand how the severe asthma treatment journey will evolve given new entrants in the next several years Novartis US asthma presence currently consists of Xolair additional products are under development including QAW QGE and QVM As new treatments come to market e g triple FDC biologics etc the market is likely to change significantly which will impact how Novartis positions its asthma product portfolio Background Novartis would like to understand the severe asthma patient population and where opportunities lie in the future Previous research on the Asthma Patient Journey has been conducted across multiple countries but have not been aggregated in a consistent manner Develop a global Asthma Patient Journey that can provide cross regional guidance on brand and portfolio strategy Gain insight into future treatment paradigm shifts that will impact clinical development and commercial execution Key Objectives Prioritized opportunities to pursue within asthma Novartis portfolio co positioning for current and future asthma products Xolair QAW QGE QVM Commercial Decision s that Novartis needs to Inform'

data = data.split(' ')

matrix, vocab_index = generate_co_occurrence_matrix(data)

min_dims = get_embedding_lower_bound(data)

min_dims = get_embedding_dims_pca(data)




