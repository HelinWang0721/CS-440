# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import numpy as np


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.39, bigram_laplace=0.99, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    # get the vocabulary
    vocab = set()
    for doc in train_set:
        vocab.update(doc)
    #get the bigram gram vocabulary
    bigram_vocab = set()
    for doc in train_set:
        bigram_vocab.update(zip(doc, doc[1:]))

    # get the count of each word in each class
    pos_unigram_counts = Counter()
    neg_unigram_counts = Counter()

    pos_bigram_counter = Counter()
    neg_bigram_counter = Counter()
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            pos_unigram_counts.update(train_set[i])
            pos_bigram_counter.update(zip(train_set[i], train_set[i][1:]))
        else:
            neg_unigram_counts.update(train_set[i])
            neg_bigram_counter.update(zip(train_set[i], train_set[i][1:]))

    # get the unigram total count of each class
    pos_total = sum(pos_unigram_counts.values()) #P all words in positive class
    neg_total = sum(neg_unigram_counts.values()) #P all words in negative class

    # get the bigram total count of each class
    pos_bigram_total = sum(pos_bigram_counter.values()) #P all words in positive class
    neg_bigram_total = sum(neg_bigram_counter.values()) #P all words in negative class
    
    #get the negative prior
    neg_prior = 1 - pos_prior

    # get the unigram conditional probabilities
    pos_cond_probs = {}
    neg_cond_probs = {}

    for word in vocab:

        pos_cond_probs[word] = (pos_unigram_counts[word] + unigram_laplace) / (pos_total + unigram_laplace * (len(vocab)+1))
        neg_cond_probs[word] = (neg_unigram_counts[word] + unigram_laplace) / (neg_total + unigram_laplace * (len(vocab)+1))

    # get the bigram conditional probabilities
    pos_bigram_cond_probs = {}
    neg_bigram_cond_probs = {}
    
    for bigram in bigram_vocab:

        pos_bigram_cond_probs[bigram] = (pos_bigram_counter[bigram] + bigram_laplace) / (pos_bigram_total + bigram_laplace * (len(bigram_vocab)+1))
        neg_bigram_cond_probs[bigram] = (neg_bigram_counter[bigram] + bigram_laplace) / (neg_bigram_total + bigram_laplace * (len(bigram_vocab)+1))
    # predict the labels
    yhats = []

    for doc in tqdm(dev_set, disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(neg_prior)
        
        for word in doc:
            if word in vocab:
                pos_prob += (1 - bigram_lambda)*math.log(pos_cond_probs[word])
                neg_prob += (1 - bigram_lambda)*math.log(neg_cond_probs[word])

        for i in range(len(doc)-1):
            bigram = (doc[i], doc[i+1])
            # predict by using bigram
            if bigram in bigram_vocab:
                pos_prob += math.log(bigram_lambda * pos_bigram_cond_probs[bigram] + (1 - bigram_lambda) * pos_cond_probs[doc[i+1]])
                neg_prob += math.log(bigram_lambda * neg_bigram_cond_probs[bigram] + (1 - bigram_lambda) * neg_cond_probs[doc[i+1]])

        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats