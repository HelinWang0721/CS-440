# naive_bayes.py
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
import numpy as np
import nltk as nltk
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(-1)

    return yhats
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    # get the vocabulary
    vocab = set()
    for doc in train_set:
        vocab.update(doc)

    # get the count of each word in each class
    pos_counts = Counter()
    neg_counts = Counter()
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            pos_counts.update(train_set[i])
        else:
            neg_counts.update(train_set[i])

    # get the total count of each class
    pos_total = sum(pos_counts.values()) #P all words in positive class
    neg_total = sum(neg_counts.values()) #P all words in negative class
    
    #get the negative prior
    neg_prior = 1 - pos_prior
 

    # get the conditional probabilities
    pos_cond_probs = {}
    neg_cond_probs = {}
    
    for doc in train_set:
        for word in doc:
            if word in doc:
                pos_cond_probs[word] = (pos_counts[word] + laplace) / (pos_total + laplace * (len(vocab)+1))
                neg_cond_probs[word] = (neg_counts[word] + laplace) / (neg_total + laplace * (len(vocab)+1))
            else:
                pos_cond_probs[word] = laplace / (pos_total + laplace * (len(vocab)+1))
                neg_cond_probs[word] = laplace / (neg_total + laplace * (len(vocab)+1))

    # predict the labels
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(neg_prior)
        for word in doc:
            if word in vocab:
                pos_prob += math.log(pos_cond_probs[word])
                neg_prob += math.log(neg_cond_probs[word])
            else:
                pos_prob += math.log(laplace / (pos_total + laplace * (len(vocab)+1)))
                neg_prob += math.log(laplace / (neg_total + laplace * (len(vocab)+1)))
        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats