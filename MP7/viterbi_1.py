"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

def training(sentences):
    """
    Computes initial tags, emission words, and transition tag-to-tag probabilities.
    :param sentences: List of sentences with word-tag pairs.
    :param epsilon_for_pt: Smoothing parameter for Laplace smoothing.
    :return: Initial tag probabilities, emission probabilities, transition probabilities.
    """
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    
    tag_counts = defaultdict(lambda: 0)
    tag_pair_counts = defaultdict(lambda: defaultdict(lambda: 0))
    tag_word_counts = defaultdict(lambda: defaultdict(lambda: 0))
    
    # Count occurrences of tags, tag pairs, and tag/word pairs
    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            tag_counts[tag] += 1
            if prev_tag:
                tag_pair_counts[prev_tag][tag] += 1
    
            tag_word_counts[tag][word] += 1
            tag_word_counts[tag]["UNKNOWN"] += 1
            prev_tag = tag
    
    # Compute initial tag probabilities
    total_sentences = len(sentences)
    for tag, count in tag_counts.items():
        init_prob[tag] = count / total_sentences
    
    # Compute emission probabilities with reduced Laplace smoothing for unseen words
    for tag in tag_counts:
        total_tag_words = len(tag_word_counts[tag])
        total_tag_pairs = len(tag_pair_counts[tag])
        
        for word, count in tag_word_counts[tag].items():
            # Modify the emission probability calculation to reduce the impact of unseen words
            if word =="UNKNOWN":
                emit_prob[tag][word] = (epsilon_for_pt) / (tag_counts[tag] + epsilon_for_pt * (total_tag_words + 1))
            else:
                emit_prob[tag][word] = (count + epsilon_for_pt) / (tag_counts[tag] + epsilon_for_pt * (total_tag_words + 1))
         
        # Compute transition probabilities with Laplace smoothing
        for next_tag in tag_counts:
            trans_prob[tag][next_tag] = (tag_pair_counts[tag][next_tag] + epsilon_for_pt) / (tag_counts[tag] + epsilon_for_pt * (total_tag_pairs + 1))

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    log_prob = {}
    predict_tag_seq = {}
    
    if i == 0:
        # Special case for the first word in the sentence
        for tag in emit_prob:
            log_prob[tag] = prev_prob[tag] + log(emit_prob[tag].get(word, emit_epsilon))
            predict_tag_seq[tag] = [tag]
    else:
        # General case for subsequent words
        for tag in emit_prob:
            max_log_prob = float('-inf')
            best_prev_tag = None
            
            for prev_tag in prev_prob:
                prev_log_prob = prev_prob[prev_tag]
                transition_prob = trans_prob[prev_tag][tag]
                
                if emit_prob[tag].get(word):
                    x = log(emit_prob[tag].get(word))
                else:
                    x = log(emit_prob[tag].get("UNKNOWN"))

                current_log_prob = prev_log_prob + log(transition_prob) + x
                
                if current_log_prob > max_log_prob:
                    max_log_prob = current_log_prob
                    best_prev_tag = prev_tag
            
            log_prob[tag] = max_log_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[best_prev_tag] + [tag]
    
    return log_prob, predict_tag_seq



def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # DONE:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        # Find the tag sequence with the maximum log probability
        best_final_tag = max(log_prob, key=lambda k: log_prob[k])
        best_tag_sequence = predict_tag_seq[best_final_tag]

        predicted_sentence = [(word, tag) for word, tag in zip(sentence, best_tag_sequence)]
        predicts.append(predicted_sentence)

    return predicts
