import numpy as np
from collections import Counter
from math import log
from math import inf

# Define a function to create an empty matrix for sentence probabilities
def getMatrix(sentence, tags):
    s_matrix = []
    for i in range(len(sentence)):
        s_matrix.append({tag: 0 for tag in tags})
    return s_matrix

# Define a function to create a backpointer matrix for tracking tag probabilities
def find_s_probabilities(sentence, tags):
    back_ptr = []
    for i in range(len(sentence)):
        back_ptr.append({tag: None for tag in list(tags)})
    return back_ptr

# Train the model using Viterbi algorithm
def train_version_2(sentence, s_matrix, back_ptr, initial_prob, ip_uk, transition_prob, tp_uk, emission_prob, ep_uk, myDict):
    # Calculate initial probabilities for the first word in the sentence
    for key, value in s_matrix[0].items():
        dft_pi = 0
        xvalue = 0
        if key in initial_prob:
            dft_pi = initial_prob[key]
        else:
            dft_pi = ip_uk
        if (sentence[0], key) in emission_prob:
            xvalue = emission_prob[(sentence[0], key)]
        else:
            if myDict == 0:
                xvalue = ep_uk
            else:
                xvalue = ep_uk[key]
        s_matrix[0][key] = dft_pi + xvalue

    # Calculate probabilities for the rest of the sentence and track backpointers
    for i in range(1, len(s_matrix)):
        for k in s_matrix[i].keys():
            max_prob = -inf
            max_key = ""
            xvalue = 0
            if (sentence[i], k) in emission_prob:
                xvalue = emission_prob[(sentence[i], k)]
            else:
                if myDict == 0:
                    xvalue = ep_uk
                else:
                    xvalue = ep_uk[key]
            for n_prime in s_matrix[i - 1].keys():
                aValue = 0
                if (n_prime, k) in transition_prob:
                    aValue = transition_prob[(n_prime, k)]
                else:
                    if myDict == 0:
                        aValue = tp_uk
                    else:
                        aValue = tp_uk[n_prime]
                if (aValue + xvalue + s_matrix[i - 1][n_prime]) > max_prob:
                    max_prob = aValue + xvalue + s_matrix[i - 1][n_prime]
                    max_key = n_prime
            s_matrix[i][k] = max_prob
            back_ptr[i][k] = max_key
    # Determine the most likely tag sequence using backpointers
    index = len(s_matrix) - 1
    key_ = max(s_matrix[index], key=lambda key: s_matrix[index][key])
    return_s = []
    while key_ is not None and index >= 0:
        return_s = [(sentence[index], key_)] + return_s
        key_ = back_ptr[index][key_]
        index -= 1
    return return_s

# Calculate hapax probabilities for words
def hapax(train, tags, alpha):
    wc = dict()
    twc = dict()

    for sentc in train:
        for p in sentc:
            w, t = p
            wc[w] = wc.get(w, 0) + 1

            if not t in twc:
                twc[t] = dict()
            twc[t][w] = twc[t].get(w, 0) + 1

    hapax = list(map(lambda x: x[0], filter(lambda x: x[1] == 1, wc.items())))

    h = {t: (sum(twc[t].get(w, 0) for w in hapax) + alpha) / (len(hapax) + alpha * len(tags)) for t in list(tags)}
    return h

# Main function for Viterbi algorithm with the provided training and test data
def viterbi_3(train, test):
    k = 10**(-4)

    # Create sets of unique words and tags
    tot_pairs = 0
    words = set()
    tags = set()
    for sentc in train:
        for pair in sentc:
            tot_pairs += 1
            w, t = pair
            words.add(w)
            tags.add(t)

    # Calculate initial probabilities for the first tag in a sentence
    init_tag_ct = dict()
    for sentence in train:
        init_tag_ct[sentence[0][1]] = init_tag_ct.get(sentence[0][1], 0) + 1
    initial_prob = {t: log((c + k) / (len(train) + k * len(tags))) for (t, c) in init_tag_ct.items()}
    ip_uk = log(k / (len(train) + k * len(tags)))

    # Calculate transition probabilities
    transition_prob = dict()
    pct = dict()
    for sentc in train:
        for i in range(0, len(sentc)):
            curr_t = sentc[i][1]
            prev_t = sentc[i - 1][1]
            pct[prev_t] = pct.get(prev_t, 0) + 1
            transition_prob[(prev_t, curr_t)] = transition_prob.get((prev_t, curr_t), 0) + 1

    for tag_1 in list(tags):
        for tag in list(tags):
            if (tag_1, tag) in transition_prob:
                transition_prob[(tag_1, tag)] = log((transition_prob[(tag_1, tag)] + k) / (pct[tag_1] + k * (len(tags) + 1)))
            else:
                transition_prob[(tag_1, tag)] = log(k / (tot_pairs + k * len(tags)))

    tp_uk = {tp: log(k / (pct.get(tp, 0) + k * len(tags))) for tp in list(tags)}

    # Calculate hapax probabilities
    hapax_dict = hapax(train, tags, k)

    # Calculate emission probabilities
    emission_prob = Counter()
    tag_ct = Counter()
    for sentc in train:
        for wt_pair in sentc:
            emission_prob[wt_pair] += 1
            tag_ct[wt_pair[1]] += 1
    emission_prob = dict(emission_prob)

    for tag in list(tags):
        for word in list(words):
            if (word, tag) in emission_prob:
                emission_prob[(word, tag)] = log((emission_prob[(word, tag)] + k * hapax_dict[tag]) / (tag_ct[tag] + k * (len(words) + 1) * hapax_dict[tag]))

    ep_uk = {tag: log((k * hapax_dict[tag]) / (tag_ct.get(tag, 0) + k * hapax_dict[tag] * (len(words) + 1))) for tag in list(tags)}

    # Apply Viterbi algorithm to estimate tags for test sentences
    estimated_test = [[] for i in range(len(test))]
    i = 0
    for sentc in test:
        s_matrix = getMatrix(sentc, tags)
        back_ptr = find_s_probabilities(sentc, tags)
        estimated_test[i] = train_version_2(sentc, s_matrix, back_ptr, initial_prob, ip_uk, transition_prob, tp_uk, emission_prob, ep_uk, 1)
        i += 1
    predicts = estimated_test
    return predicts
