"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # get word-tag frequencies from the training data
    wordTagFreq = defaultdict(lambda: defaultdict(int))
    tagFreq = defaultdict(int)

    for stnce in train:
        for word, tag in stnce:
            wordTagFreq[word][tag] += 1
            tagFreq[tag] += 1

    # assign the most frequent tag from training
    tagged_test = []

    for stnce in test:
        tagged_stnce = []
        for word in stnce:
            if word in wordTagFreq:
                # get tag with the highest frequency for this word
                maxFreqTag = max(wordTagFreq[word], key=wordTagFreq[word].get)
                tagged_stnce.append((word, maxFreqTag))
            else:
                # handle unseen words by assigning the most frequent overall tag
                maxFreqTag_overall = max(tagFreq, key=tagFreq.get)
                tagged_stnce.append((word, maxFreqTag_overall))
        tagged_test.append(tagged_stnce)

    return tagged_test
   
