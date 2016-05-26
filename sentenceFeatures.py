from __future__ import division
from itertools import product
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import skipgrams
from nltk.corpus import wordnet
from treetagger import TreeTagger
import numpy as np
import text2int as t2i
import copy
import sys
import pickle
import word2vec
import math
import os
from munkres import Munkres
import scipy.spatial

def combinations(n, k):
    f = math.factorial
    return f(n) / (f(k) * f(n - k))

# Comprising sage advice from:
# http://www.kozareva.com/papers/fintalKozareva.pdf
# http://web.science.mq.edu.au/~rdale/publications/papers/2006/swan-final.pdf
def computeSimple(sentence1, sentence2):
    features = [0] * 7
    tokenizer = RegexpTokenizer(r'\w+')
    words1 = tokenizer.tokenize(sentence1)
    words2 = tokenizer.tokenize(sentence2)
    n = len(words1)
    m = len(words2)

    # word overlap features
    count = 0 # num of same words in sentence
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                count += 1

    features[0] = count / n # "precision"
    features[1] = count / m # "recall"

    features[2] = sentence_bleu([sentence1], sentence2)
    features[3] = sentence_bleu([sentence2], sentence1)

    # Obtain pairs of adjacent words
    skipgrams1 = skipgrams(words1, 2, 0)
    skipgrams2 = skipgrams(words2, 2, 0)

    count = 0
    for gram1 in skipgrams1:
        for gram2 in skipgrams2:
            if gram1 == gram2:
                count += 1

    features[4] = count / combinations(n, count)
    features[5] = count / combinations(m, count)


    """if (n > m):
        features[6] = m / n
    else:
        features[6] = n / m"""

    if len(sentence1) > len(sentence2):
        features[7] = len(sentence2) / len(sentence1)
    else:
        features[7] = len(sentence1) / len(sentence2)

    return features

# Uses treetagger-python (Installation https://github.com/miotto/treetagger-python ; http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
try:
    semanticsimilarity_lookuptable = pickle.load(open('semanticsimilarity_lookuptable.pkl', 'rb'))
except Exception:
    semanticsimilarity_lookuptable = {}

print "Build Word2Vec Corpus"
dir = os.path.dirname(os.path.abspath(__file__))
try:
    # on OSX for some reason this does not work
    word2vec.word2phrase(dir + '/text8', dir + '/text8-phrases', verbose=True)
    word2vec.word2vec(dir + '/text8-phrases', dir + '/text8.bin', size=100, verbose=True)
except Exception as e:
    print e

model = word2vec.load(dir + '/text8.bin')
print "Finish"

def computeSemantics(sentence1, sentence2):
def computeSemanticSimilarityFeatures(sentence1, sentence2):
    features = [0] * 9

    if (sentence1 + sentence2) not in semanticsimilarity_lookuptable:
        def prepareSentence(sentence):
            return sentence.replace('-', ' ').replace('$', ' ')

        tt = TreeTagger(language='english')
        tags1 = [a for a in tt.tag(prepareSentence(sentence1)) if len(a) > 1]
        tags2 = [a for a in tt.tag(prepareSentence(sentence2)) if len(a) > 1]

        semanticsimilarity_lookuptable[sentence1 + sentence2] = [tags1, tags2]

    tags1 = copy.deepcopy(semanticsimilarity_lookuptable[sentence1 + sentence2][0])
    tags2 = copy.deepcopy(semanticsimilarity_lookuptable[sentence1 + sentence2][1])

    # Feature: noun/web semantic similarity
    # Get Synonym set
    def synSet(tags):
        for word in tags:
            # Only compare Nouns or Verbs
            # Python does not have short circuit operators, wtf?!
            if (word[1][0] != 'N' if len(word[1]) >= 1 else 1) and (word[1][:2] != 'VV' if len(word[1]) >= 2 else 1):
                continue

            word.append(wordnet.synsets(word[2]))

    synSet(tags=tags1)
    synSet(tags=tags2)

    simsMaxNoun = []
    simsAvgNoun = []
    simsMaxVerb = []
    simsAvgVerb = []

    for word1, word2 in product(tags1, tags2):
        type1 = word1[1]
        type2 = word2[1]

        if (type1[0] != 'N' and type1[:2] != 'VV') or type1 != type2:
            continue

        similarityMax = 0
        similarityAvg = 0
        if word1[2] == word2[2]:
            similarityAvg = 1
            similarityMax = 1
        else:
            for sense1, sense2 in product(word1[3], word2[3]):
                sim = wordnet.wup_similarity(sense1, sense2)
                similarityMax = max(similarityMax, sim)
                similarityAvg += sim if sim is not None else 0

        if type1[0] != 'N':
            simsMaxNoun.append(similarityMax)
            simsAvgNoun.append(similarityAvg / (len(word1[3]) + len(word2[3])) if len(word1[3]) + len(word2[3]) > 0 else 0)
        else:
            simsMaxVerb.append(similarityMax)
            simsAvgVerb.append(similarityAvg / (len(word1[3]) + len(word2[3])) if len(word1[3]) + len(word2[3]) > 0 else 0)


    features[0] = np.sum(simsMaxNoun) / len(simsMaxNoun) if len(simsMaxNoun) > 0 else 0
    features[1] = np.sum(simsAvgNoun) / len(simsAvgNoun) if len(simsAvgNoun) > 0 else 0

    features[2] = np.sum(simsMaxVerb) / len(simsMaxVerb) if len(simsMaxVerb) > 0 else 0
    features[3] = np.sum(simsAvgVerb) / len(simsAvgVerb) if len(simsAvgVerb) > 0 else 0

    # Feature: Cardinal number similarity
    def findCardinals(tags):
        cardinals = []
        for index, word1 in enumerate(tags):
            if word1[1] == 'CD':
                # is "more", "over" or "above" before?
                before = [a[0] for a in tags[max(index-2, 0):index]]

                try:
                    val = float(word1[0])
                except ValueError:
                    val = t2i.text2int(word1[0])

                maxValue = minValue = val

                if ("more" in before) or ("over" in before) or ("above" in before) or ("greater" in before):
                    maxValue = sys.maxint
                    minValue += 1
                elif ("less" in before) or ("under" in before) or ("below" in before) or ("smaller" in before):
                    minValue = -sys.maxint - 1
                    maxValue -= 1

                cardinals.append([minValue, maxValue])
        return cardinals

    cardinals1 = findCardinals(tags=tags1)
    cardinals2 = findCardinals(tags=tags2)

    def countCDMatches(cardinals1, cardinals2):
        count = 0
        for cd1 in cardinals1:
            for cd2 in cardinals2:
                if cd1[0] == cd2[0] and cd1[1] == cd2[1]:
                    count += 1
                    break
        return count

    features[4] = (countCDMatches(cardinals1, cardinals2) + countCDMatches(cardinals2, cardinals1)) / (len(cardinals1) + len(cardinals2)) if len(cardinals1) + len(cardinals2) > 0 else 1
    #features[2] = countCDMatches(cardinals1, cardinals2) / len(cardinals1) if len(cardinals1) > 0 else 1
    #features[3] = countCDMatches(cardinals2, cardinals1) / len(cardinals2) if len(cardinals2) > 0 else 1


    # Feature: Proper Name
    def findProperNouns(tags):
        nouns = []
        for word in tags:
            if word[1] == 'NPS':
                nouns.append(word[0])
        return nouns

    def countNounMatches(nouns1, nouns2):
        count = 0
        for noun1 in nouns1:
            for noun2 in nouns2:
                if noun1 == noun2:
                    count += 1
                    break
        return count

    nouns1 = findProperNouns(tags1)
    nouns2 = findProperNouns(tags2)

    features[5] = (countNounMatches(nouns1, nouns2) + countNounMatches(nouns2, nouns1)) / (len(nouns1) + len(nouns2)) if len(nouns1) + len(nouns2) > 0 else 1
    # features[4] = countNounMatches(nouns1, nouns2) / len(nouns1) if len(nouns1) > 0 else 1
    # features[5] = countNounMatches(nouns2, nouns1) / len(nouns2) if len(nouns2) > 0 else 1

    # Feature: Word2Vec (all)
    meaning1 = np.zeros(model.vectors.shape[1])
    for word in tags1:
        if word[2] in model:
            meaning1 += model[word[2]]

    meaning2 = np.zeros(model.vectors.shape[1])
    for word in tags2:
        if word[2] in model:
            meaning2 += model[word[2]]

    diffMeaning = meaning1 - meaning2
    features[6] = np.linalg.norm(diffMeaning)
    features[7] = scipy.spatial.distance.cosine(meaning1, meaning2)

    similarityMatrix = [0] * len(tags1)
    for index1, word1 in enumerate(tags1):
        row = [0]*len(tags2)
        for index2, word2 in enumerate(tags2):
            similarityMax = 0
            if len(word1) > 3 and len(word2) > 3:
                for sense1, sense2 in product(word1[3], word2[3]):
                    sim = wordnet.wup_similarity(sense1, sense2)
                    similarityMax = max(similarityMax, sim)
                similarityMax = 1 - similarityMax
            else:
                similarityMax = 1

            row[index2] = similarityMax
        similarityMatrix[index1] = row
    m = Munkres()
    totalCost = 0
    indices = m.compute(similarityMatrix)
    for row, column in indices:
        totalCost += similarityMatrix[row][column]

    features[8] = totalCost / len(indices)

    return features

