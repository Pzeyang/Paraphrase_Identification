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
import zss
import StanfordDependencies
from bllipparser import RerankingParser, Tree

def combinations(n, k):
    f = math.factorial
    return f(n) / (f(k) * f(n - k))

def getChildren(node):
    return node.subtrees()

def getLabel(node):
    return node.label

def labelDist(label1, label2):
    if label1 == label2:
        return 0
    else:
        return 1

def dependencyRelations(tree):
    rez = []
    depend = sdc.convert_tree(str(tree))
    for tok in depend:
        if tok.head > 0:
            rez.append((depend[tok.head-1].form, tok.form))
    return rez

rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2')
sdc = StanfordDependencies.get_instance(backend='subprocess')

# Comprising sage advice from:
# http://www.kozareva.com/papers/fintalKozareva.pdf
# http://web.science.mq.edu.au/~rdale/publications/papers/2006/swan-final.pdf
def computeSimple(sentence1, sentence2):
    features = [0] * 10
    tokenizer = RegexpTokenizer(r'\w+')
    words1 = tokenizer.tokenize(sentence1)
    words2 = tokenizer.tokenize(sentence2)
    n = len(words1)
    m = len(words2)

    # word overlap - 'unigram'
    count = 0 # num of same words in sentence
    for word1 in words1:
        if word1 in words2:
            count += 1
    # A symmetric measure is worse for some reason
    features[0] = count / n # "precision"
    features[1] = count / m # "recall"

    # Obtain pairs of adjacent words - 'bigram' overlap
    skipgrams1 = skipgrams(words1, 2, 0)
    skipgrams2 = skipgrams(words2, 2, 0)
    sk2 = [] #skipgrams returns iterator, which is consumed after one traversal
    for g in skipgrams2:
        sk2.append(g)

    count = 0
    for gram1 in skipgrams1:
        if gram1 in sk2:
            count += 1

    features[2] = count / (n - 1)
    features[3] = count / (m - 1)

    # BLEU recall and precision
    features[4] = sentence_bleu([sentence1], sentence2)
    features[5] = sentence_bleu([sentence2], sentence1)

    # Difference of sentence length
    if (n > m):
        features[6] = m / n
    else:
        features[6] = n / m

    # Create semantic tree, compute similarity
    T1 = Tree(rrp.simple_parse(sentence1))
    T2 = Tree(rrp.simple_parse(sentence2))
    R1 = dependencyRelations(T1)
    R2 = dependencyRelations(T2)
    count = 0
    for rel1 in R1:
        for rel2 in R2:
            if rel1 == rel2:
                count += 1
    features[7] = count / len(R1)
    features[8] = count / len(R2)
    features[9] = zss.simple_distance(T1, T2, getChildren, getLabel, labelDist)

    return features

"""# Uses treetagger-python (Installation https://github.com/miotto/treetagger-python ; http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
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
    features = [0] * 7

    # Maybe: Word2Vec and Bipartite

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

    # Feature: Word2Vec (Nouns+Verbs)
    meaning1 = np.zeros(model.vectors.shape[1])
    for word in tags1:
        if word[2] in model:
            meaning1 += model[word[2]]

    meaning2 = np.zeros(model.vectors.shape[1])
    for word in tags2:
        if word[2] in model:
            meaning2 += model[word[2]]

    features[6] = np.linalg.norm(meaning1 - meaning2)
    #features[1] = np.linalg.norm(meaning1 + meaning2)

    return features
"""
