from __future__ import division
import nltk
from nltk.util import ngrams
from nltk.util import skipgrams
import math
import numpy

def LCSLength (sen1, sen2):
    n = len(sen1)
    m = len(sen2)
    aux = numpy.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            if sen1[i] == sen2[j]:
                aux[i][j] = aux[i-1][j-1] + 1
            else:
                aux[i][j] = max(aux[i][j-1], aux[i-1][j])
    return aux[n-1][m-1]

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def learnAll(corpus):
    return learnSVM(corpus), learnKNN(corpus), learnMaxEnt(corpus)

def learnSVM(corpus):
    pass

def learnKNN(corpus):
    pass

def learnMaxEnt(corpus):
    pass


def computeWordOverlapFeatures(sentence1, sentence2):
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    count = 0
    for word1 in tokens1:
        for word2 in tokens2:
            if word1 == word2:
                count += 1

    wordOverlapRatio = (count*2) / (len(tokens1) + len(tokens2))

    trigrams1 = ngrams(tokens1, 3)
    trigrams2 = ngrams(tokens2, 3)

    lcsLen = LCSLength(sentence1, sentence2)

    skipgrams1 = skipgrams(tokens1, 2, len(tokens1))
    skipgrams2 = skipgrams(tokens2, 2, len(tokens2))

    count = 0
    for gram1 in skipgrams1:
        for gram2 in skipgrams2:
            if gram1 == gram2:
                count += 1

    skipgramT1 = count / nCr(len(tokens1), count)
    skipgramT2 = count / nCr(len(tokens2), count)


    return wordOverlapRatio, skipgramT1, skipgramT2, lcsLen

def computeTextSimilarityFeatures(sentence1, sentence2):
    pass

def predict(sentence1, sentence2):
    return 0

print(computeWordOverlapFeatures("Hello my name is Robert", "Hello Sir, wonderful weather, my name is to your information robert"))
