import nltk
import math


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

    trigrams1 = nltk.util.trigrams(tokens1)
    trigrams2 = nltk.util.trigrams(tokens2)

    count = 0
    #TODO: find maximum "substring"
    for gram1 in trigrams1:
        pass


    skipgrams1 = nltk.util.skipgrams(tokens1, 2, len(tokens1))
    skipgrams2 = nltk.util.skipgrams(tokens2, 2, len(tokens2))

    count = 0
    for gram1 in skipgrams1:
        for gram2 in skipgrams2:
            if gram1 == gram2:
                count += 1

    skipgramT1 = count / nCr(len(tokens1))
    skipgramT2 = count / nCr(len(tokens2))



    return wordOverlapRatio, skipgramT1 / skipgramT2

def computeTextSimilarityFeatures(sentence1, sentence2):
    pass

def predict(sentence1, sentence2):
    return 0

print(computeWordOverlapFeatures("Hello my name is Robert", "Hello Sir, wonderful weather, my name is to your information robert"))