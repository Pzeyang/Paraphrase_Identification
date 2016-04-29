from __future__ import division
import pickle
import numpy
import math
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import NMF

# sentences is an array of tokenized sentences (matrix of words, basically)
# K is the number of distributional features we'll have at the end
def distribFeat(sentences, K):
    paraphraseMap = pickle.load(open("paraphraseMap", "rb"))
    notParaphrMap = pickle.load(open("notParaphrMap", "rb"))

    n = len(sentences)
    uniqWords = []
    for s in sentences:
        for word in s:
            if word not in uniqWords:
                uniqWords.append(word)
    
    # M will hold TF-KLD score for each word for each sentence
    M = numpy.zeros((len(uniqWords), n))
    for word in uniqWords:
        if word in paraphraseMap:
            if word in notParaphrMap:
                p = paraphraseMap[word]
                np = 1 - p
                q = notParaphrMap[word]
                nq = 1 - q
                kl = p * math.log(p/q) + np * math.log(np/nq)
            else:
                kl = 1
        else:
            kl = 0
        for i in range(0,n):
            if word in sentences[i]:
                M[uniqWords.index(word)][i] += kl

    # Step 2: Matrix factorization
    factory = NMF(n_components = 100)
    W = factory.fit_transform(M)
    print(W.shape)

    #Step 3: obtain feature set for paraphrase pair
    for i in range(0, int(n/2)):
        features = [0] * (K * 2)
        for j in range(0, K):
            features[j] = W[i * 2][j] + W[i * 2 + 1][j]
            features[j * 2] = abs(W[i * 2][j] - W[i * 2 + 1][j])

    return features

def getData():
    tokenizer = RegexpTokenizer(r'\w+')
    f = open("msr_paraphrase_train.txt", "r")
    f.readline()
    sentences = []
    trainClass = [0] * 4076
    for i in range(0,4076):
        tokens = f.readline().strip().split('\t')
        trainClass = int(tokens[0])
        sentences.append(tokenizer.tokenize(tokens[3].lower()))
        sentences.append(tokenizer.tokenize(tokens[4].lower()))

    f.close()
    trainFeat = distribFeat(sentences, 100)

    f = open("msr_paraphrase_test.txt", "r")
    f.readline()
    sentences = []
    testClass = [0] * 1725
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass = int(tokens[0])
        sentences.append(tokenizer.tokenize(tokens[3].lower()))
        sentences.append(tokenizer.tokenize(tokens[4].lower()))

    f.close()
    testFeat = distribFeat(sentences, 100)
    return trainFeat, trainClass, testFeat, testClass
