from __future__ import division
import pickle
import numpy
import math
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import NMF, TruncatedSVD
import sentenceFeatures

# sentences is an array of tokenized sentences (matrix of words, basically)
# K is the number of distributional features we'll have at the end
def distribFeat(fullSent, sentences, K):
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
    factory = TruncatedSVD(n_components = K)
    #factory = NMF(n_components = K)
    factory.fit_transform(M) # M = W*H , returns W, which we don't need
    H = factory.components_ # should be size K * n

    #Step 3: obtain feature set for paraphrase pair
    features = []
    i = 0
    while i < n:
        feat = [0] * (K * 2)
        for j in range(0, K):
            feat[j] = H[j][i] + H[j][i + 1]
            feat[j * 2] = abs(H[j][i] - H[j][i + 1])
            if feat[j] > 0.1:
                print(str(feat[j])+" "+str(feat[j*2]))
        #feat.extend(sentenceFeatures.compute(fullSent[i],fullSent[i+1]))
        i += 2 # step to next pair of sentences
        features.append(feat)

    return features

def getData():
    tokenizer = RegexpTokenizer(r'\w+')
    f = open("msr_paraphrase_train.txt", "r")
    f.readline()
    sentences = []
    sentencesWords = []
    trainClass = [0] * 4076
    for i in range(0,4076):
        tokens = f.readline().strip().split('\t')
        trainClass[i] = int(tokens[0])
        sentences.append(tokens[3].lower())
        sentences.append(tokens[4].lower())
        sentencesWords.append(tokenizer.tokenize(tokens[3].lower()))
        sentencesWords.append(tokenizer.tokenize(tokens[4].lower()))

    f.close()
    trainFeat = distribFeat(sentences, sentencesWords, 200)

    f = open("msr_paraphrase_test.txt", "r")
    f.readline()
    sentences = []
    sentencesWords = []
    testClass = [0] * 1725
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])
        sentences.append(tokens[3].lower())
        sentences.append(tokens[4].lower())
        sentencesWords.append(tokenizer.tokenize(tokens[3].lower()))
        sentencesWords.append(tokenizer.tokenize(tokens[4].lower()))

    f.close()
    testFeat = distribFeat(sentences, sentencesWords, 200)
    return trainFeat, trainClass, testFeat, testClass
