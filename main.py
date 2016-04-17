from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import skipgrams
from nltk.corpus import wordnet
import math
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from treetagger import TreeTagger
from itertools import product
import numpy as np
import sys
import pickle
import text2int as t2i

def combinations(n, k):
    f = math.factorial
    return f(n) / (f(k) * f(n - k))

# Majority voting to determine class
def learnAll(trainFeat, trainClass, testFeat):
    vote1 = learnSVM(trainFeat, trainClass, testFeat)
    vote2 = learnKNN(trainFeat, trainClass, testFeat)
    vote3 = learnMaxEnt(trainFeat, trainClass, testFeat)
    res = [0] * len(testFeat)
    for i in range(0, len(testFeat)):
        if vote1[i] + vote2[i] + vote3[i] > 1:
            res[i] = 1
    return res

def learnSVM(trainFeat, trainClass, testFeat):
    model = svm.SVC()
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

def learnKNN(trainFeat, trainClass, testFeat):
    model = neighbors.KNeighborsClassifier(15, weights='distance')
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

def learnMaxEnt(trainFeat, trainClass, testFeat):
    model = linear_model.LogisticRegression(n_jobs=-1)
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

# Comprising sage advice from:
# http://www.kozareva.com/papers/fintalKozareva.pdf
# http://web.science.mq.edu.au/~rdale/publications/papers/2006/swan-final.pdf
def computeSentenceSimilarityFeatures(sentence1, sentence2):
    features = [0] * 6
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

    # From arrays of words, pairs of two words, separated by len at most 4
    skipgrams1 = skipgrams(words1, 2, 4)
    skipgrams2 = skipgrams(words2, 2, 4)

    count = 0
    for gram1 in skipgrams1:
        for gram2 in skipgrams2:
            if gram1 == gram2:
                count += 1

    features[4] = count / combinations(n, count)
    features[5] = count / combinations(m, count)

    return features


# Uses treetagger-python (Installation https://github.com/miotto/treetagger-python ; http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
try:
    semanticsimilarity_lookuptable = pickle.load(open('semanticsimilarity_lookuptable.pkl', 'rb'))
except Exception:
    semanticsimilarity_lookuptable = {}

def computeSemanticSimilarityFeatures(sentence1, sentence2):
    features = [0] * 3

    if (sentence1 + sentence2) in semanticsimilarity_lookuptable:
        features = semanticsimilarity_lookuptable[sentence1 + sentence2]
    else:
        def prepareSentence(sentence):
            return sentence.replace('-', ' ').replace('$', ' ')

        tt = TreeTagger(language='english')
        tags1 = [a for a in tt.tag(prepareSentence(sentence1)) if len(a) > 1]
        tags2 = [a for a in tt.tag(prepareSentence(sentence2)) if len(a) > 1]
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

        sims = []

        # TODO: Maybe do not use product() instead use max() similarity
        for word1, word2 in product(tags1, tags2):
            type1 = word1[1]
            type2 = word2[1]

            if type1[0] != 'N' and type1[:2] != 'VV' or type1 != type2:
                continue

            similarity = 0
            for sense1, sense2 in product(word1[3], word2[3]):
                similarity = max(similarity, wordnet.wup_similarity(sense1, sense2))

            sims.append(similarity)


        features[0] = np.sum(sims) / len(sims) if len(sims) > 0 else 0

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

        features[1] = (countCDMatches(cardinals1, cardinals2) + countCDMatches(cardinals2, cardinals1)) / (len(cardinals1) + len(cardinals2)) if len(cardinals1) + len(cardinals2) > 0 else 1

        # Feature: Proper Name
        def findProperNouns(tags):
            nouns = []
            for word in tags:
                if word[1] == 'NP' or word[1] == 'NPS':
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

        features[2] = (countNounMatches(nouns1, nouns2) + countNounMatches(nouns2, nouns1)) / (len(nouns1) + len(nouns2)) if len(nouns1) + len(nouns2) > 0 else 1

        semanticsimilarity_lookuptable[sentence1 + sentence2] = features

    return features


# Read and process train and test data
def readData():
    trainFeat = []
    trainClass = [0] * 4076
    testFeat = []
    testClass = [0] * 1725

    f = open("msr_paraphrase_train.txt", "r")
    f.readline() # ignore header
    for i in range(0,4076):
        tokens = f.readline().strip().split('\t')
        trainClass[i] = int(tokens[0])
        print "Training ", i, 4076, i / 4076
        features = computeSentenceSimilarityFeatures(tokens[3], tokens[4])
        features.extend(computeSemanticSimilarityFeatures(tokens[3], tokens[4]))

        trainFeat.append(features)

        #if i % 10 == 9:
        #    print("Dump Similarity Table")
        #   pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))



    f = open("msr_paraphrase_test.txt", "r")
    f.readline() # ignore header
    for i in range(0,1725):
        print "Testing ", i, 4076, i / 4076

        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])

        features = computeSentenceSimilarityFeatures(tokens[3], tokens[4])
        features.extend(computeSemanticSimilarityFeatures(tokens[3], tokens[4]))

        testFeat.append(features)

        #if i % 10 == 9:
        #print("Dump Similarity Table")
        #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))

    print "Finished"
    return trainFeat, trainClass, testFeat, testClass

trainFeat, trainClass, testFeat, testClass = readData()
predictedClass = learnAll(trainFeat, trainClass, testFeat)

count = 0
for i in range(0,1725):
    if testClass[i] == predictedClass[i]:
        count += 1
print(count/1725)
