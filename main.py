from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import skipgrams
from nltk.corpus import wordnet
import math
from treetagger import TreeTagger
from itertools import product
import numpy as np
import sys
import pickle
import text2int as t2i
import copy
import distribFeat
import learnModels

# Uses treetagger-python (Installation https://github.com/miotto/treetagger-python ; http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
#try:
#    semanticsimilarity_lookuptable = pickle.load(open('semanticsimilarity_lookuptable.pkl', 'rb'))
#except Exception:
#    semanticsimilarity_lookuptable = {}

def computeSemanticSimilarityFeatures(sentence1, sentence2):
    features = [0] * 6

    # Maybe: Word2Vec

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

    simsMax = []
    simsAvg = []

    for word1, word2 in product(tags1, tags2):
        type1 = word1[1]
        type2 = word2[1]

        if type1[0] != 'N' and type1[:2] != 'VV' or type1 != type2:
            continue

        similarityMax = 0
        similarityAvg = 0
        for sense1, sense2 in product(word1[3], word2[3]):
            sim = wordnet.wup_similarity(sense1, sense2)
            similarityMax = max(similarityMax, sim)
            similarityAvg += sim if sim is not None else 0

        simsMax.append(similarityMax)
        simsAvg.append(similarityAvg / (len(word1[3]) + len(word2[3])) if len(word1[3]) + len(word2[3]) > 0 else 0)


    features[0] = np.sum(simsMax) / len(simsMax) if len(simsMax) > 0 else 0
    features[1] = np.sum(simsAvg) / len(simsAvg) if len(simsAvg) > 0 else 0

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

    #features[2] = (countCDMatches(cardinals1, cardinals2) + countCDMatches(cardinals2, cardinals1)) / (len(cardinals1) + len(cardinals2)) if len(cardinals1) + len(cardinals2) > 0 else 1
    features[2] = countCDMatches(cardinals1, cardinals2) / len(cardinals1) if len(cardinals1) > 0 else 1
    features[3] = countCDMatches(cardinals2, cardinals1) / len(cardinals2) if len(cardinals2) > 0 else 1


    # Feature: Proper Name
    # TODO: Find Proper names
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

    #features[4] = (countNounMatches(nouns1, nouns2) + countNounMatches(nouns2, nouns1)) / (len(nouns1) + len(nouns2)) if len(nouns1) + len(nouns2) > 0 else 1
    features[4] = countNounMatches(nouns1, nouns2) / len(nouns1) if len(nouns1) > 0 else 1
    features[5] = countNounMatches(nouns2, nouns1) / len(nouns2) if len(nouns2) > 0 else 1

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
        features = computeSentenceSimilarityFeatures(tokens[3], tokens[4])
        #features.extend(computeSemanticSimilarityFeatures(tokens[3], tokens[4]))

        trainFeat.append(features)

        #if i % 10 == 9:
            #print("Dump Similarity Table")
            #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))



    f = open("msr_paraphrase_test.txt", "r")
    f.readline() # ignore header
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])

        features = computeSentenceSimilarityFeatures(tokens[3], tokens[4])
        #features.extend(computeSemanticSimilarityFeatures(tokens[3], tokens[4]))

        testFeat.append(features)

        #if i % 10 == 9:
            #print("Dump Similarity Table")
            #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))

    return trainFeat, trainClass, testFeat, testClass

trainFeat, trainClass, testFeat, testClass = distribFeat.getData()
#trainFeat, trainClass, testFeat, testClass = readData()
#pickle.dump(trainFeat, open('trainFeat', 'wb'))
#pickle.dump(trainClass, open('trainClass', 'wb'))
#pickle.dump(testFeat, open('testFeat', 'wb'))
#pickle.dump(testClass, open('testClass', 'wb'))
#trainFeat = pickle.load(open('trainFeat', 'rb'))
#trainClass = pickle.load(open('trainClass', 'rb'))
#testFeat = pickle.load(open('testFeat', 'rb'))
#testClass = pickle.load(open('testClass', 'rb'))
predictedClass = learnModels.SVM(trainFeat, trainClass, testFeat)

count = 0
for i in range(0,1725):
    if predictedClass[i] > 0 and predictedClass[i] < 1:
        print(predictedClass[i])
    if testClass[i] == predictedClass[i]:
        count += 1
print(count/1725)
