from __future__ import division
import math
import pickle
import sentenceFeatures
import distribFeat
import learnModels

# Read and process train and test data
def readData():
    trainFeat = []
    trainClass = [0] * 4076
    testFeat = []
    testClass = [0] * 1725

    print "Training Phase"

    f = open("msr_paraphrase_train.txt", "r")
    f.readline() # ignore header
    for i in range(0,4076):
        tokens = f.readline().strip().split('\t')
        trainClass[i] = int(tokens[0])
        features = sentenceFeatures.computeSimple(tokens[3], tokens[4])
        features.extend(sentenceFeatures.computeSemantics(tokens[3], tokens[4]))

        trainFeat.append(features)

        #if i % 10 == 9:
            #print("Dump Similarity Table")
            #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))


    print "Testing phase"

    f = open("msr_paraphrase_test.txt", "r")
    f.readline() # ignore header
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])

        features = sentenceFeatures.computeSimple(tokens[3], tokens[4])
        features.extend(sentenceFeatures.computeSemantics(tokens[3], tokens[4]))

        testFeat.append(features)

        #if i % 10 == 9:
            #print("Dump Similarity Table")
            #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))
    f.close()

    #pickle.dump(semanticsimilarity_lookuptable, open('semanticsimilarity_lookuptable.pkl', 'wb'))

    return trainFeat, trainClass, testFeat, testClass

#trainFeat, trainClass, testFeat, testClass = distribFeat.getData()
trainFeat, trainClass, testFeat, testClass = readData()
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
    if testClass[i] == predictedClass[i]:
        count += 1
print(count/1725)
