from __future__ import division
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import pickle

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz "
n = len(alphabet)
S = 221 # maximum length of sentences

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_dim=2*n))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

def sentenceToVec (sentence):
    sentence = sentence.lower()
    rez = []
    for char in sentence:
        i = alphabet.find(char)
        if i >= 0:
            v = [0] * n
            v[i] = 1
            rez.append(v)
    return rez

def transfSentences (sent1, sent2):
    v1 = sentenceToVec(sent1)
    v2 = sentenceToVec(sent2)
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1,S):
        v1.append([0]*n)
    for i in range(n2,S):
        v2.append([0]*n)
    for i in range(0,S):
        v1[i].extend(v2[i])
    return v1

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
        v = transfSentences(tokens[3], tokens[4])
        trainFeat.append(v)

    f.close()

    f = open("msr_paraphrase_test.txt", "r")
    f.readline() # ignore header
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])
        v = transfSentences(tokens[3], tokens[4])
        testFeat.append(v)

    f.close()
    return trainFeat, trainClass, testFeat, testClass

#trainFeat, trainClass, testFeat, testClass = readData()
#pickle.dump(trainFeat, open('trainFeat', 'wb'))
#pickle.dump(trainClass, open('trainClass', 'wb'))
#pickle.dump(testFeat, open('testFeat', 'wb'))
#pickle.dump(testClass, open('testClass', 'wb'))
trainFeat = pickle.load(open('trainFeat', 'rb'))
trainClass = pickle.load(open('trainClass', 'rb'))
testFeat = pickle.load(open('testFeat', 'rb'))
testClass = pickle.load(open('testClass', 'rb'))

model.fit(trainFeat, trainClass, nb_epoch=1)
predictedClass = model.predict(testFeat)

count = 0
for i in range(0,1725):
    if predictedClass[i] > 0.5:
        predictedClass[i] = 1
    else:
        predictedClass[i] = 0
    if testClass[i] == predictedClass[i]:
        count += 1
print(count/1725)
