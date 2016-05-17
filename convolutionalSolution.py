from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from nltk.tokenize import RegexpTokenizer
import os
import word2vec

w2v = 100 # size of word2vec embedding, as recommended in paper

print "Build Word2Vec Corpus"
dir = os.path.dirname(os.path.abspath(__file__))
"""word2vec.word2phrase(dir + '/text8', dir + '/text8-phrases', verbose=True)
try:
    # on OSX for some reason this does not work
    word2vec.word2phrase(dir + '/text8', dir + '/text8-phrases', verbose=True)
    word2vec.word2vec(dir + '/text8-phrases', dir + '/text8.bin', size=300, verbose=True)
except Exception as e:
    print e
    print "Failed to build corpus"
"""
w2vModel = word2vec.load(dir + '/text8.bin')
print "Finish"

# w is filter word width
# s is maximum nr of words in sentence
def learn(trainFeat, trainClass, testFeat, w=3, s=70, nrFilters=50):
    model = Sequential()
    model.add(Convolution2D(nrFilters, w2v, w, input_shape=(1, w2v, s)))
    print(model.output_shape)
    model.add(AveragePooling2D((2,2)))
    print(model.output_shape)
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd')

    model.fit(trainFeat, trainClass, nb_epoch=50)
    return model.predict_classes(testFeat)

def sentenceToMatrix(S):
    Smatrix = []
    for word in S:
        if word in w2vModel:
            Smatrix.append(w2vModel[word])
        else:
            Smatrix.append([1]*w2v)
    while len(Smatrix) < 70:
        Smatrix.append([0]*w2v)
    return Smatrix

def getData():
    tokenizer = RegexpTokenizer(r'\w+')
    f = open("msr_paraphrase_train.txt", "r")
    f.readline()
    trainInput = []
    trainClass = [0] * 4076
    for i in range(0,4076):
        tokens = f.readline().strip().split('\t')
        trainClass[i] = int(tokens[0])
        S = tokenizer.tokenize(tokens[3].lower())
        Smatrix = sentenceToMatrix(S)
        S = tokenizer.tokenize(tokens[4].lower())
        Smatrix.extend(sentenceToMatrix(S))
        trainInput.append(Smatrix)

    f.close()

    f = open("msr_paraphrase_test.txt", "r")
    f.readline()
    testInput = []
    testClass = [0] * 1725
    for i in range(0,1725):
        tokens = f.readline().strip().split('\t')
        testClass[i] = int(tokens[0])
        S = tokenizer.tokenize(tokens[3].lower())
        Smatrix = sentenceToMatrix(S)
        S = tokenizer.tokenize(tokens[4].lower())
        Smatrix.extend(sentenceToMatrix(S))
        testInput.append(Smatrix)

    f.close()
    return trainInput, trainClass, testInput, testClass

trainIn, trainClass, testIn, testClass = getData()
print("data is built")
predictedClass = learn(trainIn, trainClass, testIn)
print("done predicting")
count = 0
for i in range(0,1725):
    if predictedClass[i] >= 0.5:
        predictedClass[i] = 1
    else:
        predictedClass[i] = 0
    if testClass[i] == predictedClass[i]:
        count += 1
print(count/1725)
