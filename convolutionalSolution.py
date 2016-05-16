from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers.convolutional import Convolution2D
import os
import word2vec

w2v = 300 # size of word2vec embedding, as recommended in paper

print "Build Word2Vec Corpus"
dir = os.path.dirname(os.path.abspath(__file__))
try:
    # on OSX for some reason this does not work
    word2vec.word2phrase(dir + '/text8', dir + '/text8-phrases', verbose=True)
    word2vec.word2vec(dir + '/text8-phrases', dir + '/text8.bin', size=300, verbose=True)
except Exception as e:
    print e
    print "Failed to build corpus"

model = word2vec.load(dir + '/text8.bin')
print "Finish"

def learn(trainFeat, trainClass, testFeat, testClass):
    model = Sequential()
    model.add(Convolution2D(50, 3, 3, border_mode='same', input_shape=(w2v,256,256)))
