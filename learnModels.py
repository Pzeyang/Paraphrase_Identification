from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

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

def SVM(trainFeat, trainClass, testFeat):
    model = svm.SVC()
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

def KNN(trainFeat, trainClass, testFeat):
    model = neighbors.KNeighborsClassifier(15, weights='distance')
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

def MaxEnt(trainFeat, trainClass, testFeat):
    model = linear_model.LogisticRegression(n_jobs=-1)
    model.fit(trainFeat, trainClass)
    return model.predict(testFeat)

def NN(trainFeat, trainClass, testFeat):
    hn = 256
    drop = 0.75
    model = Sequential()
    model.add(Dense(input_dim = len(trainFeat[0]), output_dim = hn))
    #model.add(Activation('sigmoid'))
    model.add(PReLU())
    model.add(Dropout(drop))
    model.add(Dense(hn))
    #model.add(Activation('sigmoid'))
    model.add(PReLU())
    model.add(Dropout(drop))
    model.add(Dense(1)) # output layer
    model.add(Activation('sigmoid'))
    #model.add(PReLU())
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(trainFeat, trainClass, nb_epoch=50)
    return model.predict_classes(testFeat)

