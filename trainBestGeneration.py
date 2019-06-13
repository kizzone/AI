from __future__ import print_function
import re
import ast
import linecache
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import random
import time
from numpy.random import choice
from random import randint,uniform 
from keras import backend as K
from timeit import default_timer as timer
K.set_image_dim_ordering('th')

generationSize = 100
populationSize = 100
genomeToTrain = []
num_classes = 10
epochs = 50
batch_size= 64

#L'individuo è caratterizzato da un genotipo e la sua accuratezza
class Individual (object):
    def __init__(self, genotype, accuracy):
        self.genotype = genotype
        self.accuracy = accuracy
    
    def __repr__(self):
        return str(self.genotype) + ":" + str(self.accuracy) + "\n"

#Crea il modello sequenziale a partire dal genoma specificato
def createModelFromGenotype( genoma ,x_train):
    
    #a 45 c'è la seconda parte del genoma
    split = 45
    part1 = genoma [:split]
    part2 = genoma [split:]

    model = Sequential()
    #input layer di ingresso
    model.add(InputLayer(input_shape=x_train.shape[1:]))

    beforeFlatten = []
    afterFlatten = []

    try: 
        for j in range (0,5):
            beforeFlatten.append( part1[9 * j: (9*j) + 9 ])
        
        for row in beforeFlatten:
            for i in range (0,3):
                if row[0] == True:
                    #print("Creo una convoluzione con numero di filtri uguale a " + str(row[1]) + "e kernel di dimensione" + str(row[2]))
                    model.add(Conv2D( row[1] , (row[2], row[2]), padding='same' ) )
                    break
            for i in range (3,5):
                if row[3] == True:
                    #print("Creo un Activation con tipo " + str(row[4] ))
                    model.add( Activation(row[4]))
                    break
            for i in range (5,7):
                if row[5] == True:
                    #print("Creo un Maxpooling con  dimensione max " + str( row[6] ) )
                    model.add( MaxPooling2D ( pool_size = (row[6], row[6]), data_format='channels_first' ) )
                    break
            for i in range (7,9):
                if row[7] == True:
                    #print("Creo un Dropout  con  rate : " + str( row[8] ) )
                    model.add( Dropout (row[8] )  )
                    break
        
        for j in range (0,3):
            afterFlatten.append( part2[6 * j: (6*j) + 6 ])
    
        model.add(Flatten())

        for row in afterFlatten:
            for i in range (0,2):
                if row[0] == True:
                    #print("Creo un Dense  con neuroni " + str( row[1] ) )
                    model.add(Dense(row[1]))
                    break
            for i in range (2,4):
                if row[2] == True:
                    #print("Creo un Activation con tipo " + str(row[3] ))
                    model.add( Activation(row[3]))
                    break
            for i in range (4,6):
                if row[4] == True:
                    #print("Creo un Dropout  con  rate : " + str( row[5] ) )
                    model.add( Dropout (row[5] )  )
                    break

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        #model.summary()

        return model
    except:
        return None

#Crea il modello e lo valuta, ritorna -1 se il modello generato è inutilizzabile
def createAndEvaluate(genome, x_train, x_test, y_train,y_test):

    model = createModelFromGenotype (genome,x_train)

    if model is not None:

        print("model is not none")
        try:
                # initiate RMSprop optimizer
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            #parallel_model = multi_gpu_model(model, gpus=2)
            model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0, patience = 40 , verbose=0, mode='auto')
            callbacks_list = [earlystop]

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True,verbose=0,callbacks=callbacks_list)

            scores = model.evaluate(x_test, y_test, verbose=1)
            #clear session of tensorflow
            K.clear_session()
            return scores[1]

        except:
            return -1
    else:
        return -1

def picklines(fileName, whatlines):
    return linecache.getline(fileName, whatlines)

def getBestGeneration():
    maxAvgAcc = 0
    indexOfMaxAvg = -1
    with open("originalpopulation.txt","r") as f:
        a =  picklines("originalpopulation.txt",(populationSize + 3) )
        acc  = float(re.search("\d+\.\d+", a ).group())
        if acc > maxAvgAcc:
            maxAvgAcc = acc
        
    for i in range (0,generationSize):
        try:
            fileToOpen = "currentpopulation"+str(i)+".txt"
            with open(fileToOpen,"r") as f:
                a =  picklines(fileToOpen,(populationSize + 3) )
                acc  = float(re.search("\d+\.\d+", a ).group())
                if acc > maxAvgAcc:
                    maxAvgAcc = acc
                    indexOfMaxAvg = i
        except:
            print("couldnt open file n " + str(i))

    print(maxAvgAcc,indexOfMaxAvg)
    
    return maxAvgAcc,indexOfMaxAvg

def main():
    _, index = getBestGeneration()
    try:
        fileToOpen = "currentpopulation"+str(index)+".txt"
        with open(fileToOpen,"r") as f:
            head = [next(f) for x in range(100)]
            for line in head:
                if line != "\n":
                    row = line.split(":")
                    #sprint(row[0])
                    s_list = ast.literal_eval(row[0])
                    genomeToTrain.append(s_list)         
    except:
        print("couldnt open file n " + str(index))

    #print(genomeToTrain)

    # The data, split between train and test sets:
    (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
    #Mi prendo il 10% del training
    x_train, X_test_scartare, y_train, y_test_scartare = train_test_split(x_train1, y_train1, train_size=0.40, random_state=42, stratify=y_train1)
    #Mi prendo il 10% del test
    X_train_scartare, x_test, y_train_scartare, y_test = train_test_split(x_test1, y_test1, test_size=0.40, random_state=42, stratify=y_test1)
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    with open("fullTrainingBestGeneration.txt","w") as fp:
        for i in genomeToTrain:
            accuracy = createAndEvaluate(i, x_train,x_test,y_train,y_test)
            testSub = Individual(i,accuracy)
            fp.writelines("%s\n" % testSub)
            print("finish number ", str(i), accuracy )


if __name__ == "__main__":
    main()