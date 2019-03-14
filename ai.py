from __future__ import print_function
import numpy
import keras
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import InputLayer #per specificare l'input shape
import os
import random
import time
from numpy.random import choice

#----------------------------------------------------------------------------------------------------------------------------------------------
batch_size = 32
num_classes = 10
epochs = 20
populationSize = 50
generation = 100
#----------------------------------------------------------------------------------------------------------------------------------------------

#TODO inserire un seed per riprodurre l'esperimento
def createPseudoRandomGenotype():
    
    randomGenes = []

    for j in range (0,5):
        for i in range (0,3):
            #generate random conv paramater and append to  beforflatter
            randomGenes.extend( generateRandomConv(j) )
            break
        for i in range (3,5):
            #generate random conv paramater and append to  beforflatter
            randomGenes.extend( generateRandomActivation() )
            break
        for i in range (5,7):
            #generate random conv paramater and append to  beforflatter
            randomGenes.extend( generateRandomMaxPool(j) )
            break
        for i in range (7,9):
             #generate random conv paramater and append to  beforflatter
            randomGenes.extend( generateRandomDrop(j) )
            break
    

    for j in range (0,3):
        for i in range (0,2):
            #generate random Dense paramater and append to  afterflatter
            randomGenes.extend( generateRandomDense(j) )
            break
        for i in range (2,4):
            #generate random Activation paramater and append to  afterflatter
            randomGenes.extend( generateRandomActivation() )
            break
        for i in range (4,6):
            #generate random Dropout paramater and append to  afterflatter
            randomGenes.extend( generateRandomDrop(j) )
            break

    print ( "Randome genes :"  + str (randomGenes))
    
    return randomGenes

#----------------------------------------------------------------------------------------------------------------------------------------------
def generateRandomActivation():
    flag = bool(random.getrandbits(1))
    activationType = choice(["relu" , "elu" , "tanh" ], p=[0.6, 0.3, 0.1])
    return flag,activationType
def generateRandomConv(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        filterNum = choice([64 , 32 , 16 ], p=[0.2, 0.3, 0.5])
        filterKernel = choice([3, 5, 7], p=[0.5, 0.3, 0.2])
    else:
        filterNum = choice([64 , 32 , 16 ], p=[0.5, 0.3, 0.2])
        filterKernel = choice([3, 5, 7], p= [0.2, 0.3, 0.5])
    return flag, filterNum , filterKernel
def generateRandomDrop(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        drop = choice([0.3 , 0.4 , 0.5 ], p=[0.2, 0.3, 0.5])
    else:
        drop = choice([0.3 , 0.4 , 0.5 ], p=[0.5, 0.3, 0.2])
    return flag, drop
def generateRandomDense(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        neur = choice([64 , 128 , 32 ], p=[0.3, 0.5, 0.2])
    else:
        neur = choice([64 , 128 , 32 ], p=[0.3, 0.2, 0.5])
    return flag, neur 
def generateRandomMaxPool(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        size = choice([2 , 4 ], p=[0.6, 0.4])
    else:
        size = choice([2 , 4  ], p=[0.4, 0.6])
    return flag, size


'''
L'individuo è caratterizzato da un genotipo e la sua accuratezza
'''
class Individual (object):
    def __init__(self, genotype, accuracy):
        self.genotype = genotype
        self.accuracy = accuracy
    
    def __repr__(self):
        return "Hi my genotype is \n" + str(self.genotype) + "\nmy accuracy is:  " + str(self.accuracy) + "\n"

def createModelFromGenotype( genoma ,x_train):
    
    split = 45
    part1 = genoma [:split]
    part2 = genoma [split:]

    model = Sequential()
    model.add(InputLayer(input_shape=x_train.shape[1:]))

    beforeFlatten = []
    afterFlatten = []

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
                model.add( MaxPooling2D ( pool_size = (row[6], row[6]) ) )
                break
        for i in range (7,9):
             if row[7] == True:
                #print("Creo un Dropout  con  rate : " + str( row[8] ) )
                model.add( Dropout (row[8] )  )
                break
    
    for j in range (0,3):
        afterFlatten.append( part2[6 * j: (6*j) + 6 ])
   
    #----------------------------------------------------layer fisso-----------------------------------------------
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

    #----------------------------------------------------layers fissi----------------------------------------------- 
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

def createAndEvaluate(genome, x_train, x_test, y_train,y_test):

    #genoma = [True, 64, 3, True, 'relu', True, 2, True, 0.25, True, 32, 5, True, 'relu', True, 2, True, 0.3, True, 32, 3, True, 'relu', True, 2, True, 0.4, True, 64, 3, True, 'relu', True, 2, True, 0.25, True, 32, 5, True, 'relu', True, 2, True, 0.4, True, 64, True, 'relu', True, 0.25, True, 64, True, 'relu', True, 0.3, True, 64, True, 'relu', True, 0.4]
    

    model = createModelFromGenotype (genome,x_train)

    try:
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
        earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0, patience=3, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True,verbose=1,callbacks=callbacks_list)

        scores = model.evaluate(x_test, y_test, verbose=1)
        return scores[1]

    except:
        return -1


def main():

    # The data, split between train and test sets:
    (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
    #Mi prendo il 10% del training
    x_train, X_test_scartare, y_train, y_test_scartare = train_test_split(x_train1, y_train1, train_size=0.40, random_state=42, stratify=y_train1)
    #Mi prendo il 10% del test
    X_train_scartare, x_test, y_train_scartare, y_test = train_test_split(x_test1, y_test1, test_size=0.40, random_state=42, stratify=y_test1)
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    population = []

    

    start = time.time()

    for i in range (0, populationSize):
        #create first pseudorandom population genotype and evaluate 
        genoma = createPseudoRandomGenotype()
        accuracy = createAndEvaluate(genoma, x_train,x_test,y_train,y_test)
        testSub = Individual(genoma,accuracy)
        population.append(testSub)

    #stampa i risultati
    #for obj in population:
    ##    print(str(obj))
    end = time.time()

    elapsedTime = end - start
    with open('currentpopulation.txt', 'w') as filehandle:  
        filehandle.writelines("%s\n" % people for people in population)
        filehandle.writelines("Total time: %d\n" % elapsedTime)

    #do ngeneration times
    #create a new population from the "best" ones and run new evaluation

if __name__ == "__main__":
    main()
