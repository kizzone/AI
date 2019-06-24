from __future__ import print_function
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
import os
import random
import time
from numpy.random import choice
from random import randint,uniform 

from timeit import default_timer as timer
K.set_image_dim_ordering('th')

import genomeMutation 
import genomeCreation
import utilities
#----------------------------------------------------------------------------------------------------------------------------------------------
batch_size = 64
num_classes = 10
epochs = 25
populationSize = 30
generation = 20
pazienza= 20
crossOverRate = 4
#----------------------------------------------------------------------------------------------------------------------------------------------

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
        try:
            # initiate RMSprop optimizer
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            #parallel_model = multi_gpu_model(model, gpus=2)
            model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0, patience = pazienza , verbose=1, mode='auto')
            callbacks_list = [earlystop]

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True,verbose=1,callbacks=callbacks_list)

            scores = model.evaluate(x_test, y_test, verbose=1)
            #clear session of tensorflow
            K.clear_session()
            return scores[1]

        except:
            return -1
    else:
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
    currentpopulation = []
    currentpopulationgenome = []
    best5 = []
  
    start = timer()

    for i in range (0, populationSize):
        #create first pseudorandom population genotype and evaluate 
        genoma = genomeCreation.createPseudoRandomGenotype()
        print("evaluating index", str(i))
        accuracy = createAndEvaluate(genoma, x_train,x_test,y_train,y_test)
        testSub = Individual(genoma,accuracy)
        population.append(testSub)
        best5 = utilities.getBest5(best5,testSub)
        print("finish number ", str(i) )


    end = timer()

    elapsedTime = end - start

    avg = utilities.calculateGenerationAccuracy(population)/len(population)
    div = utilities.countWorkingModel(population)
    if div == 0:
        div = 1
    num = utilities.calculateGenerationWorkingModelAccuracy(population)
    print("DIV value",div)
    print("NUM value",num)
    avg2 = num/div
    
    with open('originalpopulation.txt', 'w') as filehandle:  
        filehandle.writelines("%s\n" % people for people in population)
        filehandle.writelines("Total time: %d\n" % elapsedTime)
        filehandle.writelines("Average generation accuracy: %s\n" % str(avg))
        filehandle.writelines("Average working model of generation accuracy: %s\n" % str(avg2)) 
    
    print("Original population best five",str(best5))
    
    
    with open('best5.txt', 'w') as filehandle:  
        filehandle.writelines("%s\n" % people for people in best5)
 
    currentpopulationgenome1 = genomeMutation.createNewGeneration(population)
    for gen in range (0, generation):
        print ("current generation " + str(gen) + "/" + str(generation))
        if gen == 0:
            currentpopulationgenome = currentpopulationgenome1
        else:
            currentpopulationgenome = genomeMutation.createNewGeneration(currentpopulation)
        # for i in range (0, len(currentpopulationgenome)):
        #     print ( "generation" + str(gen) + "index " + str(i) + " is " + str(currentpopulationgenome[i])  + "\n" )
        t0 = time.time()
        #print("start",str(start))
        #PER EVITARE IL MAX RECURSION LEVEL ERRORe
        currentpopulation.clear()
        #CONTROLLO QUI
        for index, obj in enumerate(currentpopulationgenome):
            accuracy = createAndEvaluate(obj, x_train,x_test,y_train,y_test)
            testSub = Individual(obj,accuracy)
            best5 = utilities.getBest5(best5,testSub)
            print("evaluating people n " + str(index) +"of current generation: " + str(gen))
            currentpopulation.append(testSub)

        f = time.time() - t0
        #print("elapsed time ",f)
        avg = utilities.calculateGenerationAccuracy(currentpopulation)/len(currentpopulation)
        div = utilities.countWorkingModel(currentpopulation)
        if div == 0:
            div = 1
        num = utilities.calculateGenerationWorkingModelAccuracy(currentpopulation)
        print("DIV value",div)
        print("NUM value",num)
        avg2 = num/div
        in_file = open('currentpopulation'+str(gen)+'.txt', 'w')
        in_file.writelines("%s\n" % people for people in currentpopulation)
        in_file.writelines("Total time: %d\n" % f)
        in_file.writelines("Average generation accuracy: %s\n" % str(avg)) 
        in_file.writelines("Average working model of generation accuracy: %s\n" % str(avg2)) 
        in_file.close()

        print("current best five",str(best5))
        with open('best5.txt', 'w') as filehandle:  
            filehandle.writelines("%s\n" % people for people in best5)
 

    population.clear()

    #retrieve data
    os.system("getData.py")
    os.system("getGraph.py")
    os.system("trainBestGeneration.py")


    
if __name__ == "__main__":
    main()

