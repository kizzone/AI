from __future__ import print_function
from ai import createModelFromGenotype

import numpy as np
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
from random import randint,uniform
import time
from numpy.random import choice
from keras import backend as K
K.set_image_dim_ordering('th')


batch_size = 32
num_classes = 10
epochs = 20
populationSize = 10
generation = 100

(x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
#Mi prendo il 10% del training
x_train, X_test_scartare, y_train, y_test_scartare = train_test_split(x_train1, y_train1, train_size=0.40, random_state=42, stratify=y_train1)
#Mi prendo il 10% del test
X_train_scartare, x_test, y_train_scartare, y_test = train_test_split(x_test1, y_test1, test_size=0.40, random_state=42, stratify=y_test1)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




class Individual(object):
    def __init__(self, genotype, accuracy):
        self.genotype = genotype
        self.accuracy = accuracy
    
    def __repr__(self):
        return str(self.genotype) + ":" + str(self.accuracy) + "\n"



list1 =  [False, 32, 7, False, 'relu', False, 2, False, 0.4, False, 32, 3, False, 'elu', True, 4, True, 0.4, True, 16, 7, False, 'elu', True, 4, False, 0.5, True, 64, 7, True, 'elu', True, 4, False, 0.3, False, 32, 7, False, 'elu', True, 4, False, 0.4, True, 128, True, 'tanh', True, 0.3, False, 64, False, 'elu', False, 0.3, True, 32, True, 'elu', True, 0.4]
list2 = [False, 32, 7, False, 'relu', True, 2, False, 0.4, False, 32, 3, False, 'elu', True, 4, True, 0.2, False, 16, 7, False, 'elu', True, 4, False, 0.5, True, 32, 7, True, 'relu', True, 2, True, 0.3, False, 16, 7, False, 'relu', True, 4, False, 0.2, False, 128, True, 'tanh', True, 0.3, False, 32, False, 'elu', True, 0.3, True, 32, False, 'relu', True, 0.2]
list3 = [True, 32, 7, False, 'elu', False, 2, False, 0.4, False, 32, 3, True, 'elu', True, 2, True, 0.4, True, 16, 7, False, 'elu', True, 4, False, 0.5, True, 64, 7, True, 'elu', True, 2, True, 0.3, False, 16, 7, False, 'relu', True, 4, False, 0.2, False, 128, True, 'tanh', True, 0.3, False, 32, False, 'elu', True, 0.3, True, 32, False, 'relu', True, 0.2]
list4 = [False, 32, 7, False, 'relu', True, 2, False, 0.4, False, 32, 3, False, 'relu', True, 2, True, 0.2, True, 16, 7, False, 'elu', True, 4, False, 0.5, False, 64, 7, True, 'elu', True, 2, True, 0.3, False, 16, 7, False, 'tanh', False, 4, False, 0.2, False, 128, True, 'tanh', False, 0.4, True, 32, False, 'elu', True, 0.3, True, 32, False, 'relu', True, 0.2]
list5 =[True, 32, 7, False, 'elu', False, 2, False, 0.4, False, 32, 3, True, 'elu', True, 2, True, 0.4, True, 16, 7, False, 'elu', True, 4, False, 0.5, False, 64, 7, True, 'elu', True, 2, True, 0.3, False, 16, 7, False, 'elu', True, 4, False, 0.4, True, 128, True, 'tanh', True, 0.3, False, 64, False, 'elu', False, 0.3, True, 32, True, 'elu', True, 0.4]

individuo = Individual (list1,0.8)
individuo2 = Individual (list2,0.6)
individuo3= Individual (list3,0.2)
individuo4 = Individual (list4,0.9)
individuo5 = Individual (list5,0.1)

olderGeneration = []
olderGeneration.append(individuo)
olderGeneration.append(individuo2)
olderGeneration.append(individuo3)
olderGeneration.append(individuo4)
olderGeneration.append(individuo5)


#--------------------------------------------------------------------------------
def selecteTwoRandowPeople (list):
    rand1 = randint (0, len(list)-1)
    rand2 = randint (0, len(list)-1)
    if rand2 != rand1:
        return list[rand1],list[rand2]
    else:
        return selecteTwoRandowPeople(list)

def combine(a,b):

    c = []
    split = randint (0, len(b)) 
    c.extend ( a[:split])
    c.extend ( b[split:])
    #making a random mutation at a given rate
    return c

def crossoverAndMutation(lista):
        
    newGenotypes = []
    for i in range (0, populationSize):
        a,b = selecteTwoRandowPeople(lista)
        #print ( "selected a and b " + str(a) + "\n\n" + str(b))
        c = combine(a,b)
        #print("RESULTING COMBINATION IS " + str(c))
        newGenotypes.append(c)
        #print("newGenotype is now  " + str(newGenotypes))
    return newGenotypes

def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid2 = lo + (hi - lo) / 2
        mid = int(mid2)
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind

def createNewGeneration(olderGeneration):

    #selecting individal with fitness above average
    newerGenerationGenotype = []
    toMix = []

    olderGenerationaccuracy = [ obj.accuracy for obj in olderGeneration ]
    weigthedprobability = np.cumsum( olderGenerationaccuracy)

    for i in range (0, populationSize):
        #contare pure i -1??
        rand = uniform (0, weigthedprobability[-1] )
        #trovare a quale indice appartiene e ritornare il cromosoma corrispondente a quell'accurac
        index = binarySearch(weigthedprobability, rand)

        cromosoma =  [ t.genotype for t in olderGeneration if t.accuracy ==  olderGenerationaccuracy[index] ]
    
        toMix.extend(cromosoma)
        
    #print (" Selected nodes to MIX:" + str ( toMix ) )
    x = crossoverAndMutation( toMix )
    newerGenerationGenotype.extend( x )

    return newerGenerationGenotype

#--------------------------------------------------------------------------------

p = createNewGeneration(olderGeneration)

for i in range (0, len(p)):
    print ( "index " + str(i) + " is " + str(p[i])  + "\n" )