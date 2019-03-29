from __future__ import print_function
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

#----------------------------------------------------------------------------------------------------------------------------------------------
batch_size = 64
num_classes = 10
epochs = 30
populationSize = 100
generation =  100
pazienza= 5
#----------------------------------------------------------------------------------------------------------------------------------------------
#Si assicura che i due individui da combinare siano "abbastanza" diversi
def areNotSimilar(list1, list2):
    z = 0

    for i in range(0, len(list1)):
        if list1[i] != list2[i]:
            z += 1
    if z > 3:
        return True 
    else:
        return False

#Calcola l'accuratezza media della generazione non contando i modelli sbagliati
def calculateGenerationWorkingModelAccuracy(generation):
    if len(generation)==0:
        return 0
    else:
        return sum(i.accuracy for i in generation if i.accuracy  >= 0)

def countWorkingModel(generation):
    return sum(1 for i in generation if i.accuracy  >= 0)

#Ritora una lista contentente i 5 Individui migliori
def getBest5( best5, individual):
    if len(best5) == 0:
        best5.append(individual)
    else:
        for obj in best5:
            if individual.accuracy > obj.accuracy or len(best5) < 5:
                if len(best5) == 5:
                    best5.remove(obj)
                best5.append(individual)
                break
    return sorted(best5, key=lambda x: x.accuracy, reverse=False)

#Crea, secondo la specifica,  un genoma casuale per un individuo 
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

    #print ( "Randome genes :"  + str (randomGenes))
    
    return randomGenes

#Genera i parametri per un'attivazione di tipo casuale, relu è quella più probabile
def generateRandomActivation():
    flag = bool(random.getrandbits(1))
    activationType = choice(["relu" , "elu" , "tanh" ], p=[0.6, 0.3, 0.1])
    return flag,activationType

#Genera i parametri per una convoluzione di tipo casuale, il numero di filtri e la grandezza del kernel aummenta con l'aumentare della profondità del modello, 
def generateRandomConv(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        filterNum = choice([48 , 92 , 192 , 256 ], p=[0.6, 0.2 , 0.15, 0.05])
        filterKernel = choice([3, 5, 7], p=[0.5, 0.3, 0.2])
    else:
        filterNum = choice([48 , 92 , 192 , 256 ], p=[ 0.05, 0.15 , 0.2, 0.6])
        filterKernel = choice([3, 5, 7], p= [0.2, 0.3, 0.5])
    return flag, filterNum , filterKernel

#Genera i parametri per un Dropout casuale, il valore aumenta con la profondità
def generateRandomDrop(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        drop = choice([0.3 , 0.4 , 0.5 ], p=[ 0.5 , 0.3, 0.2])
    else:
        drop = choice([0.4 , 0.5 , 0.7 ], p=[0.2, 0.3, 0.5])
    return flag, drop

#Genera i parametri per un Dense casuale, il valore aumenta con la profondità
def generateRandomDense(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        neur = choice([ 128 , 256,  512 ], p=[0.5, 0.3, 0.2])
    else:
        neur = choice([ 128 , 256,  512 ], p=[0.2, 0.3, 0.5])
    return flag, neur 

#Genera i parametri per un maxpool casuale
def generateRandomMaxPool(depth):
    flag = bool(random.getrandbits(1))
    if depth < 2:
        size = choice([2 , 4 ], p=[0.6, 0.4])
    else:
        size = choice([2 , 4  ], p=[0.8, 0.2])
    return flag, size

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
            earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0, patience = pazienza , verbose=0, mode='auto')
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

#Seleziona due individui casuali, diversi tra loro, da far combinare
def selecteTwoRandowPeople (list):
    rand1 = randint (0, len(list)-1)
    rand2 = randint (0, len(list)-1)
    if rand2 != rand1:
        return list[rand1],list[rand2]
    else:
        return selecteTwoRandowPeople(list)

# è la funzione piu brutta che abbia mai fatto, trova un modo migliore per cortesia,
# Modifica un gene casuale, il gene non riottiene lo stesso valore che aveva prima
def makingRandomMutation(genome):

    rand = randint (0,len(genome)-1)
    #print("mutation at index", rand)
    if rand < 45:
        #extract  beforeFlatten "rows"
        remainder = rand % 9
        if remainder == 0:
            #conv flag, flipping flag
            genome[rand] = not genome[rand]
        elif remainder == 1:
            #conv filter numb
            a = [48 , 92 , 192 , 256 ]
            a.remove(genome[rand])
            genome[rand] = choice(a)
        elif remainder == 2:
            #conv kernel size
            a = [3, 5, 7]
            a.remove(genome[rand])
            genome[rand]  = choice(a)
        elif remainder == 3:
            #activation flag
            genome[rand] = not genome[rand]
        elif remainder == 4:
            #activation type
            a = ["relu" , "elu" , "tanh" ]
            a.remove(genome[rand])
            genome[rand] = choice(a)
        elif remainder == 5:
            #maxpool flag
            genome[rand] = not genome[rand]
        elif remainder == 6:
            #maxpool size
            a = [2 , 4 ]
            a.remove(genome[rand])
            genome[rand] = choice(a)
        elif remainder == 7:
            #dropout flag
            genome[rand] = not genome[rand]
        elif remainder == 8:
            #dropout rate
            a = [0.3 , 0.4 , 0.5, 0.7]
            a.remove(genome[rand])
            genome[rand] = choice(a)
    else:
        remainder = rand % 6
        if remainder == 3:
            #dense flag, flipping flag
            genome[rand] = not genome[rand]
        elif remainder == 4:
            #dense neurons
            a = [ 128 , 256,  512 ]
            a.remove(genome[rand])
            genome[rand] = choice(a)
        elif remainder == 5:
            #activation flag
            genome[rand] = not genome[rand]
        elif remainder == 0:
            #activation type
            a = ["relu" , "elu" , "tanh" ]
            a.remove(genome[rand])
            genome[rand] = choice(a)
        elif remainder == 1:
            #drop flag, flipping flag
            genome[rand] = not genome[rand]
        elif remainder == 2:
            #dropout rate
            a = [0.3 , 0.4 , 0.5, 0.7]
            a.remove(genome[rand])
            genome[rand] = choice(a)

    return genome

#Esegue un Double-point crossover e una mutazione ad uno specifico rate
def combine(a,b):

    c = []
    #==============aggiustare qua
    split1 = randint (1, len(b)-1) 
    split2 = random.randint(split1, len(b))
    c.extend ( a[:split1])
    c.extend ( b[split1:split2])
    c.extend ( a[split2:])
    #making a random mutation at a given rate
    rand = randint (0,25)
    if (rand == 2):
        print("a mutation occurred")
        c = makingRandomMutation(c)
    return c

#Ritorna una lista di nuovi genotipi tramite i meccanismi di crossover e mutation
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

#Ritorna l'indice migliore di una lista associato a un dato
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

#Crea una lista di nuovi genotipi, seleziona gli individui con fitness piu alta
#https://stackoverflow.com/questions/16489449/select-element-from-array-with-probability-proportional-to-its-value/42855534#42855534
def createNewGeneration(olderGeneration):

    #selecting individal with higher fitness
    newerGenerationGenotype = []
    toMix = []

    olderGenerationaccuracy = [ obj.accuracy for obj in olderGeneration if obj.accuracy > 0] #modificato qui con il maggiore di 0
    weigthedprobability = np.cumsum( olderGenerationaccuracy)

    for i in range (0, populationSize):
        rand = uniform (0, weigthedprobability[-1] )
        #trovare a quale indice appartiene e ritornare il cromosoma corrispondente a quell'accuracy
        index = binarySearch(weigthedprobability, rand)

        cromosoma =  [ t.genotype for t in olderGeneration if t.accuracy ==  olderGenerationaccuracy[index] ]
    
        toMix.extend(cromosoma)
        
    x = crossoverAndMutation( toMix )
    newerGenerationGenotype.extend( x )

    return newerGenerationGenotype

#Calcola l'accuratezza media della generazione
def calculateGenerationAccuracy(generation):
    if len(generation)==0:
        return 0
    else:
        return generation[0].accuracy + calculateGenerationAccuracy(generation[1:]) 
#----------------------------------------------------------------------------------------------------------------------------------------
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
        genoma = createPseudoRandomGenotype()
        print("evaluating index", str(i))
        accuracy = createAndEvaluate(genoma, x_train,x_test,y_train,y_test)
        testSub = Individual(genoma,accuracy)
        population.append(testSub)
        best5 = getBest5(best5,testSub)
        print("finish number ", str(i) )


    end = timer()

    elapsedTime = end - start

    avg = calculateGenerationAccuracy(population)/len(population)
    div = countWorkingModel(population)
    if div == 0:
        div = 1
    num = calculateGenerationWorkingModelAccuracy(population)
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
 
 
    for gen in range (0, generation):
        print ("current generation " + str(gen) + "/" + str(generation))
        currentpopulationgenome = createNewGeneration(population)
        # for i in range (0, len(currentpopulationgenome)):
        #     print ( "generation" + str(gen) + "index " + str(i) + " is " + str(currentpopulationgenome[i])  + "\n" )
        t0 = time.time()
        #print("start",str(start))
        for index, obj in enumerate(currentpopulationgenome):
            accuracy = createAndEvaluate(obj, x_train,x_test,y_train,y_test)
            testSub = Individual(obj,accuracy)
            best5 = getBest5(best5,testSub)
            print("evaluating people n " + str(index) +"of current generation: " + str(gen))
            currentpopulation.append(testSub)

        f = time.time() - t0
        #print("elapsed time ",f)
        avg = calculateGenerationAccuracy(currentpopulation)/len(currentpopulation)
        div = countWorkingModel(currentpopulation)
        if div == 0:
            div = 1
        num = calculateGenerationWorkingModelAccuracy(currentpopulation)
        print("DIV value",div)
        print("NUM value",num)
        avg2 = num/div
        in_file = open('currentpopulation'+str(gen)+'.txt', 'w')
        in_file.writelines("%s\n" % people for people in currentpopulation)
        in_file.writelines("Total time: %d\n" % f)
        in_file.writelines("Average generation accuracy: %s\n" % str(avg)) 
        in_file.writelines("Average working model of generation accuracy: %s\n" % str(avg2)) 
        in_file.close()
        currentpopulation.clear()
        print("current best five",str(best5))
        with open('best5.txt', 'w') as filehandle:  
            filehandle.writelines("%s\n" % people for people in best5)
 

    population.clear()

    print("Writing Original population best five")
    with open('best5.txt', 'w') as filehandle:  
        filehandle.writelines("%s\n" % people for people in best5)

if __name__ == "__main__":
    main()

