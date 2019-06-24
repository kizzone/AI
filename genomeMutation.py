import numpy as np

from random import randint,uniform 
from ai import populationSize, crossOverRate
from numpy.random import choice
from utilities import binarySearch, getBestFromGeneration


#Seleziona due individui casuali, diversi tra loro, da far combinare
def selecteTwoRandowPeople (list):
    rand1 = randint (0, len(list))
    rand2 = randint (0, len(list))
    if rand2 != rand1:
        return list[rand1],list[rand2]
    else:
        return selecteTwoRandowPeople(list)

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
    split1 = randint (1, len(b)-1) 
    split2 = randint(split1, len(b))
    c.extend ( a[:split1])
    c.extend ( b[split1:split2])
    c.extend ( a[split2:])
    #making a random mutation at a given rate
    rand = randint (0,20)
    if (rand == 2):
        print("a mutation occurred")
        c = makingRandomMutation(c)
    return c

#Ritorna una lista di nuovi genotipi tramite i meccanismi di crossover e mutation
def crossoverAndMutation(lista):
        
    newGenotypes = []
    for i in range (0, len(lista)):
        a,b = selecteTwoRandowPeople(lista)
        #print ( "selected a and b " + str(a) + "\n\n" + str(b))
        c = combine(a,b)
        #print("RESULTING COMBINATION IS " + str(c))
        newGenotypes.append(c)
        #print("newGenotype is now  " + str(newGenotypes))
    return newGenotypes

#Crea una lista di nuovi genotipi, seleziona gli individui con fitness piu alta
def createNewGeneration(olderGeneration):

    newerGenerationGenotype = []
    toMix = []

    #getting best from previous generation
    bestGenome = getBestFromGeneration(olderGeneration,crossOverRate)
    print("gli individui migliori da portare avanti della generazione precedente sono:")
    for gen in bestGenome:
        print(gen.genotype)
        newerGenerationGenotype.append (gen.genotype)
        
    
    #selecting individal with higher fitness
    olderGenerationaccuracy = [ obj.accuracy for obj in olderGeneration if obj.accuracy > 0]
    weigthedprobability = np.cumsum( olderGenerationaccuracy)

    #IN TO MIX AGGIUNGO TUTTI  I MIGLIORI TRANNE QUELLI CHE GIà HO INCLUSO DALLA GENERAZIONE PEGGIORE
    for i in range (0, populationSize-crossOverRate):
        rand = uniform (0, weigthedprobability[-1] )
        #trovare a quale indice appartiene e ritornare il cromosoma corrispondente a quell'accuracy
        index = binarySearch(weigthedprobability, rand)
        cromosoma =  [ t.genotype for t in olderGeneration if t.accuracy ==  olderGenerationaccuracy[index] ]
        toMix.extend(cromosoma)
        
    x = crossoverAndMutation( toMix )
    newerGenerationGenotype.extend( x )

    return newerGenerationGenotype