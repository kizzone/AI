
import random 
from numpy.random import choice
def createPseudoRandomGenotype():
    
    randomGenes = []
    #Layer da aggiugnere prima del Flatten
    for j in range (0,5):
        randomGenes.extend( generateRandomConv(j) )
        randomGenes.extend( generateRandomActivation() )
        randomGenes.extend( generateRandomMaxPool(j) )
        randomGenes.extend( generateRandomDrop(j) )
    #Layer da aggiugnere dopo del Flatten
    for j in range (0,3):
        randomGenes.extend( generateRandomDense(j) )
        randomGenes.extend( generateRandomActivation() )
        randomGenes.extend( generateRandomDrop(j) )

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
