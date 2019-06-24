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

#Calcola l'accuratezza media della generazione
def calculateGenerationAccuracy(generation):
    if len(generation)==0:
        return 0
    else:
        return generation[0].accuracy + calculateGenerationAccuracy(generation[1:]) 

#Calcola i modelli "funzionanti"
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


#ritorna una lista di crossoverRate elementi migliore dalla generazione attuale
def getBestFromGeneration( currentPopulation, crossoverRate ):

    best = []
    currentPopulation.sort(key=lambda x: x.accuracy, reverse=True)
    for individual in currentPopulation:
        print("examintaing individual", individual)
        if crossoverRate > 0: 
            print("individual is better", individual)
            best.append(individual)
            crossoverRate = crossoverRate -1

    return best

