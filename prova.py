lista = []



class Individual (object):
    def __init__(self, genotype, accuracy):
        self.genotype = genotype
        self.accuracy = accuracy
    
    def __repr__(self):
        return str(self.genotype) + ":" + str(self.accuracy) + "\n"


ind = Individual([1,23,4],0.3)
lista.append(ind)

ind = Individual([1,23,4],0.7)
lista.append(ind)

ind = Individual([1,23,4],0.2)
lista.append(ind)

ind = Individual([1,23,4],0.8)
lista.append(ind)

ind = Individual([1,23,4],1)
lista.append(ind)

ind = Individual([1,23,4],0.1)
lista.append(ind)

def getBestFromGeneration( currentPopulation, crossoverRate ):

    best = []
    currentPopulation.sort(key=lambda x: x.accuracy, reverse=True)
    for individual in currentPopulation:
        print("examintaing individual", individual)
        if retrievedPeople > 0: 
            print("individual is better", individual)
            best.append(individual)
            retrievedPeople = retrievedPeople -1

    return best



print(getBestFromGeneration(lista,2))