
class Individual (object):
    def __init__(self, genotype, accuracy):
        self.genotype = genotype
        self.accuracy = accuracy
    
    def __repr__(self):
        return str(self.genotype) + ":" + str(self.accuracy) + "\n"

list1 =  [True, 4, False, 0.4, True, 128, True, 'tanh', True, 0.3, False, 64, False, 'elu', False, 0.3, True, 32, True, 'elu', True, 0.4]


ind5 = Individual(list1,0.3)
ind6 = Individual(list1,0.2)
ind3 = Individual(list1,0.4)
ind1 = Individual(list1,0.5)
ind2 = Individual(list1,0.6)
ind0 = Individual(list1,0.79)
ind4 = Individual(list1,0.1)

i5 = Individual(list1,0.8)
i6 = Individual(list1,0.99)
i3 = Individual(list1,0.7)
i1 = Individual(list1,0.1)
i2 = Individual(list1,0.02)
i0 = Individual(list1,0.21)
i4 = Individual(list1,0.001)



population = []
population2 = []
best5 = []
population.append(ind0)
population.append(ind1)
population.append(ind2)
population.append(ind3)
population.append(ind4)
population.append(ind5)
population.append(ind6)
population2.append(i0)
population2.append(i1)
population2.append(i2)
population2.append(i3)
population2.append(i4)
population2.append(i5)
population2.append(i6)

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


for obj in population:
    best5 = getBest5(best5,obj)

print("lunghezza " + str(len(best5)) + " i migliori " + str(best5))

for obj in population2:
    best5 = getBest5(best5,obj)

print("FINAL \n lunghezza " + str(len(best5)) + " i migliori " + str(best5))