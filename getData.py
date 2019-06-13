import re
import linecache
import matplotlib.pyplot as plt
import csv

def picklines(fileName, whatlines):
    return linecache.getline(fileName, whatlines)

class Generation(object):
    def __init__(self, time, acc, gen):
        self.time = time
        self.acc = acc
        self.gen = gen
    def __repr__(self):
        return "generation: " + self.gen +" "+ str(self.time) +"-"+str(self.acc)
    def return_csv(self):
        return self.gen, self.time,self.acc

populationSize = int(re.search(r'\d+', picklines("ai.py",24)).group())
generationSize = int(re.search(r'\d+', picklines("ai.py",25)).group())
generation = []
accuracy = []

with open("originalpopulation.txt","r") as f:
    t = picklines("originalpopulation.txt",(populationSize*2 + 1) )
    a =  picklines("originalpopulation.txt",(populationSize*2 + 3) )
    time = int(re.search(r'\d+', t ).group())
    acc  = float(re.search("\d+\.\d+", a ).group())
    accuracy.append(acc)
    gen = Generation(time, acc,"original")
    generation.append(gen)

for i in range (0,generationSize):
    try:
      fileToOpen = "currentpopulation"+str(i)+".txt"
      with open(fileToOpen,"r") as f:
        t = picklines(fileToOpen,(populationSize*2 + 1) )
        a =  picklines(fileToOpen,(populationSize*2 + 3) )
        time = int(re.search(r'\d+', t ).group())
        acc  = float(re.search("\d+\.\d+", a ).group())
        accuracy.append(acc)
        gen = Generation(time, acc,str(i))
        generation.append(gen)
    except:
      print("couldnt open file n " + str(i))

x = []
csv_data = [["GENERATION","TIME","ACCURACY"]]
for obj in generation:
    x.append(obj.gen)
    csv_data.append(obj.return_csv())
    

with open('results.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_data)    
    
plt.xlabel('Generation number')
plt.ylabel('Accuracy')
plt.title('AVG generations accuracy')
axes = plt.gca()
axes.set_ylim([0.0,1.0])
axes.set_xlim( [0, generationSize ] )
plt.plot(x, accuracy)
plt.savefig('results.png', bbox_inches='tight')
