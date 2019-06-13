import matplotlib.pyplot as plt

generationSize = 30
maxAccuracy = 0
maxAccuracyList = []
with open("originalpopulation.txt","r") as f:
    for line in f:
        if line != '\n':
            #print("line is : ",line)
            lst = line.split(":")
            acc = float(lst[1])
            if acc > maxAccuracy and acc <= 1:
                maxAccuracy = acc
    #print (maxAccuracy)

maxAccuracyList.append(maxAccuracy)

for i in range (0,generationSize):
    maxAccuracy = 0
    try:
      fileToOpen = "currentpopulation"+str(i)+".txt"
      with open(fileToOpen,"r") as f:
        for line in f:
            if line != '\n':                
                lst = line.split(":")
                acc = float(lst[1])
                if acc > maxAccuracy and acc <= 1:
                    maxAccuracy = acc
        #print( str(maxAccuracy) + "for file" +  str(i))
    except:
      print("couldnt open file n " + str(i))

    maxAccuracyList.append(maxAccuracy)


print(maxAccuracyList)

   
plt.ylabel('Max Accuracy')
plt.xlabel('Generation number')
plt.title('single Result')
axes = plt.gca()
axes.set_ylim([0.0,1.0])
axes.set_xlim( [0, generationSize ] )
x = list(range(0, 31))
plt.plot( x, maxAccuracyList)
plt.savefig('Singleresults.png', bbox_inches='tight')
