# https://www.it-swarm.com.de/de/python/arrays-numpy-nach-spalte-sortieren/970034579/
# https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
# https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
import numpy as np

myList = [1, 2, 3, 100, 5]
ind_sort = np.argsort(myList)  #Index der Sortierung

new_list = np.zeros(shape=(5,1), dtype=np.float32)
#print(new_list.shape[0])
for i in range(new_list.shape[0]):
    #print(i)
    new_list[i] = myList[ind_sort[i]]

print("Fertig")