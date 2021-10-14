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

################################################
#  Bessere Variante für Multidimensionale Arrays
################################################
myList_2 = np.array([[7, 3, 5],
                     [2, 1, 7],
                     [6, 1, 7]])
ind_sort = myList_2[:, 0].argsort(kind='stable')  # Sortiert nach erster Spalte von myList_2
#ind_sort = np.argsort(myList_2[:, 0], kind='stable', axis=0) # Alternative Schreibweise
myList_2 = myList[ind_sort]

print("Fertig")