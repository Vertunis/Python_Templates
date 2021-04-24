# Hier wird beschrieben wie man Vektoren entweder Zeilenweise oder Spaltenweise hinzufügt
# https://stackoverflow.com/questions/20978757/how-to-append-a-vector-to-a-matrix-in-python

import numpy as np

# Hinzufügen eines Vektors in X Richtung -> x0 Richtung aka Spalten
m = np.zeros((10, 4))
v = np.ones((10, 1))
c = np.c_[m, v]

print(m)
print(v)
print(c)

# Hinzufügen eines Vektors in Y Richtung -> x1 Richtung aka Spalten
m = np.zeros((4, 10))
v = np.ones((1, 10))
r = np.r_[m, v]

print(m)
print(v)
print(r)