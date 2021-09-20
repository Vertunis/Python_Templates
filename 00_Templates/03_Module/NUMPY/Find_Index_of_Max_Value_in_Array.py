#https://www.it-swarm.com.de/de/python/python-index-fuer-das-maximum-der-liste-suchen/1068216383/
#https://stackoverflow.com/questions/18079029/index-of-element-in-numpy-array

import numpy as np

a = [2,1,5,234,3,44,7,6,4,5,9,11,12,14,13]
b = np.arange(5)

# Aus normalem Array
index_max = max( (v, i) for i, v in enumerate(a) )[1] # Findet den Index des Maximums
print(f"Index mit max Value aus Vektor a: {index_max}")
index_max_a = a.index(max(a))
print(f"Index mit max Value aus Vektor a: {index_max_a}")

# Aus Numpy Array
index_max_b = np.where(b == max(b))
print(f"Index mit max Value aus Vektor b: {index_max_b[0][0]}")

value = 4
i, = np.where(np.isclose(b, value)) # Findet Wert in der nähe von Value
print(f"Index von Wert in nähee von Value aus Vektor b: {i[0]}")


