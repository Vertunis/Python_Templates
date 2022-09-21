# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate

import numpy as np

x = np.arange(20).reshape(2, 2, 5) # Reshaping auf Dimension 2,2,5
print(x)
print(x.shape)

y = np.arange(20, 30).reshape(2, 1, 5)
print(y)
print(y.shape)




