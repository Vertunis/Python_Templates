import numpy as np
'''
https://www.w3resource.com/numpy/manipulation/reshape.php
The reshape() function is used to give a new shape to an array without changing its data.
Syntax: 

numpy.reshape(a, newshape, order='C')
    
a 	        Array to be reshaped. 	Required

newshape 	The new shape should be compatible with the original shape. 
            If an integer, then the result will be a 1-D array of that length. 
            One shape dimension can be -1. 
            In this case, the value is inferred from the length of the
             array and remaining dimensions.
'''

# Beispiel 1
my_array = np.arange(4).reshape(2,2) # Return evenly spaced values within a given interval.

# Beispiel 1
x = np.array([[2,3,4], [5,6,7]])
y = np.reshape(x, (3, 2))

print(x)
print(y)