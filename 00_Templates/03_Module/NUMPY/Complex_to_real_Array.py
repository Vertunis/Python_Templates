import numpy as np

u = np.array([
    1.0 + 2.0j,
    2.0 + 4.0j,
    3.0 + 6.0j,
    4.0 + 8.0j
    ])

#u2 = numpy.ascontiguousarray(np.vstack((u.real, u.imag)).T)

u2 = np.concatenate((np.array(u.real, ndmin=2).T, np.array(u.imag, ndmin=2).T), axis=1)
u3 = np.concatenate(np.array(u.real, ndmin=2).T)
u4 = np.concatenate(np.array(u.real, ndmin=2))
print("finish")