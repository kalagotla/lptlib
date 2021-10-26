# TODO: Need to develop this test case for search algorithm
#  Convert the array data to p3d data. Then implement the search algorithm

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)
z = np.linspace(0, 1, 2)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
C = np.zeros(X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z)
ax.scatter3D(0.5, 0.75, 0.5, '.r')
plt.grid()
plt.show()

K = np.array((X, Y))
dist = np.sqrt((K[0, ...] - 2.5) ** 2 +
               (K[1, ...] - 1.25) ** 2)
print(X.shape)
print(K.shape)
print(dist)
print(divmod(dist.argmin(), dist.shape[1]))
