import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.dpi'] = 300

data = np.load('out/data.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

point_count = 3000
#ax.scatter(data[0:29,0], data[0:29,1], data[0:29,2])
ax.scatter(data[:point_count, 0],
           data[:point_count, 1],
           data[:point_count, 2])

ax.set_xlabel('x')
ax.set_ylabel('xdot')
ax.set_zlabel('lambda')

plt.show()
