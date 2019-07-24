import numpy as np
import matplotlib.pyplot as plt
import pdb

evolutions = np.load('soft_evolution.npy')

for ev in evolutions:
    plt.plot(ev[:, 0], ev[:, 1], color='orange', alpha=0.5, linewidth=4)

plt.xlim([0,100])
plt.xlabel('Step #')
plt.ylabel('Friction coefficient')
plt.title('Soft constraints')
plt.show()
