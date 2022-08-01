import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# values = np.load('cartpoledata/Cartpole_test_values.npy')
# states = np.load('cartpoledata/Cartpole_test_states.npy')
# values_std = np.load('cartpoledata/Cartpole_test_values_std_error.npy')



values = np.load('puckworlddata/PuckWorld_test_values.npy')
states = np.load('puckworlddata/PuckWorld_test_states_NV.npy')
values_std = np.load('puckworlddata/PuckWorld_test_values_std_error.npy')


# plt.figure()
# plt.hist(values)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_zlim((0.1, 0.3))

# x, x_dot, theta, theta_dot for cartpole
#state['player_x'], state['player_y'], state['player_velocity_x'], state['player_velocity_y'],
#                     state['good_creep_x'], state['good_creep_y'], state['bad_creep_x'], state['bad_creep_y']])

def dist(p1, p2):
    return np.sqrt((p1[:,0] - p2[:,0]) ** 2 + (p1[:,1] - p2[:,1]) ** 2)




dim1 = states[:,0]#dist(states[:, 0:2], states[:, 4:6])
dim2 = states[:,1]#dist(states[:, 0:2], states[:, 6:8])
ax.scatter(dim1, dim2, values)
# ax.scatter(dim1, dim2, values + values_std*np.sqrt(2000), color='r', alpha=0.1)
# ax.scatter(dim1, dim2, values - values_std*np.sqrt(2000), color='g', alpha=0.1)

# print(np.mean(values_std*np.sqrt(2000)))

plt.xlabel("dim1")
plt.ylabel("dim2")
plt.show()



# plt.hist(states[:, 2])