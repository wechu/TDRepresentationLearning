import numpy as np
np.set_printoptions(suppress=True)
data = np.load("gridworlddata/wallgridworld_traindata_0.npy", allow_pickle=True)

visits = np.zeros((6,6))
dones = 0
for transition in data:
    if transition[4]:
        dones += 1
    state = transition[0]
    state = np.rint(state*np.array([6,6])).astype('int32')
    visits[tuple(state)] += 1

print(visits)
print(dones)

# for x in data:
#     print(x)

