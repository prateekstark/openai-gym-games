import numpy as np
import matplotlib.pyplot as plt
timesteps = np.load('timesteps.npy')
print(timesteps)


performance_list = []
for i in range(timesteps.shape[0]):
	if(i%10 == 0):
		if(i != 0):

			performance_list.append(sum(temp_score)/10.0)
		temp_score = [timesteps[i]]
	else:
		temp_score.append(timesteps[i])

plt.plot(performance_list)
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.savefig('MountainCar-v0_semi_gradient_sarsa_linear.png')
plt.show()
plt.show()