import gym
import numpy as np
import matplotlib.pyplot as plt
import math

# Gradient Monte Carlo Algorithm Implementation

def epsilon_greedy_action(Q, state, e):
	if(np.random.uniform(0, 1) < e):
		action = env.action_space.sample()
	else:
		action = np.argmax(Q[state, :])
	return action

def get_epsilon(episode, num_episodes):
	return math.exp(-2.4*episode/num_episodes)

# Algorithm parameters: step size alpha from (0, 1], small epsilon > 0
alpha = 0.5
epsilon = 1
num_episodes =50000
gamma = 0.95

env = gym.make('BipedalWalker-v3')

# Initialize Q(s, a), for all s belongs to S+, a belongs to A(s), arbitarily except that Q(terminal, .) = 0
# Q = np.zeros((env.observation_space.n, env.action_space.n))
w = np.zeros((24, 4))
print((env.action_space))
# score_list = []
# timesteps = []
# # Loop for each episode
for episode in range(num_episodes):
# 	# Initialize S
	state = env.reset()

# 	# Loop for each step of episode
	while(True):
		env.render()
		# print(env.observation_space.high.shape)
		state = state.reshape(1, 24)
		# print(state)
# 		# Decaying epsilon
# 		epsilon = get_epsilon(episode, num_episodes)
# 		# print(epsilon)
# 		# Choose A from S using policy derived from Q (epsilon greedy)
# 		action = epsilon_greedy_action(Q, state, epsilon)
		action = env.action_space.sample()
# 		# Take action A, observe R and S`
		observation, reward, done, info = env.step(action)

# 		# Q(S, A) = Q(S, A) + alpha[R + gamma*maxa(Q(S`, A)) - Q(S, A)]
# 		Q[state, action] = Q[state, action] + alpha*(reward + (gamma*max(Q[observation, :])) - Q[state, action])
# 		# S <- S`
# 		state = observation
		
# 		# Terminal State
		if(done):
# 			print("Episode finished after {} timesteps".format(t+1))
# 			score_list.append(reward)
# 			timesteps.append(t)
			print("Reward: " + str(reward))
# 			# print(Q)
			break

# performance_list = []
# for i in range(num_episodes):
# 	if(i%100 == 0):
# 		if(i != 0):
# 			performance_list.append(sum(temp_score)/100.0)
# 		temp_score = [score_list[i]]
# 	else:
# 		temp_score.append(score_list[i])


# print("Printing Q-table: ")
# print(Q)
# plt.plot(performance_list)
# # plt.plot(score_list)
# # plt.plot(timesteps)
# plt.ylabel('Performance')
# plt.xlabel('Episodes')
# plt.savefig('FrozenLake-v0_q_learning.png')
# plt.show()
env.close()