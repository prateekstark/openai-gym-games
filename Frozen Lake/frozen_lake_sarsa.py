import gym
import numpy as np
import matplotlib.pyplot as plt
import math

# SARSA Implementation (on-policy TD control)

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

env = gym.make('FrozenLake-v0')

# Initialize Q(s, a), for all s belongs to S+, a belongs to A(s), arbitarily except that Q(terminal, .) = 0
Q = np.zeros((env.observation_space.n, env.action_space.n))

score_list = []
timesteps = []
# Loop for each episode
for episode in range(num_episodes):
	# Initialize S
	state = env.reset()
	# Decaying epsilon
	epsilon = get_epsilon(episode, num_episodes)
	# Choose A from S using policy derived from Q (epsilon greedy)
	action = epsilon_greedy_action(Q, state, epsilon)
	
	# Loop for each step of episode
	for t in range(100):
		# env.render()
		# Take action A, observe R and S`
		observation, reward, done, info = env.step(action)
		# Choose A` from S` using policy derived from Q (epsilon greedy)
		next_action = epsilon_greedy_action(Q, observation, epsilon)
		# Q(S, A) = Q(S, A) + alpha[R + gamma*Q(S`, A`) - Q(S, A)]
		Q[state, action] = Q[state, action] + alpha*(reward + (gamma*Q[observation, next_action]) - Q[state, action])
		# S <- S`
		# A <- A`
		state = observation
		action = next_action
		# Terminal State
		if(done):
			print("Episode finished after {} timesteps".format(t+1))
			score_list.append(reward)
			timesteps.append(t)
			print("Reward: " + str(reward))
			# print(Q)
			break


performance_list = []
for i in range(num_episodes):
	if(i%100 == 0):
		if(i != 0):
			performance_list.append(sum(temp_score)/100.0)
		temp_score = [score_list[i]]
	else:
		temp_score.append(score_list[i])


print("Printing Q-table: ")
print(Q)
plt.plot(performance_list)
# plt.plot(score_list)
# plt.plot(timesteps)
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.savefig('FrozenLake-v0_sarsa.png')
plt.show()
env.close()