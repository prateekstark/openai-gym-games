import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from tiles3 import IHT, tiles

# Episodic Semi-gradient Sarsa Implementation (on-policy control)
NUM_TILES = 8
MAX_SIZE = 2**12
iht = IHT(MAX_SIZE)

def get_epsilon(episode, num_episodes):
	if(episode < num_episodes):
		return 1.0 - (episode)/1000.0
	else:
		return 0.05

def q(hashtable, state, action, weights):
	active_tiles = tiles(hashtable, NUM_TILES, np.multiply(state, [1, 1, 1, 1]), [action])
	return sum(weights[active_tiles])

def epsilon_greedy_action(hashtable, weights, state, e):
	if(np.random.uniform(0, 1) < e):
		action = env.action_space.sample()
	else:
		state_action_values = []
		for a in [0, 1]:
			state_action_values.append(q(hashtable, state, a, weights))
		action = np.argmax(state_action_values)
		# print(action)
	return action

# Algorithm parameters: step size alpha from (0, 1], small epsilon > 0

alpha = 0.1/8
epsilon = 0.1
num_episodes = 2000
gamma = 0.95


env = gym.make('CartPole-v1')
env._max_episode_steps = 10000

# Initialize value function weights w belong to R^d arbitarily
# w = np.zeros((2, 3))
w = np.zeros(MAX_SIZE)
score_list = []
timesteps = []
# print(env.observation_space.low)
# print(env.observation_space.high)

# Loop for each episode
for episode in range(num_episodes):
	state = env.reset()
	# S, A <- initial state and action of episode (epsilon greedy)
	# epsilon = get_epsilon(episode, num_episodes)
	if(episode > 1000):
		epsilon = 0.05
	action = epsilon_greedy_action(iht, w, state, epsilon)

	# Loop for each step of episode
	t = 0
	while(True):
		env.render()
		# print(state)
		# print(state.shape)
		# print(np.multiply(state, [[10], [100]]))
		active_tiles = tiles(iht, NUM_TILES, state, [action])

		# Take action A, observe R and S`
		observation, reward, done, info = env.step(action)
		
		# If S` is terminal
		if(done):
			# w <- w + alpha[R + gamma*q(S`, A`, w) - q(S, A, w)]delta(q(S, A, w))
			w[active_tiles] += alpha*(reward - q(iht, np.multiply(state, [1, 1, 1, 1]), action, w))
			score_list.append(reward)
			timesteps.append(t)
			print("Episode: " + str(episode))
			print("Episode finished after {} timesteps".format(t+1))
			print("Reward: " + str(reward))
			if(episode%1000 == 0):
				np.save('weights', w)
				np.save('timesteps', timesteps)
			# Go to the next episode
			break
	
		# Choose A` as a function of q(S`, ., w) (epsilon greedy)
		action_next = epsilon_greedy_action(iht, w, observation, epsilon)
		# print(w)
		# w <- w + alpha[R + gamma*q(S`, A`, w) - q(S, A, w)]delta(q(S, A, w))
		w[active_tiles] += alpha*(reward + gamma*q(iht, observation, action_next, w) - q(iht, state, action, w))
		# print(reward)
		# S <- S`
		# A <- A`
		state = observation
		action = action_next
		t += 1
						

performance_list = []
for i in range(num_episodes):
	if(i%10 == 0):
		if(i != 0):
			performance_list.append(sum(temp_score)/10.0)
		temp_score = [timesteps[i]]
	else:
		temp_score.append(timesteps[i])



# score_list = np.array(score_list)
# np.save('score_list.pkl', score_list)
plt.plot(performance_list)
# plt.plot(score_list)
# plt.plot(timesteps)
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.savefig('CartPole-v1_semi_gradient_sarsa_linear1.png')
plt.show()
env.close()