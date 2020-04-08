import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from tiles3 import IHT, tiles

# Episodic Semi-gradient Sarsa Implementation (on-policy control)
NUM_TILES = 8
MAX_SIZE = 1024

iht = IHT(MAX_SIZE)

def get_epsilon(episode, num_episodes):
	if(episode < num_episodes):
		# return math.exp(-2.4*episode/num_episodes)
		return 1.0 - (episode)/1000.0
	else:
		return 0.05

def q(hashtable, state, action, weights):
	active_tiles = tiles(hashtable, NUM_TILES, state.reshape(2, 1), [action])
	return sum(weights[active_tiles])

def epsilon_greedy_action(hashtable, weights, state, e):
	if(np.random.uniform(0, 1) < e):
		action = env.action_space.sample()
	else:
		state_action_values = np.array(map(lambda a: q(hashtable, state, a, weights), [0, 1, 2]))
		action = np.argmax(state_action_values)
	return action

# Algorithm parameters: step size alpha from (0, 1], small epsilon > 0

alpha = 0.1/8
epsilon = 0.1
num_episodes = 50000
gamma = 0.95


env = gym.make('MountainCar-v0')
env._max_episode_steps = 50000

# Initialize value function weights w belong to R^d arbitarily
# w = np.zeros((2, 3))

w = np.zeros(MAX_SIZE)
score_list = []
timesteps = []

# Loop for each episode
for episode in range(num_episodes):
	state = env.reset()
	# S, A <- initial state and action of episode (epsilon greedy)
	epsilon = get_epsilon(episode, num_episodes)
	action = epsilon_greedy_action(iht, w, state, epsilon)

	# Loop for each step of episode
	t = 0
	while(True):
		env.render()
		active_tiles = tiles(iht, NUM_TILES, state.reshape(2, 1), [action])

		# Take action A, observe R and S`
		observation, reward, done, info = env.step(action)
		
		# If S` is terminal
		if(done):

			# w <- w + alpha[R + gamma*q(S`, A`, w) - q(S, A, w)]delta(q(S, A, w))
			w[active_tiles] = w[active_tiles] + alpha*(reward - q(iht, state.reshape(2, 1), action, w))
			score_list.append(reward)
			timesteps.append(t)

			print("Episode: " + str(episode))
			print("Episode finished after {} timesteps".format(t+1))
			if(episode%1000 == 0):
				np.save('weights.pkl', w)
				np.save('timesteps.pkl', timesteps)
			# print("Reward: " + str(reward))
			# Go to the next episode
			break
	
		# Choose A` as a function of q(S`, ., w) (epsilon greedy)
		action_next = epsilon_greedy_action(iht, w, observation, epsilon)
		# print(w)
		# w <- w + alpha[R + gamma*q(S`, A`, w) - q(S, A, w)]delta(q(S, A, w))
		w[active_tiles] = w[active_tiles] + alpha*(reward + gamma*q(iht, observation.reshape(2, 1), action_next, w) - q(iht, state.reshape(2, 1), action, w))
		
		# S <- S`
		# A <- A`
		state = observation
		action = action_next
		t += 1
						

performance_list = []
for i in range(num_episodes):
	if(i%100 == 0):
		if(i != 0):
			performance_list.append(sum(timesteps)/100.0)
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
plt.savefig('MountainCar-v0_semi_gradient_sarsa_linear.png')
plt.show()
env.close()