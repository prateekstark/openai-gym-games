import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from tiles3 import IHT, tiles

# Episodic Semi-gradient Sarsa Implementation (on-policy control)
NUM_TILES = 16
MAX_SIZE = 2**16

iht = IHT(MAX_SIZE)

def get_epsilon(episode, num_episodes):
	if(episode < 1000):
		return 1.0 - (episode)/1000.0
	else:
		return 0.05

def q(hashtable, state, action, weights):
	active_tiles = tiles(hashtable, NUM_TILES, state, [action])
	return sum(weights[active_tiles])

def epsilon_greedy_action(hashtable, weights, state, e):
	if(np.random.uniform(0, 1) < e):
		action = env.action_space.sample()
	else:
		state_action_values = []
		for a in [0, 1, 2, 3]:
			state_action_values.append(q(hashtable, state, a, weights))
		action = np.argmax(state_action_values)
		# print(action)
	return action

# Algorithm parameters: step size alpha from (0, 1], small epsilon > 0

alpha = 0.1/8
epsilon = 0.1
num_episodes = 500
gamma = 0.95
beta = 0.05
n = 2
env = gym.make('LunarLander-v2')
env._max_episode_steps = 10000

# Initialize value function weights w belong to R^d arbitarily
# w = np.zeros((2, 3))
w = np.zeros(MAX_SIZE)
score_list = []
timesteps = []
print(env.observation_space.low)
print(env.observation_space.high)

Ravg = 0.0
# Loop for each episode
for episode in range(num_episodes):
	statelist = []
	actionlist = []
	rewardlist = []

	# initialize state S0 and action A0
	state = env.reset()
	action = epsilon_greedy_action(iht, w, state, epsilon)
	statelist.append(state)
	actionlist.append(action)
	# epsilon = get_epsilon(episode, num_episodes)
	Ravg = 0.0
	# Loop for each step

	t = 0
	while(True):
		env.render()
		active_tiles = tiles(iht, NUM_TILES, state, [action])

		# Take action A, observe R and S`
		# Observe and store the next reward as Rt+1 and the next state as St+1
		state, reward, done, info = env.step(action)
		rewardlist.append(reward)
		statelist.append(state)
		# Select and store an action At+1 (epsilon greedy)
		action = epsilon_greedy_action(iht, w, state, epsilon)
		actionlist.append(action)
		
		tau = t-n+1
		if(tau >= 0):
			delta = 0
			for i in range(tau+1, tau+n+1):
				delta += rewardlist[i-1] - Ravg
			delta += q(iht, statelist[tau+n], actionlist[tau+n], w) - q(iht, statelist[tau], actionlist[tau], w)
			# Ravg = Ravg + beta*delta
			Ravg += beta*delta
			# w <- w + alpha*delta*delta(q(S[tau], A[tau], w))
			w[active_tiles] += alpha*delta
		
		# If S` is terminal
		if(done):
			score_list.append(sum(rewardlist))
			timesteps.append(t)
			print("Episode: " + str(episode))
			print("Episode finished after {} timesteps".format(t+1))
			print("Reward: " + str(sum(rewardlist)))
			if(episode%1000 == 0):
				np.save('weights', w)
				np.save('timesteps', timesteps)
			# Go to the next episode
			break
		t += 1
						

performance_list = []
for i in range(num_episodes):
	if(i%10 == 0):
		if(i != 0):
			performance_list.append(sum(temp_score)/10.0)
		temp_score = [score_list[i]]
	else:
		temp_score.append(score_list[i])



# score_list = np.array(score_list)
# np.save('score_list.pkl', score_list)
plt.plot(performance_list)
# plt.plot(score_list)
# plt.plot(timesteps)
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.savefig('LunarLander-v2_differential_semi_gradient_2step_sarsa_linear4.png')
plt.show()
env.close()