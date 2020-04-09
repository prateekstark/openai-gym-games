import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from tiles3 import IHT, tiles

# Episodic Semi-gradient Sarsa Implementation (on-policy control)
NUM_TILES = 16
MAX_SIZE = 2**15

iht = IHT(MAX_SIZE)

def get_epsilon(episode, num_episodes):
	if(episode < 480):
		return 1.0 - (episode)/500.0
	else:
		return 0.05

def q(hashtable, state, action, weights):
	active_tiles = tiles(hashtable, NUM_TILES, np.multiply(state, [1.5, 1.5, 1.5, 1.5, 1.5/12.566371, 1.5/28.274334]), [action])
	return sum(weights[active_tiles])

def epsilon_greedy_action(hashtable, weights, state, e):
	if(np.random.uniform(0, 1) < e):
		action = env.action_space.sample()
	else:
		state_action_values = []
		for a in [0, 1, 2]:
			state_action_values.append(q(hashtable, state, a, weights))
		action = np.argmax(state_action_values)
		# print(action)
	return action

# Algorithm parameters: step size alpha from (0, 1], small epsilon > 0

alpha = 0.1/8
epsilon = 0.1
num_episodes = 1200
gamma = 0.95
n = 4

env = gym.make('Acrobot-v1')
env._max_episode_steps = 10000

# Initialize value function weights w belong to R^d arbitarily
w = np.zeros(MAX_SIZE)
score_list = []
timesteps = []
# print(env.observation_space.low)
# print(env.observation_space.high)
print(env.action_space)


# Loop for each episode
for episode in range(num_episodes):
	# Initialize and store S0 != terminal
	statelist = []
	actionlist = []
	rewardlist = []
	state = env.reset()
	statelist.append(state)
	epsilon = get_epsilon(episode, num_episodes)
	# print(epsilon)
	# Select and store an action A0 (epsilon greedy wrt q(S0, ., w))
	action = epsilon_greedy_action(iht, w, state, epsilon)
	actionlist.append(action)
	# T <- 'inf'
	T = 1000000000
	# Loop for each step of episode
	t = 0
	while(True):
		env.render()
		
		# if t < T, then:
			# Take action At
			# Observe and store the next reward as R(t+1) and the next state as S(t+1)
			# If S(t+1) is terminal, then:
				# T <- t+1
			# else:
				# Select and store A(t+1) or epsilon-greedy wrt q(S(t+1), ., w)
		if(t < T):
			observation, reward, done, info = env.step(action)
			statelist.append(observation)
			rewardlist.append(reward)
			if(done):
				T = t+1
			else:
				action = epsilon_greedy_action(iht, w, observation, epsilon)
				actionlist.append(action)

		# tau <- t - n + 1 (tau is the time whose estimate is being updated)
		tau = t - n + 1
		# If tau >= 0:
		if(tau >= 0):
			# G <- sum from i = tau+1 to min(tau+n, T) (gamma**(i-tau-1)*R(i))
			G = 0.0
			for i in range(tau+1, min(tau+n, T)+1):
				G += (gamma**(i-tau-1))*rewardlist[i-1] #some contoversy
			
			# If tau + n < T, then G <- G + (gamma**n)*(q(S(tau+n, A(tau+n), w)))
			if(tau + n < T):
				G += (gamma**(n))*q(iht, statelist[tau+n], actionlist[tau+n], w)
			# w <- w + alpha[G - q(S(tau), A(tau), w)]*delta(q(S(tau), A(tau), q))
			active_tiles = tiles(iht, NUM_TILES, np.multiply(statelist[tau], [1.5, 1.5, 1.5, 1.5, 1.5/12.566371, 1.5/28.274334]), [actionlist[tau]])
			w[active_tiles] += alpha*(G - q(iht, statelist[tau], actionlist[tau], w))
		
		# Until tau = T-1
		if(tau == T-1):
			score_list.append(reward)
			timesteps.append(t)
			print("Episode: " + str(episode))
			print("Episode finished after {} timesteps".format(t+1))
			# print("Reward: " + str(reward))
			if(episode%100 == 0):
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
		temp_score = [timesteps[i]]
	else:
		temp_score.append(timesteps[i])

plt.plot(performance_list)
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.savefig('Acrobot-v1_semi_gradient_4step_sarsa_linear.png')
plt.show()
env.close()