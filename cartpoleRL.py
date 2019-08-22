import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500
DISCRETE_SIZE = [10, 10, 10, 20]


epsilon = 0.5 #how much we want to explore/try random actions
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

def get_discrete_state(state): #state = [x, x_dot, theta, theta_dot]
    units = ((env.observation_space.high.astype('float64') - env.observation_space.low.astype('float64'))/DISCRETE_SIZE)[2:]
    discrete_state = (state.astype('float64') - env.observation_space.low.astype('float64'))[2:]/units
    return tuple(discrete_state.astype(np.int))

#initializing Q table
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_SIZE + [env.action_space.n]))
observation = env.reset()


done = False
while not done:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(get_discrete_state(observation))
env.close()
