import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")

#have Q value table where we look up what the 3 Qvalues, associated with each action at the specific state, are, then choose
#the highest one
#the Qvalue at s and a is the expected reward if you start in s, take actiona, and act optimally afterwards
LEARNING_RATE = 0.1
DISCOUNT = 0.95 #how important we find future reward
EPISODES = 2000
DISCRETE_SIZE = [20]*len(env.observation_space.high) #basically a 20x20 grid in this case
discrete_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE
SHOW_EVERY = 500

epsilon = 0.5 #how much we want to explore/try random actions
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

#initializing Q table
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_SIZE + [env.action_space.n]))  #20x20x3 table

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode%SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())#do env.reset this every time you have an environment, it returns the initial state
    #since discrete_state is a tuple, we can use that as an index for q_table, eg. q_table[discrete_state]
    #np.argmax(q_table[discrete_state]) gets the index of the best action for that state
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) #there are 3 actions in this environment: push car left, do nothing, push car right
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action) #have to convert continuous values of new_state to discrete values
        episode_reward += reward #tracking reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render() #show the car
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) #get current estimate of maximum future reward if you go to new state
            current_q = q_table[discrete_state + (action, )] # the q value for this state with this action
            new_q = (1 - LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            #print(f"we made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon-=epsilon_decay_value

    ep_rewards.append(episode_reward)
    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"EPISODE: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max{max(ep_rewards[-SHOW_EVERY:])}")

env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
