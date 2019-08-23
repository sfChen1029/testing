import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.90
EPISODES = 6000
SHOW_EVERY = 500
#DISCRETE_SIZE = [10, 10, 30, 30]


epsilon = 0.8 #how much we want to explore/try random actions
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

def get_discrete_state(state): #state = [x, x_dot, theta, theta_dot]
    theta_units = ((env.observation_space.high.astype('float64') - env.observation_space.low.astype('float64'))/30)[2]
    discrete_theta = (state.astype('float64') - env.observation_space.low.astype('float64'))[2]/theta_units
    theta_dot = max(-3, min(state[3], 3))
    theta_dot_units = (3 - (- 3))/12
    discrete_theta_dot = (theta_dot -(-3))/theta_dot_units
    discrete_state = [int(discrete_theta), int(discrete_theta_dot)]
    return tuple(discrete_state)

#initializing Q table
q_table = np.random.uniform(low=0, high=2, size = ([30, 12]+ [env.action_space.n])) # 10x20x2

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
    episode_reward = 0
    if episode%SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    state = env.reset()
    discrete_state = get_discrete_state(state)
    #print(state)

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)

        updated_state, reward, done, info = env.step(action) # take a random action
        episode_reward += reward
        updated_discrete_state = get_discrete_state(updated_state)

        #reward = 24 - abs(updated_state[2])

        if not done:
            future_q = np.max(q_table[updated_discrete_state])
            q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE)*q + LEARNING_RATE*(reward + DISCOUNT*future_q)
            q_table[discrete_state + (action, )] = new_q

        discrete_state = updated_discrete_state
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon-=epsilon_decay_value

        if render:
            env.render()
            #print(updated_discrete_state)

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
