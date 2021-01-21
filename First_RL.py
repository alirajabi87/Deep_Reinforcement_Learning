import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

Learning_rate = 0.1
Discount = 0.95  # How the agent look at the future reward to return to the current reward
Episodes = 8000
show_every = 1000

# between 0 - 1, higher epsilon higher chance to make random action
epsilon = 0.5
START_epsilon_decaying = 1
END_epsilon_decaying = Episodes // 2
epsilon_decay_value = epsilon / (END_epsilon_decaying - START_epsilon_decaying)

Discrete_OS_Size = [40] * len(env.observation_space.high)
Discrete_OS_Win_Size = (env.observation_space.high - env.observation_space.low) / Discrete_OS_Size

q_table = np.random.uniform(low=-2, high=0, size=(Discrete_OS_Size + [env.action_space.n]))

ep_rewards = []
aggr_rewards = {'episodeNo': [], 'avg': [], 'min': [], 'max': []}


# print(q_table.shape)

# The env.step returns a continuous state we have to convert it to discrete state in Q-table
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / Discrete_OS_Win_Size
    return tuple(discrete_state.astype(np.int))


for episode in range(Episodes):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False
    i = 0
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, info = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - Learning_rate) * current_q + Learning_rate * (reward + Discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            # print(f"We made it on episode: {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if END_epsilon_decaying >= episode >= START_epsilon_decaying:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    STATS_EVERY = show_every
    if not episode % STATS_EVERY:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / len(ep_rewards[-STATS_EVERY:])
        aggr_rewards['episodeNo'].append(episode)
        aggr_rewards['avg'].append(average_reward)
        aggr_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))

        print(f" Episode: {episode:>5d}, Average: {average_reward:>4.1f}, current epsilon:{epsilon:>1.2f}")
        # min: {min(ep_rewards[-show_every:])}, max: {max(ep_rewards[-show_every:])}")

env.close()
plt.plot(aggr_rewards['episodeNo'], aggr_rewards['avg'], label='avg')
plt.plot(aggr_rewards['episodeNo'], aggr_rewards['min'], label='min')
plt.plot(aggr_rewards['episodeNo'], aggr_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
