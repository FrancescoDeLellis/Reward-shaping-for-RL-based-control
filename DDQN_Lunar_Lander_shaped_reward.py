import numpy as np
import gym
import tensorflow  as tf
from keras import backend as K
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import pickle

class Agent:
    def __init__(self, env, optimizer, batch_size):
        self.state_size = env.observation_space.shape[0]  
        self.action_size = env.action_space.n
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.omega = 0.8

        # experience replay
        self.replay_exp = deque(maxlen=100000)

        self.gamma = 0.99   # discount factor 
        self.epsilon = 0.1  # exploration

        # Build Policy Network
        self.brain_policy = tf.keras.models.Sequential()
        self.brain_policy.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation="relu"))
        self.brain_policy.add(tf.keras.layers.Dense(128, activation="relu"))
        self.brain_policy.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        self.brain_policy.compile(loss="mse", optimizer=self.optimizer)

        # Build Target Network
        self.brain_target = tf.keras.models.Sequential()
        self.brain_target.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation="relu"))
        self.brain_target.add(tf.keras.layers.Dense(128, activation="relu"))
        self.brain_target.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        self.brain_target.compile(loss="mse", optimizer=self.optimizer)

        self.update_brain_target()

        # set goal region to stabilize
        self.x_goal = 0.2
        self.y_goal = 0.002

        self.y_vel_goal = 0.0005
        self.x_vel_goal = 0.0000001 

        # set desired settling time (maximum number of steps required to reach goal region)
        self.ks = 500  
        # set out time horizon
        self.kout = 1000  

        # estimation of bounds of the reward function in and outside the goal region
        self.r_max = 100
        self.r_min = - 100
        self.r_G_max = 100
        self.r_G_min = - 100

        # set objective value treshold (modulate to fulfill Corollary IV.7)
        self.sigma = 12000

        # calulate prize and punishment according to Theorem IV.5 (Assumption IV.3)
        self.prize = self.sigma * (1 - self.gamma) / self.gamma ** (self.ks) - self.r_max * (1 - self.gamma ** (self.ks)) / self.gamma ** (self.ks) - self.r_G_max

        self.punishment = self.sigma * self.gamma ** (-self.kout) + \
                          self.r_max / (1 - self.gamma) - \
                          (self.r_G_max + self.prize) * (self.gamma ** (-self.kout) - 1) / (1 - self.gamma)

    # add new experience to the replay exp
    def memorize_exp(self, state, action, reward, next_state, done):
        self.replay_exp.append((state, action, reward, next_state, done))

    def update_brain_target(self):
        return self.brain_target.set_weights(self.brain_policy.get_weights())

    # Choosing action according to epsilon-greedy policy
    def choose_action_DQN(self, state, flag):
        state = np.reshape(state, [1, self.state_size])
        qhat = self.brain_policy.predict(state, verbose=0) 
        action = np.argmax(qhat[0])

        random = np.random.random()
        if flag == 0:
            if random > self.epsilon:
                return action
            else:
                return np.random.choice(self.action_size)
        else:
            return action

    # Deploy action according to a learned policy
    def deploy_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        qhat = self.brain_target.predict(state, verbose=0)
        return np.argmax(qhat[0])

    # Update parameters of neural policy
    def learn(self):
        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_exp), self.batch_size)
        mini_batch = random.sample(self.replay_exp, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, self.state_size))  # replace 128 with cur_batch_size
        sample_actions = np.ndarray(shape=(cur_batch_size, 1))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, self.state_size))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1
        
        sample_qhat_next = self.brain_target.predict(sample_next_states, verbose=0)

        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.brain_policy.predict(sample_states, verbose=0)
        
        for i in range(cur_batch_size):
            a = sample_actions[i, 0]
            sample_qhat[i, int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat

        self.brain_policy.fit(sample_states, q_target, epochs=1, verbose=0)

    # apply reward correction according to Assumption IV.5
    def reward_wrapper(self, reward, state, state_next, terminated):
        flag = 0
        next_cond = np.abs(state_next[0]) <= self.x_goal and np.abs(state_next[1]) <= self.y_goal and terminated 
        cond = np.abs(state[0]) <= self.x_goal and np.abs(state[1]) <= self.y_goal and terminated
        if next_cond == True:
            reward += self.prize
            flag = 1
        elif cond == True and next_cond == False:
            reward += self.punishment
            flag = 2

        return reward, flag


if __name__ == "__main__":
    # load Lunar Lander environment
    env = gym.make("LunarLander-v2", continuous = False, gravity = -10.0, enable_wind = False, wind_power = 0, turbulence_power = 0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    agent = Agent(env, optimizer, batch_size=128)

    rewards = np.zeros([1001])
    discounted_rewards = np.zeros([1001])
    aver_reward = []
    aver = deque(maxlen=100)

    state_size = env.observation_space.shape[0]

    start_time = time.time()

    Episodes = 1001

    for episode in range(0, Episodes):
        timestep = 0
        state, info = env.reset()
        total_reward = 0
        done = False
        time_in = 0
        exit_times = 0
        terminated = 0
        disc_tot_reward = 0

        while not done:
            action = agent.choose_action_DQN(state, terminated)
            next_state, reward, terminated, truncated, info = env.step(action)

            reward, in_flag = agent.reward_wrapper(reward, state, next_state, terminated)

            if in_flag == 1:
                # Lunar Lander enterd the goal region
                time_in += 1
                done = timestep == agent.kout - 1
            elif in_flag == 2:
                # Lunar Lander exited the goal region
                done = 1
                exit_times += 1
            else:
                done = terminated == 1 or truncated == 1

            agent.memorize_exp(state, action, reward, next_state, done)
            total_reward += reward
            disc_tot_reward += agent.gamma ** timestep * reward

            agent.learn()

            state = next_state
            timestep += 1

        aver.append(total_reward)
        aver_reward.append(np.mean(aver))

        rewards[episode] = total_reward
        discounted_rewards[episode] = disc_tot_reward

        # update model_target after each episode
        agent.update_brain_target()
        
        print("Episode {0} | reward {1} | steps {2} | time in goal {3} | exited {4} | disc reward {5}".format(episode, total_reward, timestep, 
                                                                                                              time_in, exit_times, discounted_rewards[episode]))

        # save checkpoints of policies which achieve objective over the threshold sigma
        if discounted_rewards[episode] >= agent.sigma:
            np.save("./Data/reward" + str(episode) + ".npy", rewards)
            np.save("./Data/discounted_reward" + str(episode) + ".npy", discounted_rewards)
            pickle.dump(agent.replay_exp, open('./Data/buffer' + str(episode) + '.pkl', 'wb'))
            agent.brain_policy.save("./Data/brain_policy" + str(episode) + ".h5")
            agent.brain_target.save("./Data/target_policy" + str(episode) + ".h5")

    env.close()

    np.save("./Data/reward" + str(episode) + ".npy", rewards)
    np.save("./Data/discounted_reward" + str(episode) + ".npy", discounted_rewards)
    np.save("./Data/aver_reward" + str(episode) + ".npy", aver_reward)
    pickle.dump(agent.replay_exp, open('./Data/buffer' + str(episode) + '.pkl', 'wb'))
    agent.brain_policy.save("./Data/policy" + str(episode) + ".h5")

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards)
    plt.plot(aver_reward, 'r')

    plt.show()
