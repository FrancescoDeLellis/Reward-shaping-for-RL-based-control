import gym
import numpy as np
import math
from gym import spaces
import matplotlib.pyplot as plt
import timeit

class Pole:
    def __init__(self, buckets=(41, 39,), n_episodes=1000, nstep_episode=1000, gamma=0.99, alpha=0.8, epsilon=0.05):

        self.folder = 'Data'
        self.sets = 2
        self.buckets = buckets
        self.n_episodes = n_episodes
        self.nstep_episode = nstep_episode
        self.r = np.zeros((n_episodes, nstep_episode))
        self.e = 0
        self.stabilized = np.zeros((self.n_episodes, self.sets))
        self.tutor = np.zeros((self.n_episodes, self.sets))
        self.thresholds = np.array([10, 15, 20, 30, 40, 50, 100])
        self.lambda3 = np.zeros([self.sets, len(self.thresholds)])
        self.counter = 0
        self.d_max = np.sqrt(np.pi**2 + 8**2)

        self.actions = np.array([-2.0, -1.6, -1.2, -0.8, -0.4, -0.2, -0.05, 0.0, 0.05, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
        self.n_actions = len(self.actions)
        self.index = spaces.Discrete(self.n_actions)

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make('Pendulum-v1')

        self.state_table = np.reshape(list(range(0, self.buckets[0]*self.buckets[1])), [self.buckets[0], self.buckets[1]])

        self.Xbuf = np.zeros([self.nstep_episode * self.n_episodes, self.n_actions])
        self.ybuf = np.zeros([self.nstep_episode * self.n_episodes,1])

        # reward shaping
        self.theta = 0.05 * np.linalg.norm([8, np.pi])
        self.ks = 500
        self.kout = 1000

        self.r_max = - 0.1 * self.theta ** 2
        self.r_min = - np.pi ** 2 - 0.1 * 8 ** 2 - 0.001 * 2 ** 2
        self.r_G_max = 0
        self.r_G_min = - self.theta ** 2 - 0.1 * 8 ** 2 - 0.001 * 2 ** 2

        self.sigma = 10000

        self.constant = 0

        self.prize = self.sigma * (1 - self.gamma) * self.gamma ** (-self.ks-1) + \
                     self.r_max * (1 - self.gamma ** (-self.ks-1)) - \
                     self.r_G_max

        self.punishment = self.sigma * self.gamma ** (-self.kout) + \
                          self.r_max / (1 - self.gamma) - \
                          (self.r_G_max + self.prize) * (self.gamma ** (-self.kout) - 1) / (1 - self.gamma)

        self.Q = self.sigma * np.zeros(self.buckets + (self.n_actions,))

    def discretize_non_uniformly(self, obs):
        th = obs[0]
        thd = obs[1]
        new_obs = np.array([0, 0])

        if -math.pi <= th <= -math.pi / 9:
            upper_bound = -math.pi / 9
            lower_bound = -math.pi
            ratio = (th + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[0] = int(round(8 * ratio))
            new_obs[0] = min(8, max(0, new_obs[0]))

        elif -math.pi / 9 < th <= -math.pi / 36:
            upper_bound = -math.pi / 36
            lower_bound = -math.pi / 9
            ratio = (th + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[0] = int(round(7 * ratio))
            new_obs[0] = min(7, max(0, new_obs[0])) + 8

        elif -math.pi / 36 < th <= math.pi / 36:
            upper_bound = math.pi / 36
            lower_bound = -math.pi / 36
            ratio = (th + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[0] = int(round(10 * ratio))
            new_obs[0] = min(10, max(0, new_obs[0])) + 15

        elif math.pi / 36 < th <= math.pi / 9:
            upper_bound = math.pi / 9
            lower_bound = math.pi / 36
            ratio = (th + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[0] = int(round(7 * ratio))
            new_obs[0] = min(7, max(0, new_obs[0])) + 25

        elif math.pi / 9 < th <= math.pi:
            upper_bound = math.pi
            lower_bound = math.pi / 9
            ratio = (th + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[0] = int(round(8 * ratio))
            new_obs[0] = min(8, max(0, new_obs[0])) + 32

            if new_obs[0] == 40:
                new_obs[0] = 0

        elif th == 0:
            new_obs[0] = 40

        if -8 <= thd <= -1:
            upper_bound = -1
            lower_bound = -8
            ratio = (thd + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[1] = int(round(9 * ratio))
            new_obs[1] = min(9, max(0, new_obs[1]))
        elif -1 < thd <= 1:
            upper_bound = 1
            lower_bound = -1
            ratio = (thd + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[1] = int(round(19 * ratio))
            new_obs[1] = min(19, max(0, new_obs[1])) + 9
        elif 1 < thd <= 8:
            upper_bound = 8
            lower_bound = 1
            ratio = (thd + abs(lower_bound)) / (upper_bound - lower_bound)
            new_obs[1] = int(round(9 * ratio))
            new_obs[1] = min(9, max(0, new_obs[1])) + 28

        elif thd == 0:
            new_obs[0] = 38

        return tuple(new_obs)

    def discretize_uniformly(self, obs):
        upper_bounds = [math.pi, 8]
        lower_bounds = [-math.pi, -8]

        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

        return tuple(new_obs)

    def choose_index_QL(self, state):
        ind = self.index.sample()
        a = np.argmax(self.Q[state])
        random = np.random.random()
	
        if random <= self.epsilon:
            return ind
        else:
            return a

    def update_q(self, state_old, ind, reward, state_new, alpha):
        self.Q[state_old][ind] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][ind])

    def run(self, conds=np.array([np.pi, 0])):
        for ii in range(self.sets):
            p = 0 * self.thresholds
            while self.e < self.n_episodes:
                # state reset:
                obs = self.env.reset()[0]

                obs2 = np.array([0.0, 0.0])
                obs2[0] = math.atan2(obs[1], obs[0])
                obs2[1] = obs[2]
                obs_pre = obs2

                current_state = self.discretize_uniformly(obs2)

                disc_rewrad = 0
                sum1 = 0

                for i in range(self.nstep_episode):

                    # self.env.render()

                    ind = self.choose_index_QL(current_state)
                    action = self.actions[ind]
                    out = self.env.step((action,))

                    obs = out[0]
                    reward = out[1]

                    obs2 = np.array([0.0, 0.0])
                    obs2[0] = math.atan2(obs[1], obs[0])
                    obs2[1] = obs[2]
                    new_state = self.discretize_non_uniformly(obs2)
                    if np.linalg.norm(obs2) < self.theta:
                        sum1 += 1
                        reward += self.prize
                    else:
                        sum1 = 0

                    if np.linalg.norm(obs_pre) <= self.theta < np.linalg.norm(obs2):
                        reward += self.punishment

                    disc_rewrad += self.gamma ** i * reward

                    self.update_q(current_state, ind, reward, new_state, self.alpha)
                    current_state = new_state
                    obs_pre = obs2
                self.ts = self.nstep_episode - sum1
                if sum1 > 100:
                    self.counter += 1
                else:
                    self.counter = 0
                print('ciclo for n: ', str(ii + 1), 'episodio: ', str(self.e + 1), 'discounted_reward: ', str(disc_rewrad), 'Ts: ', str(self.nstep_episode - sum1), 'counter: ',
                      self.counter)
                self.r[self.e, ii] = disc_rewrad
                self.stabilized[self.e, ii] = sum1
                self.e += 1
                for jj in range(0, len(self.thresholds)):
                    if self.counter == self.thresholds[jj] and p[jj] == 0:
                        p[jj] = 1
                        self.lambda3[ii, jj] = self.e - 1
                        np.save(self.folder + '/Tables/Q_CTQL_' + str(ii + 1) + '_' + str(jj) + '_.npy', solver.Q)
                        print('fine a episodio:', self.lambda3[ii, jj])
            print('end of training ', str(self.e))
            np.save(self.folder + '/Tables/Q_CTQL_' + str(ii + 1) + '_Final_1_.npy', solver.Q)
            self.e = 0
            self.Q = 0 * self.Q + self.sigma

    def set_table(self, table):
        self.Q = table

if __name__ == "__main__":
    start = timeit.default_timer()
    solver = Pole()
    solver.run()
    avg_r = np.mean(solver.r, 1)
    std_r = np.std(solver.r, 1)

    plt.figure(1)
    plt.title('Reward')
    plt.plot(range(0, solver.n_episodes), avg_r, 'b-')
    plt.fill_between(range(0, solver.n_episodes), avg_r - std_r, avg_r + std_r, facecolor='blue', alpha=0.2)
    plt.xlim([0, solver.n_episodes])

    np.save(solver.folder + '/R_CTQL.npy', solver.r)
    np.save(solver.folder + '/T_CTQL.npy', solver.tutor)
    np.save(solver.folder + '/lambda3.npy', solver.lambda3)

    lambda1 = np.max(avg_r)
    lambda2 = np.mean(avg_r)

    print('stats:', lambda1, ' , ', lambda2, np.mean)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
