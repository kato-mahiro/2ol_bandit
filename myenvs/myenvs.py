from re import I
import numpy as np
import gym
import random

"""
常に観測値として1を返す環境
環境に対して取るべき行動が周期的に切り替わり、
それに応じて報酬が決定される。
"""

class StaticCyclicEnv0(gym.Env):
    def __init__(self, cycle, cycle_cnt_max, action_num, noise):
        super().__init__()
        self.cycle = cycle
        assert self.cycle % action_num == 0 and self.cycle >= action_num, 'cycleはアクション数の整数倍でなくてはならない'

        self.action_num = action_num
        self.desired_action = -1
        self.cycle_cnt_max = cycle_cnt_max
        self.noise = noise
        self.action_space = gym.spaces.Discrete(self.action_num)
        self.observation_space = gym.spaces.Box(
            low = 1,
            high = 1,
            shape = [1],
            dtype = np.int
        )
        self.reward_range = [0, 1]
        self.reset()

    def reset(self):
        self.step_cnt = 0
        self.cycle_cnt = 0
        self.done = False
        self.info = {
                        'bonus_cnt':0, \
                        'bonus_max': self.cycle_cnt_max * self.action_num, #生涯中でボーナスが発生するタイミングの数 
                        'is_bonus': True #ルールが切り替わった直後および最初のステップでTrue
                    }
        return self.observe()

    def step(self, action):
        observation = self.observe()

        ### Step更新
        if(self.step_cnt % (self.cycle // self.action_num) == 0):
            if(random.random()) >= self.noise:
                self.update_action()
            self.info['is_bonus'] = True
        else:
            self.info['is_bonus'] = False

        self.step_cnt += 1

        ### rewardの計算
        if action == self.desired_action:
            reward = 1.0
        else:
            reward = 0.0

        ### 終了処理
        if(self.cycle_cnt >= self.cycle_cnt_max):
            self.done = True
        else:
            self.done = False

        return observation, reward, self.done, self.info

    def observe(self):
        return 1

    def update_action(self):
        self.desired_action += 1
        if(self.desired_action == self.action_num):
            self.desired_action = 0
            self.cycle_cnt += 1
