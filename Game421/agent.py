import torch
import random
import numpy as np
from src.model import Linear_QNet
from src.agent import Agent
from collections import Counter


class Agent421(Agent):

    def __init__(self, lr=0.001, model=Linear_QNet(3, 256, 43), alpha=40):
        super().__init__(model, lr)
        self.state = [0,0,0]
        self.nb_play = 10
        self.alpha = alpha

        

    def get_reward(self, score):
        if isinstance(score, str):
            return 0
        else:
            return score*10

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon =  self.alpha - self.n_games
        opt = [[0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0],
        [1,1], [1,2], [1,3], [1,4], [1,5], [1,6],
        [2,1], [2,2], [2,3], [2,4], [2,5], [2,6],
        [3,1], [3,2], [3,3], [3,4], [3,5], [3,6],
        [4,1], [4,2], [4,3], [4,4], [4,5], [4,6],
        [5,1], [5,2], [5,3], [5,4], [5,5], [5,6],
        [6,1], [6,2], [6,3], [6,4], [6,5], [6,6]
        ]
        if random.randint(0, 200) < self.epsilon:
            rand = random.randint(0,42)
            strat = opt[rand]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            rand = torch.argmax(prediction).item()
            strat = opt[rand]
        
        return strat

    def train(self, ai):
        keeped = []
        for i in range(ai.NB_ROLLS):
            state_old = self.state
            strat = self.get_action(state_old) 
            roll, score, keeped, combi = ai.play(keeped, strat, i)
            if combi or i==2: 
                reward = self.get_reward(score)
                self.state = roll
                state_new = self.state
                self.train_short_memory(state_old, strat, reward, state_new, combi) # train short memory
                self.remember(state_old, strat, reward, state_new, combi) # remember
                break
        return score

        
    def training(self, ai, record):
        """ State => action => play => reward => new_state """

        for game in range(self.nb_play):
            self.train(ai)

        self.n_games += 1
        self.train_long_memory()

        """ Save record and model """
        if ai.total > record:
            record = ai.total

        self.model.save('AI421.pth')

        return record

    def train_human(self, ai):
        score = self.train(ai)
        self.n_games += 1
        self.train_long_memory()
        return score




