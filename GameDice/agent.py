import torch
import random
import numpy as np
from src.model import Linear_QNet
from src.agent import Agent


class AgentDice(Agent):

    def __init__(self, lr, model=Linear_QNet(3, 256, 6), alpha=50):
        super().__init__(model, lr)
        self.state = [0,0,0]
        self.nb_play = 20
        self.alpha = alpha

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon =  self.alpha - self.n_games
        keep = [1,2,3,4,5,6]
        if random.randint(0, 200) < self.epsilon:
            rand = random.randint(0,5)
            action = keep[rand]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            rand = torch.argmax(prediction).item()
            action = keep[rand]
            print(prediction)

        return action

    def training(self, ai):
        """ State => action => play => reward => new_state """

        # Action and Play
        state_old = self.state
        action = self.get_action(state_old) 
        roll, reward, keep, done = ai.play(action)
        self.state = roll
        state_new = self.state
        self.train_short_memory(state_old, action, reward, state_new, done) # train short memory
        self.remember(state_old, action, reward, state_new, done) # remember

        self.n_games += 1
        self.train_long_memory()

        self.model.save('Dice421.pth')

        return reward, keep



