import torch
import random
import numpy as np
from src.model import Linear_QNet
from src.agent import Agent


class Agent5000(Agent):

    def __init__(self, lr=0.001, model=Linear_QNet(1, 256, 2), alpha=80):
        super().__init__(model, lr)
        self.alpha = alpha
        self.n_games = 0


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon =  self.alpha - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0,1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            print(prediction)
            action = torch.argmax(prediction).item()
        
        return action

    def training(self, game) :
        stats ={'Nombre tour': 0, 'result': [], 'reward': [], 'action': []}
        score = 0
        nbDeTour = 0
        win = False 
        while score < 5000 and not win:
            print('Nouveau tour ! Total :', score)
            nbDeTour += 1
            game.keep = []
            game.dice_left = 5
            end = False
            reward = 0
            action = 1
            main_pleine = False
            while (action and not end) or main_pleine:
                state_old = [game.dice_left]
                action = self.get_action(state_old) 
                result, reward, main_pleine, end = game.play(reward)
                state_new = [game.dice_left]
                self.train_short_memory(state_old, action, reward, state_new, end) # train short memory
                self.remember(state_old, action, reward, state_new, end) # remember
                stats['result'].append(result)
                stats['action'].append(action)
                stats['reward'].append(reward)
            score += reward
            if score > 5000:
                score -= reward
            if score == 5000:
                win = True
                print('WIN ! en ', nbDeTour)
        stats['Nombre tour'] += nbDeTour
        self.n_games += 1
        self.model.save('AI5000.pth')
        return stats




